"""
evaluate_bart.py  —  BART eBART cross-network validation
Architektúra kapacitás mérés: random inicializált súlyok, BART méretű gráf.
Torch_geometric szükséges.
"""

import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config.paths as _p
from utils.bart_validation import (
    load_bart_delta,
    build_bart_graph,
    compute_metrics,
    nonzero_metrics,
    ebart_station_metrics,
    print_bart_results,
    save_results_csv,
)

BART_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bart")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUDAPEST_MAE = {"gat": 13.06, "hypergraph": 13.26}


def build_model(model_name: str, N: int, device: torch.device):
    """
    Frissen inicializált modell BART méretű gráfra (N állomás).
    Ugyanazok a hiperparaméterek mint a Budapest training — csak num_zones más.
    """
    if model_name == "gat":
        from models.gat_lstm import GATLSTM
        model = GATLSTM(
            in_channels=1,
            hidden_channels=64,
            lstm_hidden=128,
            num_zones=N,
        )
    else:
        from models.hypergraph_lstm import HypergraphLSTM
        model = HypergraphLSTM(
            in_channels=1,
            hidden_channels=64,
            lstm_hidden=128,
            num_zones=N,
        )
    model.to(device)
    model.eval()
    return model


def run_inference(model, graph: dict, device: torch.device) -> np.ndarray:
    """Egy forward pass a BART gráfon."""
    x          = graph["x"].to(device)           # [N, 1]
    edge_index = graph["edge_index"].to(device)  # [2, E]
    N          = graph["N"]

    with torch.no_grad():
        x_seq = x.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 1]
        try:
            pred = model(x_seq, edge_index)
        except TypeError:
            batch = torch.zeros(N, dtype=torch.long, device=device)
            pred = model(x_seq, edge_index, batch)

    pred_np = pred.squeeze().cpu().numpy()
    if pred_np.ndim == 1:
        side = int(np.sqrt(pred_np.size))
        if side * side == pred_np.size:
            pred_np = pred_np.reshape(side, side)
    return pred_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["gat", "hypergraph", "all"], default="all")
    parser.add_argument("--no-seasonal", action="store_true")
    parser.add_argument("--bart-dir",    default=BART_DATA_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. ΔOD betöltés ───────────────────────────────────────────────────────
    raw_delta, sea_delta = load_bart_delta(args.bart_dir)
    raw_delta.to_csv(os.path.join(RESULTS_DIR, "bart_delta_od_raw.csv"))
    sea_delta.to_csv(os.path.join(RESULTS_DIR, "bart_delta_od_seasonal.csv"))

    delta    = raw_delta if args.no_seasonal else sea_delta
    graph    = build_bart_graph(delta)
    true     = graph["delta_od"].numpy()
    stations = graph["station_names"]
    N        = graph["N"]
    print(f"\nBART gráf: {N} állomás, {graph['edge_index'].shape[1]} él")

    # Null modell baseline
    null_metrics = compute_metrics(np.zeros_like(true).flatten(), true.flatten())
    print(f"Null modell MAE: {null_metrics['MAE']:.4f}  RMSE: {null_metrics['RMSE']:.4f}")

    # ── 2. Modellek futtatása ──────────────────────────────────────────────────
    models_to_run = ["gat", "hypergraph"] if args.model == "all" else [args.model]
    all_results   = []

    for name in models_to_run:
        print(f"\n{'─'*60}")
        print(f"  {name.upper()} — random inicializált súlyok, BART méret ({N} zóna)")
        print(f"{'─'*60}")

        try:
            model   = build_model(name, N, device)
            pred_np = run_inference(model, graph, device)

            # Méret igazítás ha kell
            n = min(pred_np.shape[0], true.shape[0]) if pred_np.ndim == 2 else true.shape[0]
            pred_eval = pred_np[:n, :n] if pred_np.ndim == 2 else pred_np[:n*n].reshape(n, n)
            true_eval = true[:n, :n]

            metrics       = compute_metrics(pred_eval.flatten(), true_eval.flatten())
            nz_metrics    = nonzero_metrics(pred_eval, true_eval)
            ebart_metrics = ebart_station_metrics(pred_eval, true_eval, stations[:n])

            print_bart_results(
                model_name=f"{name} (random weights, BART scale)",
                metrics=metrics,
                nz_metrics=nz_metrics,
                ebart_metrics=ebart_metrics,
                budapest_mae=BUDAPEST_MAE.get(name),
            )

            all_results.append({
                "model":            name,
                "init":             "random",
                "network":          "BART",
                "N_zones":          N,
                "scenario":         "eBART_Antioch_2018",
                "ground_truth":     "real_smart_card_seasonal_adj",
                "MAE":              metrics["MAE"],
                "RMSE":             metrics["RMSE"],
                "R2":               metrics["R2"],
                "MAE_nz":           nz_metrics.get("MAE_nz"),
                "null_MAE":         null_metrics["MAE"],
                "budapest_val_MAE": BUDAPEST_MAE.get(name),
            })

        except Exception as e:
            print(f"  ❌ Hiba: {e}")
            import traceback; traceback.print_exc()

    if all_results:
        out = os.path.join(RESULTS_DIR, "bart_validation_results.csv")
        save_results_csv(all_results, out)

    print("\n✅ BART architektúra validáció kész.")
    print("\nMegjegyzés: random súlyok → az eredmény az architektúra")
    print("kapacitását mutatja, nem a Budapest-on tanult tudást.")


if __name__ == "__main__":
    main()