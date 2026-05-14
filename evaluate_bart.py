"""
evaluate_bart.py
================
Cross-network validation: runs trained Budapest thesis models against
real BART eBART extension ΔOD data.

Mirrors the evaluate.py interface so results are directly comparable.

Usage (same pattern as evaluate.py):
    python evaluate_bart.py --model gat
    python evaluate_bart.py --model hypergraph
    python evaluate_bart.py --model all

    # With explicit checkpoint paths:
    python evaluate_bart.py --model gat --checkpoint checkpoints/gat_lstm_best.pt

    # Skip seasonal adjustment (use raw ΔOD):
    python evaluate_bart.py --model all --no-seasonal

Outputs:
    results/bart_validation_results.csv
    results/bart_delta_od_raw.csv
    results/bart_delta_od_seasonal.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

# ── Thesis imports ────────────────────────────────────────────────────────────
from config.paths import Paths
from models.gat_lstm import GATLSTM
from models.hypergraph_lstm import HypergraphLSTM
from utils.bart_validation import (
    load_bart_delta,
    build_bart_graph,
    compute_metrics,
    nonzero_metrics,
    ebart_station_metrics,
    print_bart_results,
    save_results_csv,
    EBART_STATIONS,
)

# ── Config ────────────────────────────────────────────────────────────────────
BART_DATA_DIR = os.path.join(Paths.DATA_DIR, "bart")
RESULTS_DIR   = os.path.join(Paths.ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Budapest M1 validation MAE (from evaluate.py results) for cross-reference
# Update these if your evaluate.py numbers change
BUDAPEST_MAE = {
    "gat":        13.06,
    "hypergraph": 13.26,
}

CHECKPOINTS = {
    "gat":        os.path.join(Paths.ROOT, "checkpoints", "gat_lstm_best.pt"),
    "hypergraph": os.path.join(Paths.ROOT, "checkpoints", "hg_lstm_best.pt"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Model inference
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, checkpoint_path: str, device: torch.device):
    """Load a trained thesis model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first, or point --checkpoint to the correct path."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint (train.py saves it as 'config' or 'args')
    cfg = checkpoint.get("config", checkpoint.get("args", {}))

    if model_name == "gat":
        model = GATLSTM(
            in_channels=cfg.get("in_channels", 1),
            hidden_channels=cfg.get("hidden_channels", 64),
            lstm_hidden=cfg.get("lstm_hidden", 128),
            num_zones=cfg.get("num_zones", 50),   # BART has ~50 stations
        )
    else:
        model = HypergraphLSTM(
            in_channels=cfg.get("in_channels", 1),
            hidden_channels=cfg.get("hidden_channels", 64),
            lstm_hidden=cfg.get("lstm_hidden", 128),
            num_zones=cfg.get("num_zones", 50),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"  Loaded {model_name} from {checkpoint_path}")
    return model


def run_inference(model, graph: dict, device: torch.device) -> np.ndarray:
    """
    Run one forward pass on the BART graph.
    Returns predicted ΔOD as a numpy array [N, N].
    """
    x          = graph["x"].to(device)
    edge_index = graph["edge_index"].to(device)
    N          = graph["N"]

    with torch.no_grad():
        # Models expect [batch, seq, features] — use seq_len=1 for single snapshot
        x_seq = x.unsqueeze(0).unsqueeze(0)   # [1, 1, N, F]

        # Forward pass — adapt to your actual model signature
        try:
            pred = model(x_seq, edge_index)
        except TypeError:
            # Some model variants take (x, edge_index, batch)
            batch = torch.zeros(N, dtype=torch.long, device=device)
            pred = model(x_seq, edge_index, batch)

    pred_np = pred.squeeze().cpu().numpy()

    # Reshape to [N, N] if model outputs flat vector
    if pred_np.ndim == 1:
        side = int(np.sqrt(pred_np.size))
        if side * side == pred_np.size:
            pred_np = pred_np.reshape(side, side)

    return pred_np


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str,
                   checkpoint_path: str,
                   delta: pd.DataFrame,
                   device: torch.device) -> dict:
    """Evaluate one model against BART ΔOD ground truth."""
    print(f"\nEvaluating {model_name}...")

    model = load_model(model_name, checkpoint_path, device)
    graph = build_bart_graph(delta)
    true  = graph["delta_od"].numpy()

    pred = run_inference(model, graph, device)

    # Align shapes (BART stations may be fewer than Budapest zones)
    N = min(pred.shape[0], true.shape[0]) if pred.ndim == 2 else true.shape[0]
    if pred.ndim == 2:
        pred_eval = pred[:N, :N]
        true_eval = true[:N, :N]
    else:
        pred_eval = pred[:N*N]
        true_eval = true[:N, :N].flatten()

    metrics       = compute_metrics(pred_eval.flatten(), true_eval.flatten())
    nz_metrics    = nonzero_metrics(pred_eval, true_eval)
    ebart_metrics = ebart_station_metrics(pred_eval, true_eval,
                                          graph["station_names"][:N])

    print_bart_results(
        model_name=model_name,
        metrics=metrics,
        nz_metrics=nz_metrics,
        ebart_metrics=ebart_metrics,
        budapest_mae=BUDAPEST_MAE.get(model_name),
    )

    return {
        "model":            model_name,
        "network":          "BART",
        "scenario":         "eBART_Antioch_2018",
        "ground_truth":     "real_smart_card",
        **metrics,
        **nz_metrics,
        **ebart_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-network BART validation for Budapest thesis models"
    )
    parser.add_argument("--model", choices=["gat", "hypergraph", "all"],
                        default="all")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (single model only)")
    parser.add_argument("--no-seasonal", action="store_true",
                        help="Use raw ΔOD instead of seasonally adjusted")
    parser.add_argument("--bart-dir", default=BART_DATA_DIR,
                        help=f"Directory with BART XLS files (default: {BART_DATA_DIR})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load BART data
    print(f"\nBART data directory: {args.bart_dir}")
    raw_delta, sea_delta = load_bart_delta(args.bart_dir)

    # Save CSVs for reference
    raw_delta.to_csv(os.path.join(RESULTS_DIR, "bart_delta_od_raw.csv"))
    sea_delta.to_csv(os.path.join(RESULTS_DIR, "bart_delta_od_seasonal.csv"))

    delta = raw_delta if args.no_seasonal else sea_delta
    print(f"Using {'raw' if args.no_seasonal else 'seasonally adjusted'} ΔOD  "
          f"[{delta.shape[0]} stations]")

    # Run evaluation
    models_to_run = (
        ["gat", "hypergraph"] if args.model == "all" else [args.model]
    )

    all_results = []
    for name in models_to_run:
        ckpt = args.checkpoint if args.checkpoint else CHECKPOINTS[name]
        try:
            result = evaluate_model(name, ckpt, delta, device)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n  ⚠️  Skipping {name}: {e}")

    # Save combined results
    if all_results:
        out = os.path.join(RESULTS_DIR, "bart_validation_results.csv")
        save_results_csv(all_results, out)

    print("\n✅  BART cross-network validation complete.")
    print(
        "\nInterpretation note:\n"
        "  BART is station-level (~50 nodes) vs Budapest zone-level (~1419 zones).\n"
        "  Quantitative MAE is not directly comparable across networks.\n"
        "  Compare patterns: does the model correctly identify which O-D pairs\n"
        "  gain/lose most trips when new stations are added?\n"
    )


if __name__ == "__main__":
    main()