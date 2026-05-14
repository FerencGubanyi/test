"""
train_bart.py  —  Mini training loop BART eBART adatokon
=========================================================
A modell architektúra ugyanaz mint Budapest, de:
  - num_zones = N_BART (~50 állomás)
  - Tanítás: havi OD mátrixok mint "snapshots" (leave-one-out CV)
  - Validáció: eBART megnyitás hónapja (June 2018) — a ΔOD ground truth

Leave-one-out CV logika:
  Minden elérhető hónap (2017-2019) = egy snapshot.
  Tanítás: összes hónap KIVÉVE June 2018.
  Validáció: June 2018 vs April 2018 ΔOD = ground truth.

Usage:
    python train_bart.py --model gat --epochs 100
    python train_bart.py --model hypergraph --epochs 100
    python train_bart.py --model all --epochs 200 --lr 1e-3
"""

import argparse, os, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config.paths as _p
from utils.bart_validation import (
    load_bart_od,
    load_bart_delta,
    align_matrices,
    compute_metrics,
    nonzero_metrics,
    ebart_station_metrics,
    print_bart_results,
    save_results_csv,
    EBART_STATIONS,
)

BART_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bart")
RESULTS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CKPT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

BUDAPEST_MAE = {"gat": 13.06, "hypergraph": 13.26}


# ─────────────────────────────────────────────────────────────────────────────
# Adat előkészítés
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_index(N: int) -> torch.Tensor:
    """Teljesen összekötött gráf N állomásra (self-loop nélkül)."""
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def od_to_features(od: pd.DataFrame) -> torch.Tensor:
    """
    OD mátrixból node feature vektor.
    Feature-ök állomásonként: [kimenő_összeg, bejövő_összeg] → [N, 2]
    """
    arr = od.values.astype(np.float32)
    out_sum = arr.sum(axis=1, keepdims=True)   # kimenő forgalom
    in_sum  = arr.sum(axis=0, keepdims=True).T  # bejövő forgalom
    feats   = np.concatenate([out_sum, in_sum], axis=1)  # [N, 2]
    # Normalizálás
    mx = feats.max()
    if mx > 0:
        feats = feats / mx
    return torch.tensor(feats, dtype=torch.float32)


def load_all_months(bart_dir: str) -> dict[str, pd.DataFrame]:
    """
    Betölti az összes elérhető havi OD fájlt.
    Kulcs: fájlnév (pl. 'April_2018'), érték: OD DataFrame.
    """
    months = {}
    for path in sorted(glob.glob(os.path.join(bart_dir, "*.xls"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_bart_od(path)
            months[name] = df
            print(f"  ✅ {name}: {df.shape}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    return months


def prepare_dataset(months: dict[str, pd.DataFrame],
                    val_before: str = "April_2018",
                    val_after:  str = "June_2018") -> tuple:
    """
    Train: minden egymást követő hónap-pár ΔOD-ja,
           KIVÉVE a validációs párt.
    Val:   April_2018 → June_2018 ΔOD (eBART megnyitás).

    Returns: train_samples, val_delta, stations, N
    """
    # Közös állomáskészlet (unió, 0-val töltve ha hiányzik)
    all_stations = sorted(set().union(*[set(df.index) for df in months.values()]))
    N = len(all_stations)
    print(f"\nÁllomások száma: {N}")

    def reindex(df):
        return df.reindex(index=all_stations, columns=all_stations, fill_value=0)

    aligned = {k: reindex(v) for k, v in months.items()}

    # Validációs ΔOD
    if val_before not in aligned or val_after not in aligned:
        raise ValueError(f"Validációs fájlok hiányoznak: {val_before}, {val_after}")

    val_delta = aligned[val_after] - aligned[val_before]

    # Train párok: minden szomszédos hónap-pár, kivéve val pár
    sorted_keys = sorted(aligned.keys())
    train_samples = []
    skip_pair = {(val_before, val_after), (val_after, val_before)}

    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i+1]
        if (k1, k2) in skip_pair:
            continue
        delta = aligned[k2] - aligned[k1]
        x     = od_to_features(aligned[k1])
        y     = torch.tensor(delta.values.astype(np.float32))
        train_samples.append((x, y, f"{k1}→{k2}"))

    print(f"Train párok: {len(train_samples)}")
    print(f"Val pár: {val_before} → {val_after}")
    return train_samples, val_delta, all_stations, N


# ─────────────────────────────────────────────────────────────────────────────
# Modell
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, N: int, device: torch.device):
    if model_name == "gat":
        from models.gat_lstm import GATLSTM
        model = GATLSTM(in_channels=2, hidden_channels=32,
                        lstm_hidden=64, num_zones=N)
    else:
        from models.hypergraph_lstm import HypergraphLSTM
        model = HypergraphLSTM(in_channels=2, hidden_channels=32,
                               lstm_hidden=64, num_zones=N)
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(model_name: str,
                   train_samples: list,
                   val_delta: pd.DataFrame,
                   stations: list,
                   N: int,
                   epochs: int,
                   lr: float,
                   device: torch.device) -> dict:

    model      = build_model(model_name, N, device)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.L1Loss()   # MAE loss — ugyanaz mint Budapest training
    edge_index = build_edge_index(N).to(device)

    val_true = val_delta.values.astype(np.float32)
    best_val_mae = float("inf")
    best_ckpt    = os.path.join(CKPT_DIR, f"bart_{model_name}_best.pt")

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()} | N={N} | epochs={epochs} | lr={lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x, y, label in train_samples:
            x = x.to(device)       # [N, 2]
            y = y.to(device)       # [N, N]

            x_seq = x.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
            optimizer.zero_grad()

            try:
                pred = model(x_seq, edge_index)
            except TypeError:
                batch = torch.zeros(N, dtype=torch.long, device=device)
                pred  = model(x_seq, edge_index, batch)

            pred = pred.squeeze()
            if pred.shape != y.shape:
                pred = pred.view_as(y)

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_samples), 1)

        # Validáció
        model.eval()
        with torch.no_grad():
            # Val input: April 2018 OD → de most csak nullával inputolunk
            # (nincs külön val input feature, a ΔOD a target)
            x_val = torch.zeros(N, 2, device=device)
            x_seq = x_val.unsqueeze(0).unsqueeze(0)
            try:
                pred_val = model(x_seq, edge_index)
            except TypeError:
                batch    = torch.zeros(N, dtype=torch.long, device=device)
                pred_val = model(x_seq, edge_index, batch)

        pred_np  = pred_val.squeeze().cpu().numpy()
        if pred_np.ndim == 1:
            pred_np = pred_np.reshape(N, N)

        val_metrics = compute_metrics(pred_np.flatten(), val_true.flatten())
        val_mae     = val_metrics["MAE"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"Train loss: {avg_loss:.4f} | Val MAE: {val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_mae": val_mae,
                        "N": N, "model_name": model_name}, best_ckpt)

    # Végső kiértékelés a legjobb checkpointtal
    print(f"\n  Legjobb Val MAE: {best_val_mae:.4f} (mentve: {best_ckpt})")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_val  = torch.zeros(N, 2, device=device)
        x_seq  = x_val.unsqueeze(0).unsqueeze(0)
        try:
            pred_f = model(x_seq, edge_index)
        except TypeError:
            pred_f = model(x_seq, edge_index,
                           torch.zeros(N, dtype=torch.long, device=device))

    pred_final = pred_f.squeeze().cpu().numpy()
    if pred_final.ndim == 1:
        pred_final = pred_final.reshape(N, N)

    metrics       = compute_metrics(pred_final.flatten(), val_true.flatten())
    nz_metrics    = nonzero_metrics(pred_final, val_true)
    ebart_metrics = ebart_station_metrics(pred_final, val_true, stations)

    print_bart_results(
        model_name=f"{model_name} (BART-trained, N={N})",
        metrics=metrics,
        nz_metrics=nz_metrics,
        ebart_metrics=ebart_metrics,
        budapest_mae=BUDAPEST_MAE.get(model_name),
    )

    return {
        "model": model_name, "init": "bart_trained", "N_zones": N,
        "epochs": epochs, "lr": lr,
        "network": "BART", "scenario": "eBART_Antioch_2018",
        "ground_truth": "real_smart_card_seasonal_adj",
        **metrics, **nz_metrics,
        "budapest_val_MAE": BUDAPEST_MAE.get(model_name),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["gat", "hypergraph", "all"], default="all")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--bart-dir", default=BART_DATA_DIR)
    parser.add_argument("--no-seasonal", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Adatok
    print(f"\nBART fájlok betöltése: {args.bart_dir}")
    months = load_all_months(args.bart_dir)
    if len(months) < 2:
        print("❌ Legalább 2 havi fájl kell. Töltsd le a többi hónapot is!")
        return

    train_samples, val_delta, stations, N = prepare_dataset(months)

    # Modellek
    models_to_run = ["gat", "hypergraph"] if args.model == "all" else [args.model]
    all_results   = []

    for name in models_to_run:
        result = train_and_eval(
            model_name=name,
            train_samples=train_samples,
            val_delta=val_delta,
            stations=stations,
            N=N,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        all_results.append(result)

    out = os.path.join(RESULTS_DIR, "bart_training_results.csv")
    save_results_csv(all_results, out)
    print(f"\n✅ Kész. Eredmények: {out}")


if __name__ == "__main__":
    main()