"""
utils/bart_validation.py
========================
BART eBART cross-network validation utility.

Computes the real measured ΔOD from the Antioch (eBART) extension
(opened May 26, 2018), then runs the trained thesis models against it
for cross-network generalization assessment.

Usage:
    python evaluate_bart.py --model gat
    python evaluate_bart.py --model hypergraph
    python evaluate_bart.py --model all

SETUP — Download these files manually from:
    https://www.bart.gov/about/reports/ridership
    Save to data/bart/ (relative to repo root):
        data/bart/April_2018.xls
        data/bart/June_2018.xls
        data/bart/April_2017.xls   (seasonal baseline)
        data/bart/June_2017.xls    (seasonal baseline)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── Reuse thesis config & utils ───────────────────────────────────────────────
from config.paths import Paths
from utils.data import load_od_matrix as load_visum_od  # VISUM loader for reference

# ── Constants ─────────────────────────────────────────────────────────────────
EBART_STATIONS = ["ANTC", "PITT"]   # new stations opened May 26 2018
SHEET_CANDIDATES = [
    "Avg Weekday OD", "Avg Weekday", "Average Weekday OD", "Average Weekday"
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_bart_od(filepath: str) -> pd.DataFrame:
    """
    Load a BART monthly ridership XLS into a station × station DataFrame.

    BART format: row 0 = station codes, col 0 = destination codes.
    Tries multiple sheet name variants for robustness across file vintages.
    """
    xl = pd.ExcelFile(filepath)

    df = None
    for sheet in SHEET_CANDIDATES:
        if sheet in xl.sheet_names:
            df = xl.parse(sheet, index_col=0, header=0)
            break

    if df is None:
        df = xl.parse(xl.sheet_names[0], index_col=0, header=0)

    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def align_matrices(before: pd.DataFrame,
                   after: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two OD matrices to the union of their stations.
    Stations present only in 'after' (new eBART nodes) get 0 in 'before'.
    """
    all_stations = sorted(set(before.index) | set(after.index))
    b = before.reindex(index=all_stations, columns=all_stations, fill_value=0)
    a = after.reindex(index=all_stations, columns=all_stations, fill_value=0)
    return b, a


def compute_delta(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """ΔOD = after − before."""
    b, a = align_matrices(before, after)
    return a - b


def seasonal_adjustment(delta_2018: pd.DataFrame,
                        before_2017: pd.DataFrame,
                        after_2017: pd.DataFrame) -> pd.DataFrame:
    """
    Remove seasonal trend:
        adjusted_ΔOD = ΔOD_2018 − ΔOD_2017
    Isolates the infrastructure-change effect from natural April→June variation.
    """
    delta_2017 = compute_delta(before_2017, after_2017)
    all_st = sorted(set(delta_2018.index) | set(delta_2017.index))
    d18 = delta_2018.reindex(index=all_st, columns=all_st, fill_value=0)
    d17 = delta_2017.reindex(index=all_st, columns=all_st, fill_value=0)
    return d18 - d17


def load_bart_delta(bart_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all 4 BART files and return (raw_delta, seasonal_delta).
    If 2017 files are missing, seasonal_delta == raw_delta with a warning.
    """
    def _path(name):
        return os.path.join(bart_dir, name)

    required = ["April_2018.xls", "June_2018.xls"]
    optional = ["April_2017.xls", "June_2017.xls"]

    missing_req = [f for f in required if not os.path.exists(_path(f))]
    if missing_req:
        raise FileNotFoundError(
            f"Missing required BART files in {bart_dir}:\n"
            + "\n".join(f"  {f}" for f in missing_req)
            + "\nDownload from: https://www.bart.gov/about/reports/ridership"
        )

    print("Loading BART OD matrices...")
    before_18 = load_bart_od(_path("April_2018.xls"))
    after_18  = load_bart_od(_path("June_2018.xls"))
    raw_delta = compute_delta(before_18, after_18)
    print(f"  Raw ΔOD shape: {raw_delta.shape}  "
          f"(eBART stations: {[s for s in raw_delta.index if s in EBART_STATIONS]})")

    missing_opt = [f for f in optional if not os.path.exists(_path(f))]
    if missing_opt:
        print(f"  Warning: 2017 baseline files missing — skipping seasonal adjustment.")
        return raw_delta, raw_delta

    before_17 = load_bart_od(_path("April_2017.xls"))
    after_17  = load_bart_od(_path("June_2017.xls"))
    sea_delta = seasonal_adjustment(raw_delta, before_17, after_17)
    print(f"  Seasonally adjusted ΔOD shape: {sea_delta.shape}")

    return raw_delta, sea_delta


# ─────────────────────────────────────────────────────────────────────────────
# Build graph input from BART topology (mirrors utils/data.py for Budapest)
# ─────────────────────────────────────────────────────────────────────────────

def build_bart_graph(delta: pd.DataFrame) -> dict:
    """
    Convert a BART ΔOD DataFrame into the same tensor format the thesis
    models expect, so evaluate.py can call them without modification.

    Returns a dict with keys matching the thesis DataLoader batch format:
        x          : node features  [N, 1]   (row-sum of |ΔOD| per station)
        edge_index : [2, E]                   (fully connected for BART)
        delta_od   : [N, N]                   (ground truth ΔOD as tensor)
        station_names : list[str]
    """
    stations = list(delta.index)
    N = len(stations)

    delta_arr = delta.values.astype(np.float32)

    # Node feature: absolute row-sum (total change in outbound flows)
    node_feats = np.abs(delta_arr).sum(axis=1, keepdims=True)  # [N, 1]

    # Edge index: fully connected (BART is small enough)
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return {
        "x":             torch.tensor(node_feats, dtype=torch.float32),
        "edge_index":    edge_index,
        "delta_od":      torch.tensor(delta_arr,  dtype=torch.float32),
        "station_names": stations,
        "N":             N,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (same as evaluate.py to keep results comparable)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """MAE, RMSE, R² — identical formulas to evaluate.py."""
    mae  = float(np.abs(pred - true).mean())
    rmse = float(np.sqrt(((pred - true) ** 2).mean()))
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def nonzero_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Metrics restricted to cells where ground-truth ΔOD ≠ 0."""
    mask = true.flatten() != 0
    if mask.sum() == 0:
        return {"MAE_nz": float("nan"), "RMSE_nz": float("nan")}
    p, t = pred.flatten()[mask], true.flatten()[mask]
    return {
        "MAE_nz":  float(np.abs(p - t).mean()),
        "RMSE_nz": float(np.sqrt(((p - t) ** 2).mean())),
    }


def ebart_station_metrics(pred: np.ndarray,
                          true: np.ndarray,
                          stations: list[str]) -> dict:
    """
    Metrics for OD pairs involving the new eBART stations only.
    This is the most interpretable subset: how well does the model
    predict the redistribution *to/from* the new infrastructure?
    """
    idx = [i for i, s in enumerate(stations) if s in EBART_STATIONS]
    if not idx:
        return {}

    mask = np.zeros(true.shape, dtype=bool)
    for i in idx:
        mask[i, :] = True
        mask[:, i] = True

    p, t = pred[mask], true[mask]
    return {
        "MAE_ebart":  float(np.abs(p - t).mean()),
        "RMSE_ebart": float(np.sqrt(((p - t) ** 2).mean())),
        "total_pred_ebart": float(p.sum()),
        "total_true_ebart": float(t.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_bart_results(model_name: str,
                       metrics: dict,
                       nz_metrics: dict,
                       ebart_metrics: dict,
                       budapest_mae: float | None = None) -> None:
    """Print results in the same style as evaluate.py console output."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  BART cross-network validation — {model_name}")
    print(sep)
    print(f"  Network:   BART (San Francisco Bay Area)")
    print(f"  Scenario:  eBART Antioch extension  (opened 2018-05-26)")
    print(f"  Ground truth: real measured smart-card OD  (seasonally adjusted)")
    print(sep)
    print(f"  MAE            : {metrics['MAE']:.4f}")
    print(f"  RMSE           : {metrics['RMSE']:.4f}")
    print(f"  R²             : {metrics['R2']:.4f}")
    print(f"  MAE (non-zero) : {nz_metrics.get('MAE_nz', float('nan')):.4f}")
    if ebart_metrics:
        print(f"  MAE (eBART OD) : {ebart_metrics.get('MAE_ebart', float('nan')):.4f}")
        print(f"  Pred ΔOD eBART : {ebart_metrics.get('total_pred_ebart', 0):+.0f} trips/day")
        print(f"  True ΔOD eBART : {ebart_metrics.get('total_true_ebart', 0):+.0f} trips/day")
    if budapest_mae is not None:
        ratio = metrics['MAE'] / budapest_mae if budapest_mae else float("nan")
        print(f"\n  Budapest MAE (M1 val) : {budapest_mae:.4f}")
        print(f"  BART/Budapest ratio   : {ratio:.2f}x")
        print(f"  (>1 expected — different network scale & aggregation level)")
    print(sep)


def save_results_csv(results: list[dict], outpath: str) -> None:
    """Append BART results to a CSV alongside Budapest evaluate.py results."""
    df = pd.DataFrame(results)
    df.to_csv(outpath, index=False)
    print(f"\n  Results saved: {outpath}")