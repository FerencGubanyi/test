"""
shanghai_data.py — Shanghai MetroFlow adapter for transfer learning.

Source: Scientific Data 2025, DOI 10.1038/s41597-025-05416-8
Data:   figshare DOI 10.6084/m9.figshare.28844942

Key facts (confirmed from file inspection):
  - 302 stations, stationID values like 112, 113, ..., 2054
  - OD: sparse CSV (date, timeslot, originStation, destinationStation, Flow)
  - 144 timeslots/day (10-min intervals)
  - Coverage: May 1 – Aug 31 2017 (123 days)
  - neighbour column in stationInfo gives physical adjacency

Strategy for ΔOD generation:
  We aggregate OD by week and subtract consecutive weeks.
  Week N+1 minus Week N = simulated "before/after" scenario.
  This gives ~17 real ΔOD training samples from one dataset.
  We also generate synthetic perturbations from baseline OD matrices.

Usage:
    from utils.shanghai_data import load_shanghai_transfer_dataset
    graph_data, scenarios = load_shanghai_transfer_dataset()
"""

import os
import ast
import numpy as np
import pandas as pd
import torch

BASE_DIR = "/content/data/shanghai/MetroFlow"


# ---------------------------------------------------------------------------
# Load station info + build station index
# ---------------------------------------------------------------------------

def load_station_info(base_dir=BASE_DIR):
    si = pd.read_csv(os.path.join(base_dir, "stationInfo.csv"))
    # Clean column names (may have leading spaces)
    si.columns = si.columns.str.strip()
    station_ids   = si["stationID"].tolist()          # e.g. [2048, 2049, ...]
    station_names = si["name"].tolist()
    lons          = si["lon"].tolist()
    lats          = si["lat"].tolist()
    neighbours    = si["neighbour"].tolist()           # stored as string lists

    id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    return {
        "station_ids":   station_ids,
        "station_names": station_names,
        "id_to_idx":     id_to_idx,
        "lons":          lons,
        "lats":          lats,
        "neighbours":    neighbours,
        "n_nodes":       len(station_ids),
    }


# ---------------------------------------------------------------------------
# Build graph from neighbour list
# ---------------------------------------------------------------------------

def build_shanghai_graph(station_info):
    n_nodes   = station_info["n_nodes"]
    id_to_idx = station_info["id_to_idx"]
    neighbours= station_info["neighbours"]

    src, dst = [], []
    for i, nb_str in enumerate(neighbours):
        try:
            nb_list = ast.literal_eval(str(nb_str))
        except Exception:
            continue
        for nb_id in nb_list:
            j = id_to_idx.get(nb_id)
            if j is not None and i != j:
                src.append(i); dst.append(j)

    if not src:
        # Fallback: fully connected
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                src += [i, j]; dst += [j, i]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.ones(len(src), 2, dtype=torch.float)

    # Hyperedges: group stations by metro line
    # Approximate: stations with IDs in same thousand = same line family
    line_groups = {}
    for sid, idx in id_to_idx.items():
        line_key = sid // 100
        if line_key not in line_groups:
            line_groups[line_key] = []
        line_groups[line_key].append(idx)

    he_station, he_route = [], []
    for r_idx, (_, members) in enumerate(line_groups.items()):
        for s_idx in members:
            he_station.append(s_idx)
            he_route.append(r_idx)

    hyperedge_index = torch.tensor(
        [he_station, he_route], dtype=torch.long
    ) if he_station else torch.zeros(2, 0, dtype=torch.long)

    return {
        "edge_index":      edge_index,
        "edge_attr":       edge_attr,
        "hyperedge_index": hyperedge_index,
        "n_hyperedges":    len(line_groups),
        "n_nodes":         n_nodes,
        "station_names":   station_info["station_names"],
    }


# ---------------------------------------------------------------------------
# Load OD data and aggregate to weekly OD matrices
# ---------------------------------------------------------------------------

def load_weekly_od_matrices(base_dir=BASE_DIR, station_info=None, verbose=True):
    """
    Load metroData_ODFlow.csv and aggregate to weekly OD matrices.
    Returns list of (week_label, OD_matrix) tuples.
    OD_matrix shape: (N, N) float32, summed over all timeslots in the week.
    Only workday ODs included (filter by workday_calendar).
    """
    if verbose:
        print("  Loading OD flow CSV (this takes ~30s)...")

    od_path  = os.path.join(base_dir, "metroData_ODFlow.csv")
    cal_path = os.path.join(base_dir, "MetaData", "workday_calendar.csv")

    # Load calendar — filter to workdays only
    cal = pd.read_csv(cal_path)
    cal.columns = cal.columns.str.strip()
    workdays = set(cal[cal["isWorday"] == 1]["date"].astype(str).tolist())

    # Load OD in chunks to avoid memory issues
    id_to_idx = station_info["id_to_idx"]
    N         = station_info["n_nodes"]

    # Read full file — 1.1GB but pandas handles it
    od = pd.read_csv(od_path)
    od.columns = od.columns.str.strip()

    # Filter to workdays
    od["date"] = od["date"].astype(str)
    od = od[od["date"].isin(workdays)].copy()

    # Map station IDs to indices
    od["orig_idx"] = od["originStation"].map(id_to_idx)
    od["dest_idx"] = od["destinationStation"].map(id_to_idx)
    od = od.dropna(subset=["orig_idx", "dest_idx"])
    od["orig_idx"] = od["orig_idx"].astype(int)
    od["dest_idx"] = od["dest_idx"].astype(int)

    # Add week number (ISO week)
    od["date_dt"] = pd.to_datetime(od["date"], format="%Y%m%d")
    od["week"]    = od["date_dt"].dt.isocalendar().week.astype(int)
    od["year"]    = od["date_dt"].dt.isocalendar().year.astype(int)
    od["year_week"] = od["year"].astype(str) + "_W" + od["week"].astype(str).str.zfill(2)

    weeks = sorted(od["year_week"].unique())
    if verbose:
        print(f"  Found {len(weeks)} workday weeks: {weeks[:3]}...{weeks[-2:]}")

    weekly_matrices = []
    for week in weeks:
        week_od = od[od["year_week"] == week]
        matrix  = np.zeros((N, N), dtype=np.float32)
        for _, row in week_od.iterrows():
            i, j = int(row["orig_idx"]), int(row["dest_idx"])
            if 0 <= i < N and 0 <= j < N:
                matrix[i, j] += float(row["Flow"])
        weekly_matrices.append((week, matrix))
        if verbose and len(weekly_matrices) % 4 == 0:
            print(f"  Processed week {week} ({len(weekly_matrices)}/{len(weeks)})")

    return weekly_matrices


# ---------------------------------------------------------------------------
# Node feature engineering (same 16-dim as BKK/BART)
# ---------------------------------------------------------------------------

def compute_node_features(od_matrix: np.ndarray) -> np.ndarray:
    N = od_matrix.shape[0]
    features = []
    for i in range(N):
        row = od_matrix[i, :]
        col = od_matrix[:, i]
        feat = [
            float(row.sum()), float(col.sum()),
            float(row.mean()), float(row.std() + 1e-8),
            float(col.mean()), float(col.std() + 1e-8),
            float((row > 0).sum()), float((col > 0).sum()),
            float(row.max()), float(col.max()),
            float(np.percentile(row, 75)), float(np.percentile(col, 75)),
            float(np.percentile(row, 25)), float(np.percentile(col, 25)),
            float(row.sum() / (col.sum() + 1e-6)),
            float(np.log1p(max(row.sum(), 0))),
        ]
        features.append(feat)
    features = np.array(features, dtype=np.float32)
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-8
    features = (features - mean) / std
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Synthetic ΔOD scenarios
# ---------------------------------------------------------------------------

def generate_synthetic_scenarios(baseline_matrices, n_synthetic=40, rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    scenarios = []
    for _ in range(n_synthetic):
        base = baseline_matrices[rng.integers(len(baseline_matrices))].copy()
        N    = base.shape[0]
        # Pick a random "corridor" of stations
        n_affected = rng.integers(3, max(5, N // 20))
        affected   = rng.choice(N, size=n_affected, replace=False)
        delta      = np.zeros_like(base)
        magnitude  = rng.uniform(0.03, 0.15)
        for s in affected:
            inflow_delta = base[:, s] * magnitude * rng.uniform(0.5, 1.5)
            non_affected = np.setdiff1d(np.arange(N), affected)
            if len(non_affected) > 0:
                reduction = inflow_delta.sum() * 0.7 / len(non_affected)
                delta[non_affected, s] -= reduction
            delta[affected, s] += inflow_delta[affected]
        delta -= delta.mean()
        if not np.all(np.isfinite(delta)):
            continue
        scenarios.append(delta)
    return scenarios


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_shanghai_transfer_dataset(
    base_dir=BASE_DIR,
    n_synthetic=40,
    val_weeks=2,
    verbose=True,
):
    """
    Load Shanghai MetroFlow as a transfer learning dataset.

    Returns:
        graph_data : dict with edge_index, hyperedge_index, n_nodes, etc.
        scenarios  : list of dicts with node_features, delta_od, is_real, split

    Strategy:
        - Aggregate OD to weekly matrices (17 weeks of workday data)
        - ΔOD = consecutive week differences (16 real scenarios)
        - Last val_weeks differences = validation
        - First (16 - val_weeks) = training
        - Plus n_synthetic synthetic scenarios
    """
    if verbose:
        print("=== Loading Shanghai MetroFlow transfer dataset ===")

    # 1. Station info + graph
    station_info = load_station_info(base_dir)
    N = station_info["n_nodes"]
    if verbose:
        print(f"  Stations: {N}")

    graph_data = build_shanghai_graph(station_info)
    if verbose:
        print(f"  Graph: {graph_data['edge_index'].shape[1]} edges, "
              f"{graph_data['n_hyperedges']} hyperedges")

    # 2. Weekly OD matrices
    weekly_matrices = load_weekly_od_matrices(base_dir, station_info, verbose)

    if len(weekly_matrices) < 3:
        raise RuntimeError(
            f"Only {len(weekly_matrices)} weeks found — need at least 3. "
            "Check that workday_calendar.csv dates match OD dates."
        )

    # 3. Consecutive-week ΔOD scenarios
    real_scenarios = []
    for i in range(1, len(weekly_matrices)):
        week_label_before, od_before = weekly_matrices[i-1]
        week_label_after,  od_after  = weekly_matrices[i]
        delta         = od_after - od_before
        node_features = compute_node_features(od_before)
        std           = float(delta.std()) + 1e-8

        real_scenarios.append({
            "node_features":       node_features,
            "delta_od":            delta,
            "delta_od_normalized": delta / std,
            "std":   std,
            "is_real": True,
            "label":   f"shanghai_{week_label_before}_to_{week_label_after}",
        })

    # Last val_weeks real scenarios = validation; rest = training
    n_real = len(real_scenarios)
    for i, s in enumerate(real_scenarios):
        s["split"] = "val" if i >= n_real - val_weeks else "train"

    if verbose:
        train_r = sum(1 for s in real_scenarios if s["split"] == "train")
        val_r   = sum(1 for s in real_scenarios if s["split"] == "val")
        print(f"  Real ΔOD scenarios: {n_real} "
              f"(train={train_r}, val={val_r})")

    # 4. Synthetic scenarios from baseline matrices
    baseline_matrices = [od for _, od in weekly_matrices[:4]]
    if verbose:
        print(f"  Generating {n_synthetic} synthetic scenarios...")
    synthetic_deltas = generate_synthetic_scenarios(
        baseline_matrices, n_synthetic=n_synthetic
    )
    ref_od = baseline_matrices[0]
    synthetic_scenarios = []
    for i, delta in enumerate(synthetic_deltas):
        std = float(delta.std()) + 1e-8
        synthetic_scenarios.append({
            "node_features":       compute_node_features(ref_od),
            "delta_od":            delta,
            "delta_od_normalized": delta / std,
            "std":   std,
            "is_real": False,
            "label":   f"synthetic_sh_{i:03d}",
            "split":   "train",
        })

    all_scenarios = real_scenarios + synthetic_scenarios

    train_count = sum(1 for s in all_scenarios if s["split"] == "train")
    val_count   = sum(1 for s in all_scenarios if s["split"] == "val")
    real_count  = sum(1 for s in all_scenarios if s["is_real"])

    if verbose:
        print(f"  Total: {len(all_scenarios)} scenarios "
              f"({real_count} real, {len(all_scenarios)-real_count} synthetic)")
        print(f"  Train: {train_count}  |  Val: {val_count}")
        print("=== Shanghai dataset ready ===\n")

    return graph_data, all_scenarios