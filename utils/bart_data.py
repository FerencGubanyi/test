"""
BART data adapter for transfer learning.
Fully patched version — reads extracted xlsx files directly,
handles both old and new BART sheet name formats.
"""

import os
import io
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BART_GTFS_URL = "https://www.bart.gov/dev/schedules/google_transit.zip"
DATA_DIR = "data/bart"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _download_file(url: str, dest_path: str, desc: str = ""):
    if os.path.exists(dest_path):
        print(f"  [cached] {desc or dest_path}")
        return True
    print(f"  [download] {desc or url}")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  [warning] Could not download {desc}: {e}")
        return False


MONTHS_TO_FETCH = {
    "before_berryessa": ("2020", "02"),
    "after_berryessa":  ("2020", "09"),
    "before_antioch":   ("2023", "09"),
    "after_antioch":    ("2024", "01"),
    "baseline_2019_01": ("2019", "01"),
    "baseline_2019_06": ("2019", "06"),
    "baseline_2019_10": ("2019", "10"),
}


def download_bart_od_matrices(data_dir: str = DATA_DIR) -> dict:
    _ensure_dir(data_dir)
    paths = {}
    for label, (year, month) in MONTHS_TO_FETCH.items():
        filename = f"{year}_{month}_average_weekday.xlsx"
        dest = os.path.join(data_dir, filename)
        if os.path.exists(dest):
            print(f"  [cached] BART OD {year}-{month}")
            paths[label] = dest
        else:
            print(f"  [download] BART OD {year}-{month}")
            print(f"  [warning] Could not download {filename}; will use synthetic fallback for '{label}'")
    return paths


def download_bart_gtfs(data_dir: str = DATA_DIR) -> str:
    _ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "google_transit.zip")
    gtfs_dir = os.path.join(data_dir, "gtfs")
    _download_file(BART_GTFS_URL, zip_path, desc="BART GTFS")
    if not os.path.exists(gtfs_dir) and os.path.exists(zip_path):
        print("  [extract] BART GTFS")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(gtfs_dir)
    return gtfs_dir


# ---------------------------------------------------------------------------
# OD matrix loading — handles both sheet name formats
# ---------------------------------------------------------------------------

def _load_od_excel(path: str) -> tuple:
    xl = pd.ExcelFile(path)
    candidates = ["Avg Weekday OD", "Average Weekday", "Avg Weekday"]
    sheet = next((s for s in candidates if s in xl.sheet_names), None)
    if sheet is None:
        raise ValueError(f"No weekday sheet in {path}. Found: {xl.sheet_names}")

    df = pd.read_excel(path, sheet_name=sheet, header=None)

    # Row 1 col 1 onward = station codes
    raw_stations = [str(s).strip() for s in df.iloc[1, 1:].tolist()]
    stations = [s for s in raw_stations
                if s not in ("nan", "", "None")
                and len(s) <= 6
                and not s.startswith("#")]
    N = len(stations)
    if N == 0:
        raise ValueError(f"No valid stations found in {path}")

    matrix_raw = df.iloc[2:2+N, 1:1+N]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = matrix_raw.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0).values.astype(np.float32)

    return stations, matrix


# ---------------------------------------------------------------------------
# Matrix alignment
# ---------------------------------------------------------------------------

def _align_matrices(before_stations, before_matrix, after_stations, after_matrix):
    all_stations = sorted(set(before_stations) | set(after_stations))
    N = len(all_stations)
    idx_before = {s: i for i, s in enumerate(before_stations)}
    idx_after  = {s: i for i, s in enumerate(after_stations)}

    before_aligned = np.zeros((N, N), dtype=np.float32)
    after_aligned  = np.zeros((N, N), dtype=np.float32)

    for i, s_i in enumerate(all_stations):
        for j, s_j in enumerate(all_stations):
            if s_i in idx_before and s_j in idx_before:
                before_aligned[i, j] = before_matrix[idx_before[s_i], idx_before[s_j]]
            if s_i in idx_after and s_j in idx_after:
                after_aligned[i, j]  = after_matrix[idx_after[s_i],  idx_after[s_j]]

    return all_stations, before_aligned, after_aligned


# ---------------------------------------------------------------------------
# Graph from GTFS
# ---------------------------------------------------------------------------

def _fuzzy_match_station(name: str, stations: list) -> str:
    name_lower = name.lower().strip()
    for s in stations:
        if s.lower().strip() == name_lower:
            return s
    for s in stations:
        if name_lower in s.lower() or s.lower() in name_lower:
            return s
    return name


def build_bart_graph(gtfs_dir: str, stations: list) -> dict:
    try:
        stops_df      = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"))
        trips_df      = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
        stop_times_df = pd.read_csv(
            os.path.join(gtfs_dir, "stop_times.txt"),
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
    except Exception as e:
        print(f"  [warning] GTFS parse failed: {e} — using fully connected graph")
        N = len(stations)
        src, dst = [], []
        for i in range(N):
            for j in range(i+1, N):
                src += [i, j]; dst += [j, i]
        return {
            "edge_index": torch.tensor([src, dst], dtype=torch.long),
            "edge_attr":  torch.ones(len(src), 2, dtype=torch.float),
            "station_to_idx": {s: i for i, s in enumerate(stations)},
            "hyperedge_index": torch.zeros(2, 0, dtype=torch.long),
            "n_hyperedges": 0,
        }

    stop_name_map  = dict(zip(stops_df["stop_id"], stops_df["stop_name"]))
    station_to_idx = {s: i for i, s in enumerate(stations)}
    trip_to_route  = dict(zip(trips_df["trip_id"], trips_df["route_id"]))

    edges      = {}
    hyperedges = {}

    stop_times_sorted = stop_times_df.sort_values(["trip_id", "stop_sequence"])
    current_trip = None
    prev_idx     = None

    for _, row in stop_times_sorted.iterrows():
        trip_id   = row["trip_id"]
        stop_name = stop_name_map.get(row["stop_id"], "")
        matched   = _fuzzy_match_station(stop_name, stations)
        node_idx  = station_to_idx.get(matched)
        route_id  = trip_to_route.get(trip_id, "unknown")

        if node_idx is not None:
            if route_id not in hyperedges:
                hyperedges[route_id] = set()
            hyperedges[route_id].add(node_idx)

        if trip_id != current_trip:
            current_trip = trip_id
            prev_idx = node_idx
            continue

        if prev_idx is not None and node_idx is not None and prev_idx != node_idx:
            key = (min(prev_idx, node_idx), max(prev_idx, node_idx))
            if key not in edges:
                edges[key] = set()
            edges[key].add(route_id)

        prev_idx = node_idx

    if edges:
        edge_list  = list(edges.keys())
        edge_index = torch.tensor(
            [[e[0] for e in edge_list] + [e[1] for e in edge_list],
             [e[1] for e in edge_list] + [e[0] for e in edge_list]],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [[len(edges[e]), 5.0] for e in edge_list] * 2, dtype=torch.float
        )
    else:
        N = len(stations)
        src, dst = [], []
        for i in range(N):
            for j in range(i+1, N):
                src += [i, j]; dst += [j, i]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.ones(len(src), 2, dtype=torch.float)

    route_ids   = sorted(hyperedges.keys())
    he_station, he_route = [], []
    for r_idx, route_id in enumerate(route_ids):
        for s_idx in hyperedges[route_id]:
            he_station.append(s_idx)
            he_route.append(r_idx)

    hyperedge_index = torch.tensor(
        [he_station, he_route], dtype=torch.long
    ) if he_station else torch.zeros(2, 0, dtype=torch.long)

    return {
        "edge_index":      edge_index,
        "edge_attr":       edge_attr,
        "station_to_idx":  station_to_idx,
        "hyperedge_index": hyperedge_index,
        "n_hyperedges":    len(route_ids),
    }


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def compute_bart_node_features(od_matrix: np.ndarray) -> np.ndarray:
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
# Synthetic scenarios
# ---------------------------------------------------------------------------

def generate_bart_synthetic_scenarios(baseline_matrices, n_synthetic=60, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    scenarios = []
    for _ in range(n_synthetic):
        base = baseline_matrices[rng.integers(len(baseline_matrices))].copy()
        N    = base.shape[0]
        n_affected = rng.integers(2, max(3, N // 4))
        affected   = rng.choice(N, size=n_affected, replace=False)
        delta      = np.zeros_like(base)
        magnitude  = rng.uniform(0.05, 0.25)
        for s in affected:
            inflow_delta  = base[:, s] * magnitude * rng.uniform(0.5, 1.5)
            non_affected  = np.setdiff1d(np.arange(N), affected)
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

def load_bart_transfer_dataset(data_dir=DATA_DIR, n_synthetic=60, verbose=True):
    if verbose:
        print("=== Loading BART transfer dataset ===")

    od_paths = download_bart_od_matrices(data_dir)
    gtfs_dir = download_bart_gtfs(data_dir)

    od_matrices = {}
    for label, path in od_paths.items():
        try:
            stations, matrix = _load_od_excel(path)
            od_matrices[label] = (stations, matrix)
            if verbose:
                print(f"  Loaded {label}: {len(stations)} stations")
        except Exception as e:
            if verbose:
                print(f"  [warning] Failed to parse {label}: {e}")

    if not od_matrices:
        raise RuntimeError(
            "No BART OD matrices could be loaded. "
            "Check your internet connection or download them manually from "
            "https://www.bart.gov/about/reports/ridership"
        )

    ref_key      = "after_berryessa" if "after_berryessa" in od_matrices else next(iter(od_matrices))
    all_stations = od_matrices[ref_key][0]

    if verbose:
        print(f"  Reference station list: {len(all_stations)} stations")
        print("  Building graph from GTFS...")

    graph_data = build_bart_graph(gtfs_dir, all_stations)
    graph_data["n_nodes"]       = len(all_stations)
    graph_data["station_names"] = all_stations

    scenarios = []

    def _make_real_scenario(before_label, after_label, scenario_label, split):
        if before_label not in od_matrices or after_label not in od_matrices:
            if verbose:
                print(f"  [skip] {scenario_label}: missing OD files")
            return
        b_stations, b_matrix = od_matrices[before_label]
        a_stations, a_matrix = od_matrices[after_label]
        _, b_aligned, a_aligned = _align_matrices(b_stations, b_matrix, a_stations, a_matrix)
        _, b_final, a_final = _align_matrices(
            all_stations, b_aligned, all_stations, a_aligned
        )
        delta         = a_final - b_final
        node_features = compute_bart_node_features(b_final)
        std           = float(delta.std()) + 1e-8
        scenarios.append({
            "node_features":       node_features,
            "delta_od":            delta,
            "delta_od_normalized": delta / std,
            "std":  std,
            "is_real": True,
            "label":   scenario_label,
            "split":   split,
        })
        if verbose:
            print(f"  Real scenario '{scenario_label}': MAE={np.abs(delta).mean():.2f}, split={split}")

    _make_real_scenario("before_berryessa", "after_berryessa",
                        "berryessa_extension_2020", split="val")
    _make_real_scenario("before_antioch",   "after_antioch",
                        "antioch_extension_2023",  split="train")

    baseline_matrices = []
    for label in ["baseline_2019_01", "baseline_2019_06", "baseline_2019_10",
                  "before_berryessa"]:
        if label in od_matrices:
            b_st, b_mx = od_matrices[label]
            _, _, aligned = _align_matrices(
                b_st, b_mx,
                all_stations, np.zeros((len(all_stations), len(all_stations)))
            )
            baseline_matrices.append(aligned)

    if baseline_matrices:
        if verbose:
            print(f"  Generating {n_synthetic} synthetic scenarios...")
        synthetic_deltas = generate_bart_synthetic_scenarios(
            baseline_matrices, n_synthetic=n_synthetic
        )
        ref_od = baseline_matrices[0]
        for i, delta in enumerate(synthetic_deltas):
            std = float(delta.std()) + 1e-8
            scenarios.append({
                "node_features":       compute_bart_node_features(ref_od),
                "delta_od":            delta,
                "delta_od_normalized": delta / std,
                "std":   std,
                "is_real": False,
                "label":   f"synthetic_bart_{i:03d}",
                "split":   "train",
            })

    train_count = sum(1 for s in scenarios if s["split"] == "train")
    val_count   = sum(1 for s in scenarios if s["split"] == "val")
    real_count  = sum(1 for s in scenarios if s["is_real"])

    if verbose:
        print(f"\n  Total scenarios: {len(scenarios)} "
              f"({real_count} real, {len(scenarios)-real_count} synthetic)")
        print(f"  Train: {train_count}  |  Val: {val_count}")
        print("=== BART dataset ready ===\n")

    return graph_data, scenarios