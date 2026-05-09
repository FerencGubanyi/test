"""
BART data adapter for transfer learning.

Downloads BART station-to-station OD matrices and GTFS, converts them
into the same format used by the BKK pipeline (zone features + adjacency
+ delta OD). Plugs directly into the existing training loop.

Network change events used:
  - Berryessa/North San Jose extension: June 13 2020 (+2 stations)
    → used as HELD-OUT validation (same role as M1 metro in BKK)
  - Antioch extension: November 10 2023 (+2 stations)
    → used as training scenario

Usage:
    from utils.bart_data import load_bart_transfer_dataset
    graph, scenarios = load_bart_transfer_dataset()
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BART_OD_BASE = "https://www.bart.gov/sites/default/files/docs/ridership"

# Monthly OD Excel filenames follow the pattern: YYYY_MM_average_weekday.xlsx
# We download specific months bracketing each network change.
# Berryessa extension: June 13 2020
#   before: Feb 2020 (pre-COVID stable)
#   after:  Sep 2020 (settled, post-opening)
# Antioch extension: Nov 10 2023
#   before: Sep 2023
#   after:  Jan 2024
MONTHS_TO_FETCH = {
    "before_berryessa": ("2020", "02"),
    "after_berryessa":  ("2020", "09"),
    "before_antioch":   ("2023", "09"),
    "after_antioch":    ("2024", "01"),
    # Extra baseline months for synthetic scenario generation
    "baseline_2019_01": ("2019", "01"),
    "baseline_2019_06": ("2019", "06"),
    "baseline_2019_10": ("2019", "10"),
}

BART_GTFS_URL = "https://www.bart.gov/dev/schedules/google_transit.zip"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "bart")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _download_file(url: str, dest_path: str, desc: str = ""):
    if os.path.exists(dest_path):
        print(f"  [cached] {desc or dest_path}")
        return
    print(f"  [download] {desc or url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)


def download_bart_od_matrices(data_dir: str = DATA_DIR) -> dict[str, str]:
    """
    Download BART monthly OD Excel files.
    Returns dict: label -> local file path.
    Falls back to synthetic generation if download fails (BART site may
    restructure URLs; format has changed historically).
    """
    _ensure_dir(data_dir)
    paths = {}
    for label, (year, month) in MONTHS_TO_FETCH.items():
        # BART has used several URL patterns over the years; try both.
        filename = f"{year}_{month}_average_weekday.xlsx"
        dest = os.path.join(data_dir, filename)

        candidate_urls = [
            f"{BART_OD_BASE}/{filename}",
            f"https://www.bart.gov/sites/default/files/docs/{filename}",
        ]

        downloaded = False
        for url in candidate_urls:
            try:
                _download_file(url, dest, desc=f"BART OD {year}-{month}")
                downloaded = True
                break
            except Exception:
                continue

        if downloaded:
            paths[label] = dest
        else:
            print(f"  [warning] Could not download {filename}; "
                  f"will use synthetic fallback for '{label}'")

    return paths


def download_bart_gtfs(data_dir: str = DATA_DIR) -> str:
    """Download and unzip BART GTFS feed. Returns path to extracted folder."""
    _ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "google_transit.zip")
    gtfs_dir = os.path.join(data_dir, "gtfs")

    _download_file(BART_GTFS_URL, zip_path, desc="BART GTFS")

    if not os.path.exists(gtfs_dir):
        print("  [extract] BART GTFS")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(gtfs_dir)

    return gtfs_dir


# ---------------------------------------------------------------------------
# OD matrix loading
# ---------------------------------------------------------------------------

def _load_od_excel(path: str) -> tuple[list[str], np.ndarray]:
    """
    Load a BART OD Excel file.
    Returns (station_names, matrix) where matrix shape is (N, N).

    BART format (as of 2023):
      - Row 1: header row with destination station names
      - Column 1: origin station names
      - Values: average weekday trips
    """
    df = pd.read_excel(path, header=0, index_col=0)
    # Drop any totally empty rows/cols (formatting artefacts)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna(0)
    stations = list(df.index.astype(str))
    matrix = df.values.astype(np.float32)
    return stations, matrix


def _align_matrices(
    before_stations: list[str],
    before_matrix: np.ndarray,
    after_stations: list[str],
    after_matrix: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Align before/after OD matrices to a common station list.
    New stations in 'after' that are absent in 'before' get zero rows/cols.
    Returns (all_stations, before_aligned, after_aligned).
    """
    all_stations = sorted(set(before_stations) | set(after_stations))
    N = len(all_stations)
    idx_before = {s: i for i, s in enumerate(before_stations)}
    idx_after  = {s: i for i, s in enumerate(after_stations)}

    before_aligned = np.zeros((N, N), dtype=np.float32)
    after_aligned  = np.zeros((N, N), dtype=np.float32)

    for i, s_i in enumerate(all_stations):
        for j, s_j in enumerate(all_stations):
            if s_i in idx_before and s_j in idx_before:
                before_aligned[i, j] = before_matrix[
                    idx_before[s_i], idx_before[s_j]
                ]
            if s_i in idx_after and s_j in idx_after:
                after_aligned[i, j] = after_matrix[
                    idx_after[s_i], idx_after[s_j]
                ]

    return all_stations, before_aligned, after_aligned


# ---------------------------------------------------------------------------
# Graph construction from GTFS
# ---------------------------------------------------------------------------

def build_bart_graph(gtfs_dir: str, stations: list[str]) -> dict:
    """
    Build a zone-level graph from BART GTFS.
    Nodes = stations (treated as zones, analogous to BKK zones).
    Edges = consecutive stop pairs on any route.

    Returns dict with keys:
        edge_index  : (2, E) LongTensor
        edge_attr   : (E, F_edge) FloatTensor  [n_routes, avg_headway_min]
        node_features: (N, F_node) FloatTensor  (filled later by feature engineering)
        station_to_idx: dict str -> int
        hyperedge_index: (2, total_incidences) — stations × routes membership
    """
    stops_df = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"))
    trips_df = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
    stop_times_df = pd.read_csv(
        os.path.join(gtfs_dir, "stop_times.txt"),
        usecols=["trip_id", "stop_id", "stop_sequence"],
    )
    routes_df = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"))

    # Map stop_id -> stop_name (BART stop names ≈ station names in OD matrix)
    stop_name_map = dict(zip(stops_df["stop_id"], stops_df["stop_name"]))

    # Station index
    station_to_idx = {s: i for i, s in enumerate(stations)}
    N = len(stations)

    # Build edges: for each trip, consecutive stops that are both in station list
    stop_times_sorted = stop_times_df.sort_values(["trip_id", "stop_sequence"])
    trip_to_route = dict(zip(trips_df["trip_id"], trips_df["route_id"]))

    edges = {}        # (i, j) -> set of route_ids
    hyperedges = {}   # route_id -> set of station indices

    current_trip = None
    prev_idx = None
    prev_stop = None

    for _, row in stop_times_sorted.iterrows():
        trip_id = row["trip_id"]
        stop_id = row["stop_id"]
        stop_name = stop_name_map.get(stop_id, "")

        # Fuzzy match stop_name to station list
        matched = _fuzzy_match_station(stop_name, stations)
        node_idx = station_to_idx.get(matched)
        route_id = trip_to_route.get(trip_id, "unknown")

        if node_idx is not None:
            if route_id not in hyperedges:
                hyperedges[route_id] = set()
            hyperedges[route_id].add(node_idx)

        if trip_id != current_trip:
            current_trip = trip_id
            prev_idx = node_idx
            prev_stop = stop_name
            continue

        if prev_idx is not None and node_idx is not None and prev_idx != node_idx:
            key = (min(prev_idx, node_idx), max(prev_idx, node_idx))
            if key not in edges:
                edges[key] = set()
            edges[key].add(route_id)

        prev_idx = node_idx
        prev_stop = stop_name

    # Build edge_index and edge_attr
    if edges:
        edge_list = list(edges.keys())
        edge_index = torch.tensor(
            [[e[0] for e in edge_list] + [e[1] for e in edge_list],
             [e[1] for e in edge_list] + [e[0] for e in edge_list]],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [[len(edges[e]), 5.0] for e in edge_list] * 2,  # n_routes, dummy headway
            dtype=torch.float,
        )
    else:
        # Fallback: fully connected graph (BART is a simple linear system)
        src, dst = [], []
        for i in range(N):
            for j in range(i + 1, N):
                src += [i, j]
                dst += [j, i]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.ones(len(src), 2, dtype=torch.float)

    # Build hyperedge incidence
    he_station, he_route = [], []
    route_ids = sorted(hyperedges.keys())
    for r_idx, route_id in enumerate(route_ids):
        for s_idx in hyperedges[route_id]:
            he_station.append(s_idx)
            he_route.append(r_idx)

    hyperedge_index = torch.tensor(
        [he_station, he_route], dtype=torch.long
    ) if he_station else torch.zeros(2, 0, dtype=torch.long)

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "station_to_idx": station_to_idx,
        "hyperedge_index": hyperedge_index,
        "n_hyperedges": len(route_ids),
    }


def _fuzzy_match_station(name: str, stations: list[str]) -> str:
    """
    Case-insensitive partial match of a GTFS stop name to a station in the OD list.
    BART GTFS stop names often differ slightly from OD matrix headers.
    """
    name_lower = name.lower().strip()
    # Exact match first
    for s in stations:
        if s.lower().strip() == name_lower:
            return s
    # Prefix / substring match
    for s in stations:
        if name_lower in s.lower() or s.lower() in name_lower:
            return s
    return name  # no match — will be filtered out by station_to_idx lookup


# ---------------------------------------------------------------------------
# Node feature engineering (mirrors BKK utils/data.py logic)
# ---------------------------------------------------------------------------

def compute_bart_node_features(od_matrix: np.ndarray) -> np.ndarray:
    """
    Compute 16-dim node feature vector per station from OD matrix.
    Identical structure to BKK zone feature engineering so transferred
    embedding weights are compatible.
    """
    N = od_matrix.shape[0]
    features = []
    for i in range(N):
        row = od_matrix[i, :]      # outgoing
        col = od_matrix[:, i]      # incoming
        feat = [
            float(row.sum()),
            float(col.sum()),
            float(row.mean()),
            float(row.std() + 1e-8),
            float(col.mean()),
            float(col.std() + 1e-8),
            float((row > 0).sum()),
            float((col > 0).sum()),
            float(row.max()),
            float(col.max()),
            float(np.percentile(row, 75)),
            float(np.percentile(col, 75)),
            float(np.percentile(row, 25)),
            float(np.percentile(col, 25)),
            float(row.sum() / (col.sum() + 1e-6)),
            float(np.log1p(max(row.sum(), 0))),
        ]
        features.append(feat)

    features = np.array(features, dtype=np.float32)

    # Z-score normalisation (same as BKK pipeline)
    mean = features.mean(axis=0)
    std  = features.std(axis=0) + 1e-8
    features = (features - mean) / std
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


# ---------------------------------------------------------------------------
# Synthetic ΔOD scenarios from BART baseline months
# ---------------------------------------------------------------------------

def generate_bart_synthetic_scenarios(
    baseline_matrices: list[np.ndarray],
    n_synthetic: int = 60,
    rng_seed: int = 42,
) -> list[np.ndarray]:
    """
    Generate synthetic ΔOD scenarios from BART baseline OD matrices.
    Uses same gravity-model-inspired redistribution as BKK synthetic generator.
    Produces 'what if a station's connectivity changed' perturbations.
    """
    rng = np.random.default_rng(rng_seed)
    scenarios = []

    for _ in range(n_synthetic):
        base = baseline_matrices[rng.integers(len(baseline_matrices))].copy()
        N = base.shape[0]

        # Pick a random subset of stations to "modify"
        n_affected = rng.integers(2, max(3, N // 4))
        affected = rng.choice(N, size=n_affected, replace=False)

        delta = np.zeros_like(base)
        magnitude = rng.uniform(0.05, 0.25)

        for s in affected:
            # Inflow increase to affected stations
            inflow_delta = base[:, s] * magnitude * rng.uniform(0.5, 1.5)
            # Compensate: reduce inflow from nearby stations (redistribution)
            non_affected = np.setdiff1d(np.arange(N), affected)
            if len(non_affected) > 0:
                reduction = inflow_delta.sum() * 0.7 / len(non_affected)
                delta[non_affected, s] -= reduction
            delta[affected, s] += inflow_delta[affected]

        # Enforce approximate conservation
        delta -= delta.mean()

        # Validate: skip if NaN/inf
        if not np.all(np.isfinite(delta)):
            continue

        scenarios.append(delta)

    return scenarios


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_bart_transfer_dataset(
    data_dir: str = DATA_DIR,
    n_synthetic: int = 60,
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """
    Full BART dataset loader for transfer learning.

    Returns:
        graph_data : dict with edge_index, edge_attr, hyperedge_index, n_hyperedges
        scenarios  : list of dicts, each with keys:
                       'node_features'  : (N, 16) np.ndarray
                       'delta_od'       : (N, N)  np.ndarray
                       'is_real'        : bool
                       'label'          : str
                       'split'          : 'train' | 'val'

    The Berryessa scenario is tagged split='val' (held out).
    All others are split='train'.
    """
    if verbose:
        print("=== Loading BART transfer dataset ===")

    # 1. Download data
    od_paths = download_bart_od_matrices(data_dir)
    gtfs_dir = download_bart_gtfs(data_dir)

    # 2. Load OD matrices that were successfully downloaded
    od_matrices = {}
    for label, path in od_paths.items():
        try:
            stations, matrix = _load_od_excel(path)
            od_matrices[label] = (stations, matrix)
            if verbose:
                print(f"  Loaded {label}: {len(stations)} stations")
        except Exception as e:
            print(f"  [warning] Failed to parse {label}: {e}")

    # 3. Build unified station list from all available matrices
    all_station_sets = [set(v[0]) for v in od_matrices.values()]
    if not all_station_sets:
        raise RuntimeError(
            "No BART OD matrices could be loaded. "
            "Check your internet connection or download them manually from "
            "https://www.bart.gov/about/reports/ridership"
        )

    # Use the after_berryessa matrix as reference (most complete station list)
    ref_key = "after_berryessa" if "after_berryessa" in od_matrices else \
              next(iter(od_matrices))
    all_stations = od_matrices[ref_key][0]

    if verbose:
        print(f"  Reference station list: {len(all_stations)} stations")

    # 4. Build graph from GTFS
    if verbose:
        print("  Building graph from GTFS...")
    graph_data = build_bart_graph(gtfs_dir, all_stations)
    graph_data["n_nodes"] = len(all_stations)
    graph_data["station_names"] = all_stations

    # 5. Build real ΔOD scenarios
    scenarios = []

    def _make_real_scenario(before_label, after_label, scenario_label, split):
        if before_label not in od_matrices or after_label not in od_matrices:
            if verbose:
                print(f"  [skip] {scenario_label}: missing OD files")
            return

        b_stations, b_matrix = od_matrices[before_label]
        a_stations, a_matrix = od_matrices[after_label]
        _, b_aligned, a_aligned = _align_matrices(
            b_stations, b_matrix, a_stations, a_matrix
        )

        # Re-align to reference station list
        _, b_final, a_final = _align_matrices(
            all_stations, b_aligned,
            all_stations, a_aligned,
        )

        delta = a_final - b_final
        node_features = compute_bart_node_features(b_final)
        std = float(delta.std()) + 1e-8

        scenarios.append({
            "node_features": node_features,
            "delta_od": delta,
            "delta_od_normalized": delta / std,
            "std": std,
            "is_real": True,
            "label": scenario_label,
            "split": split,
        })
        if verbose:
            print(f"  Real scenario '{scenario_label}': "
                  f"MAE={np.abs(delta).mean():.2f}, split={split}")

    # Berryessa = validation (held out, like M1 in BKK)
    _make_real_scenario(
        "before_berryessa", "after_berryessa",
        "berryessa_extension_2020", split="val"
    )

    # Antioch = training
    _make_real_scenario(
        "before_antioch", "after_antioch",
        "antioch_extension_2023", split="train"
    )

    # 6. Generate synthetic scenarios from baselines
    baseline_matrices = []
    for label in ["baseline_2019_01", "baseline_2019_06", "baseline_2019_10"]:
        if label in od_matrices:
            _, matrix = od_matrices[label]
            _, _, aligned = _align_matrices(
                od_matrices[label][0], matrix,
                all_stations, np.zeros((len(all_stations), len(all_stations)))
            )
            baseline_matrices.append(aligned)

    # Also use the before_berryessa as baseline if available
    if "before_berryessa" in od_matrices:
        b_stations, b_matrix = od_matrices["before_berryessa"]
        _, _, b_aligned = _align_matrices(
            b_stations, b_matrix,
            all_stations, np.zeros((len(all_stations), len(all_stations)))
        )
        baseline_matrices.append(b_aligned)

    if baseline_matrices:
        if verbose:
            print(f"  Generating {n_synthetic} synthetic scenarios...")
        synthetic_deltas = generate_bart_synthetic_scenarios(
            baseline_matrices, n_synthetic=n_synthetic
        )
        ref_od = baseline_matrices[0]
        for i, delta in enumerate(synthetic_deltas):
            node_features = compute_bart_node_features(ref_od)
            std = float(delta.std()) + 1e-8
            scenarios.append({
                "node_features": node_features,
                "delta_od": delta,
                "delta_od_normalized": delta / std,
                "std": std,
                "is_real": False,
                "label": f"synthetic_bart_{i:03d}",
                "split": "train",
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
