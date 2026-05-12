"""
BART data adapter for transfer learning.
Reads directly from the ridership_YYYY.zip files stored in the repo
under data/bart/ — no internet required, no Drive needed.

Expected repo layout:
    data/bart/ridership_2019.zip   (contains Ridership_201901.xlsx etc.)
    data/bart/ridership_2020.zip
    data/bart/ridership_2023.zip
    data/bart/ridership_2024.zip   (optional — only needed for antioch scenario)
    data/bart/google_transit.zip   (BART GTFS — downloaded once automatically)
"""

import os
import io
import zipfile
import requests
import warnings
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BART_GTFS_URL = "https://www.bart.gov/dev/schedules/google_transit.zip"
DATA_DIR      = "data/bart"

# Labels → (zip_year, month) mapping
# zip_year: which ridership_YYYY.zip to open
# month:    two-digit month string
MONTHS_TO_FETCH = {
    "before_berryessa": ("2020", "02"),   # Feb 2020, before June 2020 opening
    "after_berryessa":  ("2020", "09"),   # Sep 2020, post-opening
    "before_antioch":   ("2023", "09"),   # Sep 2023, before eBART extension
    "after_antioch":    ("2024", "01"),   # Jan 2024, post-antioch (optional)
    "baseline_2019_01": ("2019", "01"),
    "baseline_2019_06": ("2019", "06"),
    "baseline_2019_10": ("2019", "10"),
}

# Station codes that are totals rows/cols, not real stations
_NON_STATION_LABELS = {"exits", "entries", "total", "nan", "", "none"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _is_valid_station(s: str) -> bool:
    sl = s.lower().strip()
    if sl in _NON_STATION_LABELS:
        return False
    if len(s) > 6 or len(s) == 0:
        return False
    if s.startswith("#"):
        return False
    if s.isdigit() and len(s) > 4:
        return False
    return True


# ---------------------------------------------------------------------------
# OD loading — reads directly from zip, no extraction needed
# ---------------------------------------------------------------------------

def _load_od_from_zip(zip_path: str, year: str, month: str) -> tuple:
    """
    Open ridership_YYYY.zip, find Ridership_YYYYmm.xlsx inside,
    parse and return (stations, matrix).

    File format (confirmed from real BART data):
      Sheet 'Avg Weekday OD':
        Row 0: header labels (ignored)
        Row 1: col 0 = NaN, cols 1..N = station codes, last col = 'Exits'
        Rows 2..N+1: col 0 = exit code, cols 1..N = float OD, last col = row total
        Last row: 'Entries' column totals
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Zip not found: {zip_path}\n"
            f"Upload ridership_{year}.zip to data/bart/ in the repo."
        )

    # Filename inside the zip: Ridership_YYYYmm.xlsx
    inner_name = f"ridership_{year}/Ridership_{year}{month}.xlsx"

    with zipfile.ZipFile(zip_path, "r") as zf:
        available = zf.namelist()
        # Be flexible about the inner folder name
        matches = [n for n in available
                   if n.endswith(f"Ridership_{year}{month}.xlsx")]
        if not matches:
            raise FileNotFoundError(
                f"Could not find Ridership_{year}{month}.xlsx in {zip_path}.\n"
                f"Available: {[n for n in available if n.endswith('.xlsx')][:5]}"
            )
        with zf.open(matches[0]) as f:
            raw = f.read()

    xl = pd.ExcelFile(io.BytesIO(raw))
    candidates = ["Avg Weekday OD", "Average Weekday", "Avg Weekday"]
    sheet = next((s for s in candidates if s in xl.sheet_names), xl.sheet_names[0])

    df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet, header=None)

    # Row 1: station codes (col 0 = NaN, last col = 'Exits' — both filtered out)
    raw_stations = [str(s).strip() for s in df.iloc[1, 1:].tolist()]
    stations = [s for s in raw_stations if _is_valid_station(s)]
    N = len(stations)

    if N == 0:
        raise ValueError(
            f"No valid stations in {zip_path} / {year}{month}. "
            f"Row-1 sample: {raw_stations[:8]}"
        )

    # OD data: rows 2..2+N, cols 1..1+N (excludes Exits totals col)
    matrix_raw = df.iloc[2:2 + N, 1:1 + N]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = (
            matrix_raw
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values
            .astype(np.float32)
        )

    if matrix.shape != (N, N):
        raise ValueError(
            f"Matrix shape mismatch: got {matrix.shape}, expected ({N},{N})"
        )

    return stations, matrix


def _load_od_matrices(data_dir: str, verbose: bool) -> dict:
    """
    Load all MONTHS_TO_FETCH entries from the zip files in data_dir.
    Returns dict: label → (stations, matrix).
    Missing zips are skipped with a warning (not a crash).
    """
    od_matrices = {}
    for label, (year, month) in MONTHS_TO_FETCH.items():
        zip_path = os.path.join(data_dir, f"ridership_{year}.zip")
        try:
            stations, matrix = _load_od_from_zip(zip_path, year, month)
            od_matrices[label] = (stations, matrix)
            if verbose:
                print(f"  [ok] {label}: {len(stations)} stations, "
                      f"total={matrix.sum():,.0f} trips")
        except FileNotFoundError as e:
            if verbose:
                print(f"  [skip] {label}: {e}")
        except Exception as e:
            if verbose:
                print(f"  [warning] {label}: {e}")
    return od_matrices


# ---------------------------------------------------------------------------
# GTFS download (only needed once; falls back to fully-connected graph)
# ---------------------------------------------------------------------------

def _download_gtfs(data_dir: str, verbose: bool) -> str:
    _ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "google_transit.zip")
    gtfs_dir = os.path.join(data_dir, "gtfs")

    if not os.path.exists(zip_path):
        if verbose:
            print("  [download] BART GTFS from bart.gov...")
        try:
            r = requests.get(BART_GTFS_URL, timeout=30)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            if verbose:
                print(f"  [warning] GTFS download failed: {e} "
                      f"— will use fully-connected fallback graph")
            return gtfs_dir
    else:
        if verbose:
            print("  [cached] BART GTFS")

    if not os.path.exists(gtfs_dir) and os.path.exists(zip_path):
        if verbose:
            print("  [extract] BART GTFS")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(gtfs_dir)

    return gtfs_dir


# ---------------------------------------------------------------------------
# Matrix alignment
# ---------------------------------------------------------------------------

def _align_matrices(before_stations, before_matrix, after_stations, after_matrix):
    all_stations = sorted(set(before_stations) | set(after_stations))
    N = len(all_stations)
    idx_b = {s: i for i, s in enumerate(before_stations)}
    idx_a = {s: i for i, s in enumerate(after_stations)}

    before_aligned = np.zeros((N, N), dtype=np.float32)
    after_aligned  = np.zeros((N, N), dtype=np.float32)

    for i, si in enumerate(all_stations):
        for j, sj in enumerate(all_stations):
            if si in idx_b and sj in idx_b:
                before_aligned[i, j] = before_matrix[idx_b[si], idx_b[sj]]
            if si in idx_a and sj in idx_a:
                after_aligned[i, j]  = after_matrix[idx_a[si],  idx_a[sj]]

    return all_stations, before_aligned, after_aligned


# ---------------------------------------------------------------------------
# Graph from GTFS
# ---------------------------------------------------------------------------

def _fuzzy_match(name: str, stations: list) -> str:
    nl = name.lower().strip()
    for s in stations:
        if s.lower().strip() == nl:
            return s
    for s in stations:
        if nl in s.lower() or s.lower() in nl:
            return s
    return name


def build_bart_graph(gtfs_dir: str, stations: list) -> dict:
    def _fallback():
        N = len(stations)
        src, dst = [], []
        for i in range(N):
            for j in range(i + 1, N):
                src += [i, j]; dst += [j, i]
        return {
            "edge_index":      torch.tensor([src, dst], dtype=torch.long),
            "edge_attr":       torch.ones(len(src), 2, dtype=torch.float),
            "station_to_idx":  {s: i for i, s in enumerate(stations)},
            "hyperedge_index": torch.zeros(2, 0, dtype=torch.long),
            "n_hyperedges":    0,
        }

    try:
        stops_df      = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"))
        trips_df      = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
        stop_times_df = pd.read_csv(
            os.path.join(gtfs_dir, "stop_times.txt"),
            usecols=["trip_id", "stop_id", "stop_sequence"],
        )
    except Exception as e:
        print(f"  [warning] GTFS parse failed: {e} — using fully-connected graph")
        return _fallback()

    stop_name_map  = dict(zip(stops_df["stop_id"], stops_df["stop_name"]))
    station_to_idx = {s: i for i, s in enumerate(stations)}
    trip_to_route  = dict(zip(trips_df["trip_id"], trips_df["route_id"]))

    edges, hyperedges = {}, {}
    current_trip, prev_idx = None, None

    for _, row in stop_times_df.sort_values(["trip_id", "stop_sequence"]).iterrows():
        trip_id  = row["trip_id"]
        sname    = stop_name_map.get(row["stop_id"], "")
        matched  = _fuzzy_match(sname, stations)
        node_idx = station_to_idx.get(matched)
        route_id = trip_to_route.get(trip_id, "unknown")

        if node_idx is not None:
            hyperedges.setdefault(route_id, set()).add(node_idx)

        if trip_id != current_trip:
            current_trip, prev_idx = trip_id, node_idx
            continue

        if prev_idx is not None and node_idx is not None and prev_idx != node_idx:
            key = (min(prev_idx, node_idx), max(prev_idx, node_idx))
            edges.setdefault(key, set()).add(route_id)

        prev_idx = node_idx

    if not edges:
        return _fallback()

    edge_list  = list(edges.keys())
    edge_index = torch.tensor(
        [[e[0] for e in edge_list] + [e[1] for e in edge_list],
         [e[1] for e in edge_list] + [e[0] for e in edge_list]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [[len(edges[e]), 5.0] for e in edge_list] * 2, dtype=torch.float
    )

    route_ids   = sorted(hyperedges.keys())
    he_s, he_r  = [], []
    for r_idx, rid in enumerate(route_ids):
        for s_idx in hyperedges[rid]:
            he_s.append(s_idx); he_r.append(r_idx)

    return {
        "edge_index":      edge_index,
        "edge_attr":       edge_attr,
        "station_to_idx":  station_to_idx,
        "hyperedge_index": (torch.tensor([he_s, he_r], dtype=torch.long)
                            if he_s else torch.zeros(2, 0, dtype=torch.long)),
        "n_hyperedges":    len(route_ids),
    }


# ---------------------------------------------------------------------------
# Node features  (mirrors utils/data.py od_matrix_to_zone_features)
# ---------------------------------------------------------------------------

def compute_bart_node_features(od_matrix: np.ndarray) -> np.ndarray:
    N = od_matrix.shape[0]
    features = []
    for i in range(N):
        row = od_matrix[i, :]
        col = od_matrix[:, i]
        features.append([
            float(row.sum()), float(col.sum()),
            float(row.mean()), float(row.std() + 1e-8),
            float(col.mean()), float(col.std() + 1e-8),
            float((row > 0).sum()), float((col > 0).sum()),
            float(row.max()), float(col.max()),
            float(np.percentile(row, 75)), float(np.percentile(col, 75)),
            float(np.percentile(row, 25)), float(np.percentile(col, 25)),
            float(row.sum() / (col.sum() + 1e-6)),
            float(np.log1p(max(row.sum(), 0))),
        ])
    feats = np.array(features, dtype=np.float32)
    mean  = feats.mean(axis=0)
    std   = feats.std(axis=0) + 1e-8
    return np.nan_to_num((feats - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Synthetic scenarios
# ---------------------------------------------------------------------------

def generate_bart_synthetic_scenarios(baseline_matrices, n_synthetic=60, rng_seed=42):
    rng       = np.random.default_rng(rng_seed)
    scenarios = []
    for _ in range(n_synthetic):
        base       = baseline_matrices[rng.integers(len(baseline_matrices))].copy()
        N          = base.shape[0]
        n_affected = rng.integers(2, max(3, N // 4))
        affected   = rng.choice(N, size=n_affected, replace=False)
        delta      = np.zeros_like(base)
        magnitude  = rng.uniform(0.05, 0.25)
        for s in affected:
            inflow_delta = base[:, s] * magnitude * rng.uniform(0.5, 1.5)
            non_affected = np.setdiff1d(np.arange(N), affected)
            if len(non_affected) > 0:
                delta[non_affected, s] -= inflow_delta.sum() * 0.7 / len(non_affected)
            delta[affected, s] += inflow_delta[affected]
        delta -= delta.mean()
        if np.all(np.isfinite(delta)):
            scenarios.append(delta)
    return scenarios


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_bart_transfer_dataset(data_dir=DATA_DIR, n_synthetic=60, verbose=True):
    if verbose:
        print("=== Loading BART transfer dataset ===")
        print(f"  Reading from: {os.path.abspath(data_dir)}")

    od_matrices = _load_od_matrices(data_dir, verbose)

    if not od_matrices:
        raise RuntimeError(
            "No BART OD matrices could be loaded.\n"
            f"Place ridership_YYYY.zip files in {os.path.abspath(data_dir)}/\n"
            "Required zips: ridership_2019.zip, ridership_2020.zip\n"
            "Download from: https://www.bart.gov/about/reports/ridership"
        )

    gtfs_dir = _download_gtfs(data_dir, verbose)

    # Reference station list from the first successfully loaded matrix
    ref_key = next(
        (k for k in ["after_berryessa", "before_berryessa"] if k in od_matrices),
        next(iter(od_matrices))
    )
    all_stations = od_matrices[ref_key][0]

    if verbose:
        print(f"  Reference: '{ref_key}' — {len(all_stations)} stations")
        print("  Building graph from GTFS...")

    graph_data = build_bart_graph(gtfs_dir, all_stations)
    graph_data["n_nodes"]       = len(all_stations)
    graph_data["station_names"] = all_stations

    scenarios = []

    def _make_real_scenario(before_lbl, after_lbl, name, split):
        if before_lbl not in od_matrices or after_lbl not in od_matrices:
            if verbose:
                print(f"  [skip] {name}: missing '{before_lbl}' or '{after_lbl}'")
            return
        b_st, b_mx = od_matrices[before_lbl]
        a_st, a_mx = od_matrices[after_lbl]
        _, b_al, a_al = _align_matrices(b_st, b_mx, a_st, a_mx)
        _, b_fin, a_fin = _align_matrices(all_stations, b_al, all_stations, a_al)
        delta = a_fin - b_fin
        std   = float(delta.std()) + 1e-8
        scenarios.append({
            "node_features":       compute_bart_node_features(b_fin),
            "delta_od":            delta,
            "delta_od_normalized": delta / std,
            "std":     std,
            "is_real": True,
            "label":   name,
            "split":   split,
        })
        if verbose:
            print(f"  Real scenario '{name}': "
                  f"MAE={np.abs(delta).mean():.2f}, split={split}")

    _make_real_scenario("before_berryessa", "after_berryessa",
                        "berryessa_extension_2020", split="val")
    _make_real_scenario("before_antioch",   "after_antioch",
                        "antioch_extension_2023",   split="train")

    # Baseline matrices for synthetic generation
    baseline_matrices = []
    for lbl in ["baseline_2019_01", "baseline_2019_06",
                "baseline_2019_10", "before_berryessa"]:
        if lbl in od_matrices:
            b_st, b_mx = od_matrices[lbl]
            _, b_al, _ = _align_matrices(
                b_st, b_mx,
                all_stations, np.zeros((len(all_stations), len(all_stations)))
            )
            baseline_matrices.append(b_al)

    if baseline_matrices:
        if verbose:
            print(f"  Generating {n_synthetic} synthetic scenarios...")
        ref_od = baseline_matrices[0]
        for i, delta in enumerate(
            generate_bart_synthetic_scenarios(baseline_matrices, n_synthetic)
        ):
            std = float(delta.std()) + 1e-8
            scenarios.append({
                "node_features":       compute_bart_node_features(ref_od),
                "delta_od":            delta,
                "delta_od_normalized": delta / std,
                "std":     std,
                "is_real": False,
                "label":   f"synthetic_bart_{i:03d}",
                "split":   "train",
            })

    train_c = sum(1 for s in scenarios if s["split"] == "train")
    val_c   = sum(1 for s in scenarios if s["split"] == "val")
    real_c  = sum(1 for s in scenarios if s["is_real"])

    if verbose:
        print(f"\n  Total: {len(scenarios)} scenarios "
              f"({real_c} real, {len(scenarios)-real_c} synthetic)")
        print(f"  Train: {train_c}  |  Val: {val_c}")
        print("=== BART dataset ready ===\n")

    return graph_data, scenarios