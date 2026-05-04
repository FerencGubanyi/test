"""
utils/metr_la_loader.py
-----------------------
METR-LA benchmark dataset loader for architecture validation.

METR-LA: 207 loop detectors on LA highways, ~4 months of 5-min speed readings.
Graph: adjacency from sensor distances (Gaussian kernel, threshold 0.1).

Output format is intentionally compatible with the existing BKK pipeline:
  - node features:  (N, in_channels)   — matches GAT+LSTM node_feat input
  - edge_index:     (2, E)             — standard PyG format
  - edge_weight:    (E,)
  - targets:        (N,)               — per-node scalar (speed)

The "scenario" concept is mapped as follows for benchmark purposes:
  - Each sliding window of T=12 steps (1h) → "base state"  x
  - Next H=12 steps (1h)               → "target state"   y
  - A dummy scenario_mask of zeros is used (no topology change)

Usage:
    from utils.metr_la_loader import METRLADataset, get_metr_dataloaders
    train_loader, val_loader, test_loader, meta = get_metr_dataloaders()
"""

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

METR_LA_URL = (
    "https://zenodo.org/record/5146275/files/METR-LA.zip?download=1"
)
# Fallback: the original DCRNN repo hosts the same files
METR_LA_ALT_SPEED = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/"
    "data/sensor_graph/metr_la_speed.h5"
)
METR_LA_ALT_ADJ = (
    "https://raw.githubusercontent.com/liyaguang/DCRNN/master/"
    "data/sensor_graph/adj_mx.pkl"
)

CACHE_DIR = Path(os.environ.get("METR_LA_CACHE", "data/metr_la"))


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download url → dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Downloading {desc} …")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  ✗ Failed ({e})")
        return False


def _ensure_data() -> tuple[Path, Path]:
    """
    Returns (speed_h5_path, adj_pkl_path).
    Downloads from zenodo zip first; falls back to individual files from DCRNN repo.
    """
    speed_path = CACHE_DIR / "metr_la_speed.h5"
    adj_path   = CACHE_DIR / "adj_mx.pkl"

    if speed_path.exists() and adj_path.exists():
        return speed_path, adj_path

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- try zenodo zip ---
    zip_path = CACHE_DIR / "METR-LA.zip"
    if _download_file(METR_LA_URL, zip_path, "METR-LA.zip from Zenodo"):
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(CACHE_DIR)
            # look for the files inside extracted dirs
            for p in CACHE_DIR.rglob("*.h5"):
                p.rename(speed_path)
            for p in CACHE_DIR.rglob("adj_mx*"):
                p.rename(adj_path)
            if speed_path.exists() and adj_path.exists():
                return speed_path, adj_path
        except Exception as e:
            print(f"  Zip extract failed: {e}")

    # --- fallback: individual files from DCRNN repo ---
    print("Falling back to DCRNN GitHub raw files …")
    _download_file(METR_LA_ALT_SPEED, speed_path, "speed HDF5")
    _download_file(METR_LA_ALT_ADJ,   adj_path,   "adjacency pickle")

    if not speed_path.exists() or not adj_path.exists():
        raise RuntimeError(
            "Could not download METR-LA data. "
            "Please manually place metr_la_speed.h5 and adj_mx.pkl "
            f"in {CACHE_DIR} and re-run."
        )
    return speed_path, adj_path


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _load_adjacency(adj_path: Path, sigma2: float = 0.1, eps: float = 0.5):
    """
    Load the distance-based adjacency matrix and convert to PyG edge_index.

    adj_mx.pkl is a tuple: (sensor_ids, sensor_id_to_ind, adj_matrix)
    adj_matrix[i,j] = exp(-d²/σ²) if d > 0, else 0  (pre-computed in DCRNN).

    We threshold at eps and return sparse COO format.
    """
    import pickle
    with open(adj_path, "rb") as f:
        try:
            _, _, adj_mx = pickle.load(f, encoding="latin1")
        except Exception:
            f.seek(0)
            obj = pickle.load(f)
            adj_mx = obj[2] if isinstance(obj, (list, tuple)) else obj

    adj_mx = np.array(adj_mx, dtype=np.float32)
    np.fill_diagonal(adj_mx, 0)          # no self-loops (added later if needed)

    # threshold
    mask = adj_mx > eps
    rows, cols = np.where(mask)
    edge_index  = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(adj_mx[rows, cols], dtype=torch.float32)

    print(f"  Graph: 207 nodes, {edge_index.shape[1]} edges (eps={eps})")
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# Speed data loading & normalisation
# ---------------------------------------------------------------------------

def _load_speed(speed_path: Path):
    """Load HDF5 speed file → numpy array (T, N)."""
    try:
        import pandas as pd
        df = pd.read_hdf(speed_path)        # index=datetime, cols=sensor_ids
        return df.values.astype(np.float32) # (T, N)
    except Exception as e:
        raise RuntimeError(f"Could not read {speed_path}: {e}")


def _zscore(data: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = data.mean()
        std  = data.std() + 1e-8
    return (data - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class METRLAWindowDataset(Dataset):
    """
    Each sample = (x, y, edge_index, edge_weight)

      x : (N, T_in, F)   node features over T_in time steps
      y : (N, T_out)     target speeds over T_out future steps
      edge_index : (2, E)
      edge_weight: (E,)

    The model-facing interface flattens x → (N, T_in * F) for the GAT input,
    matching the existing BKK pipeline where node_feat is (N, in_channels).
    """

    def __init__(
        self,
        speed: np.ndarray,       # (T, N)
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        T_in:  int = 12,         # 12 × 5 min = 1 h history
        T_out: int = 12,         # 12 × 5 min = 1 h forecast
        flatten_x: bool = True,  # if True, x shape → (N, T_in)
    ):
        super().__init__()
        self.speed      = speed          # (T, N)  normalised
        self.edge_index = edge_index
        self.edge_weight= edge_weight
        self.T_in       = T_in
        self.T_out      = T_out
        self.flatten_x  = flatten_x
        self.N          = speed.shape[1]

        # valid start indices
        self.indices = list(range(T_in, len(speed) - T_out + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        # x: history window  (T_in, N) → transpose → (N, T_in)
        x = self.speed[t - self.T_in : t].T          # (N, T_in)
        # y: forecast target (T_out, N) → (N, T_out)
        y = self.speed[t : t + self.T_out].T          # (N, T_out)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # flatten if requested: (N, T_in) — directly usable as node_feat
        if self.flatten_x:
            node_feat = x          # (N, T_in)
        else:
            node_feat = x.unsqueeze(-1)  # (N, T_in, 1)

        # dummy scenario_mask (all zeros = no topology change)
        scenario_mask = torch.zeros(self.N, dtype=torch.float32)

        return {
            "node_feat":      node_feat,       # (N, T_in)
            "edge_index":     self.edge_index, # (2, E)
            "edge_weight":    self.edge_weight,# (E,)
            "target":         y[:, -1],        # (N,) — last-step target (simplest)
            "target_full":    y,               # (N, T_out) — multi-step
            "scenario_mask":  scenario_mask,
        }


# ---------------------------------------------------------------------------
# Collate + DataLoaders
# ---------------------------------------------------------------------------

def _collate_fn(batch):
    """
    Custom collate: keeps edge_index/edge_weight shared (same graph every sample).
    Stacks node_feat and target along a new batch dimension.

    Returns a dict that mimics what the BKK pipeline passes to the model,
    extended with a 'batch_size' key and a PyG Batch object for convenience.
    """
    # All samples share the same graph topology — use first sample's edges
    edge_index  = batch[0]["edge_index"]
    edge_weight = batch[0]["edge_weight"]
    N           = batch[0]["node_feat"].shape[0]
    B           = len(batch)

    node_feat     = torch.stack([s["node_feat"]    for s in batch], dim=0)  # (B, N, T_in)
    target        = torch.stack([s["target"]        for s in batch], dim=0)  # (B, N)
    target_full   = torch.stack([s["target_full"]   for s in batch], dim=0)  # (B, N, T_out)
    scenario_mask = torch.stack([s["scenario_mask"] for s in batch], dim=0)  # (B, N)

    return {
        "node_feat":      node_feat,
        "edge_index":     edge_index,
        "edge_weight":    edge_weight,
        "target":         target,
        "target_full":    target_full,
        "scenario_mask":  scenario_mask,
        "batch_size":     B,
        "num_nodes":      N,
    }


def get_metr_dataloaders(
    cache_dir:   str  = None,
    T_in:        int  = 12,
    T_out:       int  = 12,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.10,
    batch_size:  int  = 32,
    num_workers: int  = 0,
    adj_eps:     float = 0.5,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Download (if needed) and prepare METR-LA DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, meta_dict

    meta_dict contains:
        num_nodes   : 207
        in_channels : T_in   (node feature dimension)
        speed_mean  : float
        speed_std   : float
        edge_index  : torch.Tensor
        edge_weight : torch.Tensor
    """
    global CACHE_DIR
    if cache_dir:
        CACHE_DIR = Path(cache_dir)

    print("=== METR-LA Benchmark Loader ===")
    speed_path, adj_path = _ensure_data()

    print("Loading adjacency …")
    edge_index, edge_weight = _load_adjacency(adj_path, eps=adj_eps)

    print("Loading speed data …")
    speed_raw = _load_speed(speed_path)     # (T, N)
    T_total, N = speed_raw.shape
    print(f"  Speed matrix: {T_total} timesteps × {N} sensors")

    # chronological split (no shuffle!)
    t_train = int(T_total * train_ratio)
    t_val   = int(T_total * (train_ratio + val_ratio))

    train_speed = speed_raw[:t_train]
    val_speed   = speed_raw[t_train:t_val]
    test_speed  = speed_raw[t_val:]

    # normalise using train statistics
    train_norm, mean, std = _zscore(train_speed)
    val_norm,  _,    _   = _zscore(val_speed,  mean, std)
    test_norm, _,    _   = _zscore(test_speed, mean, std)

    print(f"  Train: {len(train_speed)} steps | Val: {len(val_speed)} | Test: {len(test_speed)}")
    print(f"  Normalisation  mean={mean:.2f}  std={std:.2f} (mph)")

    def make_loader(speed, shuffle):
        ds = METRLAWindowDataset(speed, edge_index, edge_weight, T_in, T_out)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_fn,
        )

    train_loader = make_loader(train_norm, shuffle=True)
    val_loader   = make_loader(val_norm,   shuffle=False)
    test_loader  = make_loader(test_norm,  shuffle=False)

    meta = {
        "num_nodes":   N,
        "in_channels": T_in,
        "out_channels": 1,       # predicting 1 value per node (last step)
        "speed_mean":  float(mean),
        "speed_std":   float(std),
        "edge_index":  edge_index,
        "edge_weight": edge_weight,
        "T_in":        T_in,
        "T_out":       T_out,
        "dataset":     "METR-LA",
    }

    print(f"  Loaders ready — train batches: {len(train_loader)}")
    return train_loader, val_loader, test_loader, meta