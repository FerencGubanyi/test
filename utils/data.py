import numpy as np
import pandas as pd
import torch
from typing import List, Optional


def load_od_matrix_with_header(filepath: str) -> pd.DataFrame:
    """datas from BKK simulations"""
    df       = pd.read_excel(filepath, header=None)
    zone_ids = df.iloc[0, 3:].values.astype(int)
    data     = df.iloc[3:, :]
    row_ids  = data.iloc[:, 0].values.astype(int)
    matrix   = data.iloc[:, 3:].values.astype(float)
    print(f'  {filepath.split("/")[-1]}: {len(row_ids)}x{len(zone_ids)}')
    return pd.DataFrame(matrix, index=row_ids, columns=zone_ids)


def load_od_matrix_no_header(filepath: str, zone_ids: List[int]) -> pd.DataFrame:
    """datas from VISUM"""
    df     = pd.read_excel(filepath, header=None)
    matrix = df.values.astype(float)
    assert matrix.shape == (len(zone_ids), len(zone_ids)), \
        f'Size error: {matrix.shape} != {len(zone_ids)}x{len(zone_ids)}'
    return pd.DataFrame(matrix, index=zone_ids, columns=zone_ids)


def load_od_matrix_sheet(filepath: str, sheet_name: str,
                          zone_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """
    More data from BKK, but with different shape
    sheet_name: 'kk' vagy 'dm-diff'
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    if zone_ids is None:
        zone_ids = df.iloc[0, 3:].values.astype(int)
        row_ids  = df.iloc[3:, 0].values.astype(int)
        matrix   = df.iloc[3:, 3:].values.astype(float)
    else:
        row_ids = zone_ids
        matrix  = df.values.astype(float)
    print(f'  {filepath.split("/")[-1]} [{sheet_name}]: {len(row_ids)}x{len(zone_ids)}')
    return pd.DataFrame(matrix, index=row_ids, columns=zone_ids)


NUM_FEATURES = 16

def od_matrix_to_zone_features(od_matrix: pd.DataFrame,
                                in_channels: int = NUM_FEATURES) -> torch.Tensor:
    """OD matrices → normalized zone feature vektors (N x in_channels)."""
    features = []
    for zone_id in od_matrix.index:
        row = od_matrix.loc[zone_id].values.astype(float)
        col = (od_matrix[zone_id].values.astype(float)
               if zone_id in od_matrix.columns
               else np.zeros(len(od_matrix)))
        feat = [
            row.sum(), col.sum(),
            row.mean(), row.std(),
            col.mean(), col.std(),
            (row > 0).sum(), (col > 0).sum(),
            row.max(), col.max(),
            np.percentile(row, 75), np.percentile(col, 75),
            np.percentile(row, 25), np.percentile(col, 25),
            row.sum() / (col.sum() + 1e-6),
            np.log1p(row.sum()),
        ]
        features.append(feat[:in_channels])
    t = torch.tensor(features, dtype=torch.float32)
    return (t - t.mean(dim=0)) / (t.std(dim=0) + 1e-8)


def diff_to_target(diff_matrix: pd.DataFrame,
                   zone_ids: List[int],
                   device: str = 'cpu') -> torch.Tensor:
    """ΔOD matrices → zone level changes(1 x N)."""
    net = diff_matrix.sum(axis=1) + diff_matrix.sum(axis=0)
    net = net.reindex(zone_ids).fillna(0).values.astype(float)
    return torch.tensor(net, dtype=torch.float32).unsqueeze(0).to(device)


def build_scenario_features(scenario_type: str, affected_zones: List[int],
                              num_new_stops: int = 0) -> torch.Tensor:
    """infrastructure changes (1 x 8)."""
    enc  = {'metro_extension': [1,0,0], 'bus_new': [0,1,0], 'tram_new': [0,0,1]}
    feat = enc.get(scenario_type, [0,0,0]) + [
        len(affected_zones), num_new_stops,
        np.log1p(len(affected_zones)),
        min(len(affected_zones) / 100, 1.0),
        1.0 if scenario_type == 'metro_extension' else 0.0,
    ]
    return torch.tensor([feat], dtype=torch.float32)


def get_affected_zones(diff_matrix: pd.DataFrame,
                        zone_ids: List[int],
                        threshold_pct: float = 0.8) -> List[int]:
    total = diff_matrix.abs().sum(axis=1) + diff_matrix.abs().sum(axis=0)
    total = total.reindex(zone_ids).fillna(0)
    return total[total > total.quantile(threshold_pct)].index.tolist()
