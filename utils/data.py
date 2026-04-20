"""
utils/data.py
Data loading and feature engineering for BKK OD matrix prediction.
"""

import zipfile
import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict
from scipy.spatial import cKDTree


NUM_FEATURES_BASE = 16
NUM_FEATURES_GTFS =  6
NUM_FEATURES      = 22


#      OD matrix loaders                                                                                   

def load_od_matrix_with_header(filepath: str) -> pd.DataFrame:
    """Load VISUM OD matrix export (3-column offset, row 0 = zone IDs)."""
    df       = pd.read_excel(filepath, header=None)
    zone_ids = df.iloc[0, 3:].values.astype(int)
    data     = df.iloc[3:, :]
    row_ids  = data.iloc[:, 0].values.astype(int)
    matrix   = data.iloc[:, 3:].values.astype(float)
    print(f'  {filepath.split("/")[-1].split(chr(92))[-1]}: {len(row_ids)}x{len(zone_ids)}')
    return pd.DataFrame(matrix, index=row_ids, columns=zone_ids)


def load_od_matrix_no_header(filepath: str, zone_ids: List[int]) -> pd.DataFrame:
    """Load VISUM OD matrix with no header row."""
    df     = pd.read_excel(filepath, header=None)
    matrix = df.values.astype(float)
    assert matrix.shape == (len(zone_ids), len(zone_ids)), \
        f'Shape mismatch: {matrix.shape} != {len(zone_ids)}x{len(zone_ids)}'
    return pd.DataFrame(matrix, index=zone_ids, columns=zone_ids)


def load_od_matrix_sheet(filepath: str, sheet_name: str,
                          zone_ids: Optional[List[int]] = None) -> pd.DataFrame:
    """Load VISUM OD matrix from a specific sheet."""
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


#      GTFS zone features                                                                                 

def build_gtfs_zone_features(
    gtfs_zip_path: str,
    zone_ids: List[int],
    zones_shp_path: Optional[str] = None,
) -> Dict[int, List[float]]:
    """
    Compute 6 GTFS-derived features per zone from a GTFS zip file.

    Features: [n_routes, n_trips, n_stops, has_metro, has_tram, has_rail]

    Stop-to-zone assignment uses shapefile centroids (EOV) when available,
    falling back to a WGS84 k-NN proxy otherwise.

    GTFS route_type codes: 0=tram, 1=metro, 2=rail, 3=bus, 11=trolleybus
    """
    print('Building GTFS zone features...')

    with zipfile.ZipFile(gtfs_zip_path, 'r') as z:
        stops      = pd.read_csv(z.open('stops.txt'))
        stop_times = pd.read_csv(z.open('stop_times.txt'), usecols=['trip_id', 'stop_id'])
        trips      = pd.read_csv(z.open('trips.txt'), usecols=['trip_id', 'route_id'])
        routes     = pd.read_csv(z.open('routes.txt'), usecols=['route_id', 'route_type'])

    stop_to_zone = _map_stops_to_zones(stops, zone_ids, zones_shp_path)
    stops['zone_id'] = stops['stop_id'].map(stop_to_zone)

    trip_route = dict(zip(trips['trip_id'], trips['route_id']))
    route_type = dict(zip(routes['route_id'], routes['route_type']))
    stop_times['route_id']   = stop_times['trip_id'].map(trip_route)
    stop_times['route_type'] = stop_times['route_id'].map(route_type)
    stop_times['zone_id']    = stop_times['stop_id'].map(stop_to_zone)
    stop_times = stop_times.dropna(subset=['zone_id', 'route_id'])
    stop_times['zone_id'] = stop_times['zone_id'].astype(int)

    gtfs_feats: Dict[int, List[float]] = {}
    for zid in zone_ids:
        zone_stops = stops[stops['zone_id'] == zid]
        zone_st    = stop_times[stop_times['zone_id'] == zid]
        types      = set(zone_st['route_type'].dropna().astype(int))
        gtfs_feats[zid] = [
            float(zone_st['route_id'].nunique()),
            float(zone_st['trip_id'].nunique()),
            float(len(zone_stops)),
            float(1 in types),   # has_metro
            float(0 in types),   # has_tram
            float(2 in types),   # has_rail
        ]

    mapped = sum(1 for v in gtfs_feats.values() if v[2] > 0)
    print(f'  Done: {mapped}/{len(zone_ids)} zones have stops')
    return gtfs_feats


def _map_stops_to_zones(
    stops: pd.DataFrame,
    zone_ids: List[int],
    zones_shp_path: Optional[str],
) -> Dict:
    """Assign each stop to its nearest zone centroid."""
    if zones_shp_path:
        try:
            import geopandas as gpd
            from pyproj import Transformer

            gdf = gpd.read_file(zones_shp_path).to_crs('EPSG:23700')
            # Priority: exact 'NO' → known names → partial (exclude 'fid')
            _exact   = [c for c in gdf.columns if c.upper() == 'NO']
            _known   = [c for c in gdf.columns
                        if c.lower() in ('zone_id', 'zoneid', 'zone_no')]
            _partial = [c for c in gdf.columns
                        if any(k in c.lower() for k in ['no', 'zone', 'kod'])
                        and c.lower() != 'fid']
            id_col   = (_exact or _known or _partial or [gdf.columns[0]])[0]
            print(f'  Zone ID column: {id_col}')
            gdf['_cx'] = gdf.geometry.centroid.x
            gdf['_cy'] = gdf.geometry.centroid.y
            gi = gdf.set_index(id_col)

            zone_coords = np.array([
                [gi.loc[z, '_cx'] if z in gi.index else 0.0,
                 gi.loc[z, '_cy'] if z in gi.index else 0.0]
                for z in zone_ids
            ])
            tf = Transformer.from_crs('EPSG:4326', 'EPSG:23700', always_xy=True)
            sx, sy = tf.transform(stops['stop_lon'].values, stops['stop_lat'].values)
            _, idxs = cKDTree(zone_coords).query(np.column_stack([sx, sy]))
            return dict(zip(stops['stop_id'], [zone_ids[i] for i in idxs]))

        except Exception as e:
            print(f'  Shapefile unavailable ({e}), using WGS84 fallback')

    # Fallback: WGS84 k-NN proxy
    n = len(zone_ids)
    zone_coords = np.column_stack([
        np.linspace(18.8, 19.3, n),
        np.linspace(47.3, 47.7, n),
    ])
    _, idxs = cKDTree(zone_coords).query(stops[['stop_lon', 'stop_lat']].values)
    return dict(zip(stops['stop_id'], [zone_ids[i] for i in idxs]))


#      Zone feature vector                                                                               

def od_matrix_to_zone_features(
    od_matrix: pd.DataFrame,
    in_channels: int = NUM_FEATURES,
    gtfs_features: Optional[Dict[int, List[float]]] = None,
) -> torch.Tensor:
    """
    Build a zone feature vector from an OD matrix.

    Returns a (N x in_channels) tensor, z-score normalised.
    If in_channels=22 and gtfs_features is provided, appends
    [n_routes, n_trips, n_stops, has_metro, has_tram, has_rail].
    Zones without GTFS data receive zeros for those 6 features.
    """
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
            float((row > 0).sum()),
            float((col > 0).sum()),
            row.max(), col.max(),
            float(np.percentile(row, 75)),
            float(np.percentile(col, 75)),
            float(np.percentile(row, 25)),
            float(np.percentile(col, 25)),
            row.sum() / (col.sum() + 1e-6),
            float(np.log1p(max(row.sum(), 0))),
        ]

        if in_channels == NUM_FEATURES:
            if gtfs_features and zone_id in gtfs_features:
                feat.extend(gtfs_features[zone_id])
            else:
                feat.extend([0.0] * NUM_FEATURES_GTFS)

        features.append(feat[:in_channels])

    t = torch.tensor(features, dtype=torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return (t - t.mean(dim=0)) / (t.std(dim=0) + 1e-8)


#      Target and scenario features                                                             

def diff_to_target(diff_matrix: pd.DataFrame,
                   zone_ids: List[int],
                   device: str = 'cpu') -> torch.Tensor:
    """Convert a ΔOD matrix to a per-zone net change vector (1 x N)."""
    net = diff_matrix.sum(axis=1) + diff_matrix.sum(axis=0)
    net = net.reindex(zone_ids).fillna(0).values.astype(float)
    return torch.tensor(net, dtype=torch.float32).unsqueeze(0).to(device)


def build_scenario_features(scenario_type: str,
                              affected_zones: List[int],
                              num_new_stops: int = 0) -> torch.Tensor:
    """Encode an infrastructure change as an 8-dim scenario feature vector."""
    enc = {
        'metro_extension':         [1, 0, 0],
        'bus_new':                 [0, 1, 0],
        'tram_extension':          [0, 0, 1],
        'bus_frequency_increase':  [0, 1, 0],
        'parallel_route_addition': [0, 0, 1],
    }
    feat = enc.get(scenario_type, [0, 0, 0]) + [
        len(affected_zones),
        num_new_stops,
        float(np.log1p(len(affected_zones))),
        min(len(affected_zones) / 100.0, 1.0),
        1.0 if scenario_type == 'metro_extension' else 0.0,
    ]
    return torch.tensor([feat], dtype=torch.float32)


def get_affected_zones(diff_matrix: pd.DataFrame,
                        zone_ids: List[int],
                        threshold_pct: float = 0.8) -> List[int]:
    """Return the most affected zones based on absolute ΔOD magnitude."""
    total = diff_matrix.abs().sum(axis=1) + diff_matrix.abs().sum(axis=0)
    total = total.reindex(zone_ids).fillna(0)
    return total[total > total.quantile(threshold_pct)].index.tolist()