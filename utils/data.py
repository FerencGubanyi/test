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


NUM_FEATURES = 22

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

def od_matrix_to_zone_features(od_matrix, gtfs_zone_stats=None, in_channels=NUM_FEATURES):
    features = []
    for i, zone_id in enumerate(od_matrix.index):
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
            np.log1p(max(row.sum(), 0)),
        ]
        # GTFS feature-ök hozzáadása, ha van
        if gtfs_zone_stats is not None and zone_id in gtfs_zone_stats.index:
            g = gtfs_zone_stats.loc[zone_id]
            feat += [
                g.get('n_routes', 0),      # hány útvonal érinti
                g.get('n_trips', 0),       # napi járatszám
                g.get('n_stops', 0),       # megállók száma
                g.get('has_metro', 0),     # van-e metró
                g.get('has_tram', 0),      # van-e villamos
                g.get('has_rail', 0),      # van-e HÉV
            ]
        features.append(feat)

    t = torch.tensor(features, dtype=torch.float32)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return (t - t.mean(dim=0)) / (t.std(dim=0) + 1e-8)

def build_gtfs_zone_stats(stops_df: pd.DataFrame,
                           stop_times_df: pd.DataFrame,
                           trips_df: pd.DataFrame,
                           routes_df: pd.DataFrame,
                           stop_zone_map: dict) -> pd.DataFrame:
    """GTFS adatokból zónánkénti hálózati statisztikák."""
    stops_df = stops_df.copy()
    stops_df['zone_id'] = stops_df['stop_id'].map(stop_zone_map)

    merged = (stop_times_df
              .merge(trips_df[['trip_id', 'route_id']], on='trip_id')
              .merge(routes_df[['route_id', 'route_type']], on='route_id'))
    merged['zone_id'] = merged['stop_id'].map(stop_zone_map)
    merged = merged.dropna(subset=['zone_id'])
    merged['zone_id'] = merged['zone_id'].astype(int)

    stats = merged.groupby('zone_id').agg(
        n_routes  = ('route_id',   'nunique'),
        n_trips   = ('trip_id',    'nunique'),
        n_stops   = ('stop_id',    'nunique'),
        has_metro = ('route_type', lambda x: int((x == 1).any())),
        has_tram  = ('route_type', lambda x: int((x == 0).any())),
        has_rail  = ('route_type', lambda x: int((x == 2).any())),
    )
    return stats

def build_gtfs_from_zip(gtfs_zip, zones_shp, zone_ids):
    """Visszaad: (gtfs_zone_stats, gtfs_routes) — mindkét modellnek kell."""
    try:
        import zipfile
        import geopandas as gpd
        from scipy.spatial import cKDTree

        gdf = gpd.read_file(zones_shp).to_crs(epsg=4326)
        gdf['NO'] = gdf['NO'].astype(int)
        gdf['lon'] = gdf.geometry.centroid.x
        gdf['lat'] = gdf.geometry.centroid.y

        with zipfile.ZipFile(gtfs_zip) as z:
            stops      = pd.read_csv(z.open('stops.txt'))
            stop_times = pd.read_csv(z.open('stop_times.txt'))
            trips      = pd.read_csv(z.open('trips.txt'))
            routes     = pd.read_csv(z.open('routes.txt'))

        gi = gdf.set_index('NO')
        coords = np.column_stack([
            gi.reindex(zone_ids)['lon'].fillna(19.0),
            gi.reindex(zone_ids)['lat'].fillna(47.5)
        ])
        tree = cKDTree(coords)
        _, idx = tree.query(stops[['stop_lon', 'stop_lat']].values, k=1)
        stops['zone_id'] = [zone_ids[i] for i in idx]
        stop_to_zone = dict(zip(stops['stop_id'], stops['zone_id']))

        gtfs_zone_stats = build_gtfs_zone_stats(stops, stop_times, trips, routes, stop_to_zone)
        print(f'GTFS zone stats: {len(gtfs_zone_stats)} zone')

        st = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
        st['zone_id'] = st['stop_id'].map(stop_to_zone)
        gtfs_routes = (
            st.groupby('route_id')['zone_id']
            .apply(lambda x: list(set(x.dropna())))
            .to_dict()
        )
        print(f'GTFS routes: {len(gtfs_routes)} line')

        return gtfs_zone_stats, gtfs_routes

    except Exception as e:
        print(f'GTFS processing error: {e}')
        return None, None