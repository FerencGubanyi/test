"""
Training script — GAT+LSTM és Hypergraph+LSTM 

Usage:
  python train.py --model gat
  python train.py --model hypergraph
  python train.py --model gat --epochs 200 --lr 5e-4

Colab:
  !python train.py --model gat
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.paths import (
    BASE_DIR, BUS35_DIFF_KK, M1_DIFF_KK, ZONES_SHP, GTFS_ZIP, SYNTHETIC_DIR,
    M2_BASE_KK, M2_DEV_KK,
    S144_BASE_KK, S144_DIFF_KK,
    M1_KK, BUS35_KK,
    GAT_CHECKPOINT, HG_CHECKPOINT
)
from utils.data import (
    load_od_matrix_with_header, load_od_matrix_no_header,
    load_od_matrix_sheet, od_matrix_to_zone_features,
    diff_to_target, build_scenario_features, get_affected_zones
)
from utils.synthetic_scenarios import load_scenarios


def load_all_scenarios(zone_ids, device, in_channels):
    """Load all of the defined scenarios (real + syntetic)."""
    scenarios = []

    print('\nLoad real scenarios...')

    # M2 extension
    if os.path.exists(M2_BASE_KK) and os.path.exists(M2_DEV_KK):
        m2_base = load_od_matrix_with_header(M2_BASE_KK)
        m2_dev  = load_od_matrix_with_header(M2_DEV_KK)
        m2_diff = m2_dev - m2_base
        scenarios.append({
            'name':          'M2 extension',
            'x_seq':         [
                od_matrix_to_zone_features(m2_base, in_channels).to(device),
                od_matrix_to_zone_features(m2_dev,  in_channels).to(device),
            ],
            'scenario_feat': build_scenario_features(
                'metro_extension', get_affected_zones(m2_diff, zone_ids)
            ).to(device),
            'target':        diff_to_target(m2_diff, zone_ids, device),
        })
        print(f'  ✅ M2 extension')
    else:
        print(f'  ⚠️  M2 data missing: {M2_BASE_KK}')
        m2_base = None

    # S000144
    if os.path.exists(S144_BASE_KK) and os.path.exists(S144_DIFF_KK) and m2_base is not None:
        s144_base = load_od_matrix_no_header(S144_BASE_KK, zone_ids)
        s144_diff = load_od_matrix_no_header(S144_DIFF_KK, zone_ids)
        scenarios.append({
            'name':          'S000144',
            'x_seq':         [
                od_matrix_to_zone_features(m2_base,   in_channels).to(device),
                od_matrix_to_zone_features(s144_base, in_channels).to(device),
            ],
            'scenario_feat': build_scenario_features(
                'metro_extension', get_affected_zones(s144_diff, zone_ids)
            ).to(device),
            'target':        diff_to_target(s144_diff, zone_ids, device),
        })
        print(f'  ✅ S000144')

    # M1 extension
    if os.path.exists(M1_KK) and m2_base is not None:
        try:
            m1_kk   = load_od_matrix_with_header(M1_KK)
            m1_diff = load_od_matrix_with_header(M1_DIFF_KK)
            # reindex if it is needed
            m1_kk   = m1_kk.reindex(index=zone_ids, columns=zone_ids).fillna(0)
            m1_diff = m1_diff.reindex(index=zone_ids, columns=zone_ids).fillna(0)
            scenarios.append({
                'name':          'M1 extension',
                'x_seq':         [
                    od_matrix_to_zone_features(m2_base, in_channels).to(device),
                    od_matrix_to_zone_features(m1_kk,   in_channels).to(device),
                ],
                'scenario_feat': build_scenario_features(
                    'metro_extension', get_affected_zones(m1_diff, zone_ids)
                ).to(device),
                'target':        diff_to_target(m1_diff, zone_ids, device),
            })
            print(f'  ✅ M1 extension')
        except Exception as e:
            print(f'  ⚠️  M1 load error: {e}')

    # 35 bus
    if os.path.exists(BUS35_KK) and m2_base is not None:
        try:
            bus_kk   = load_od_matrix_with_header(BUS35_KK)
            bus_diff = load_od_matrix_with_header(BUS35_DIFF_KK)
            bus_kk   = bus_kk.reindex(index=zone_ids, columns=zone_ids).fillna(0)
            bus_diff = bus_diff.reindex(index=zone_ids, columns=zone_ids).fillna(0)
            scenarios.append({
                'name':          '35 bus',
                'x_seq':         [
                    od_matrix_to_zone_features(m2_base, in_channels).to(device),
                    od_matrix_to_zone_features(bus_kk,  in_channels).to(device),
                ],
                'scenario_feat': build_scenario_features(
                    'bus_new', get_affected_zones(bus_diff, zone_ids)
                ).to(device),
                'target':        diff_to_target(bus_diff, zone_ids, device),
            })
            print(f'  ✅ 35 bus loaded')
        except Exception as e:
            print(f'  ⚠️  35 bus load error: {e}')

    # synthetic scenarios
    if os.path.exists(os.path.join(SYNTHETIC_DIR, 'metadata.json')) and m2_base is not None:
        print('\nLoading synthetic scenarios...')
        syn = load_scenarios(SYNTHETIC_DIR, zone_ids)
        for diff, meta in syn:
            x_dev = od_matrix_to_zone_features(m2_base + diff, in_channels)
            affected = (
                meta.get('affected_zones') or 
                meta.get('corridor_zones') or 
                meta.get('stops') or 
                []
            )
            
            scenarios.append({
                'name':          meta['scenario_id'],
                'x_seq':         [
                    od_matrix_to_zone_features(m2_base, in_channels).to(device),
                    x_dev.to(device),
                ],
                'scenario_feat': build_scenario_features(
                    meta['type'], affected
                ).to(device),
                'target':        diff_to_target(diff, zone_ids, device),
            })
        print(f'  ✅ {len(syn)} synthetic scenarios')

    return scenarios


def run_training(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    #base zone IDs from M2 matrix
    print('\nLoad zone IDs...')
    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    zone_ids = m2_base.index.tolist()
    print(f'Zones: {len(zone_ids)} ({min(zone_ids)}—{max(zone_ids)})')

    # Pick model
    if args.model == 'gat':
        from models.gat_lstm import GATLSTMModel, Config, train
        cfg            = Config()
        cfg.NUM_EPOCHS = args.epochs
        cfg.LEARNING_RATE = args.lr
        model          = GATLSTMModel(cfg).to(device)
        save_path      = GAT_CHECKPOINT
        in_channels    = cfg.GAT_IN_CHANNELS

        # Load grpah
        from models.gat_lstm import build_zone_graph
        try:
            import geopandas as gpd
            gdf = gpd.read_file(ZONES_SHP).to_crs(epsg=4326)
            gdf['NO'] = gdf['NO'].astype(int)
        except Exception:
            gdf = None
        edge_index = build_zone_graph(zone_ids, gdf=gdf).to(device)

    elif args.model == 'hypergraph':
        from models.hypergraph_lstm import HypergraphLSTMModel, HypergraphConfig, train
        from models.hypergraph_lstm import build_incidence_matrix
        cfg            = HypergraphConfig()
        cfg.NUM_EPOCHS = args.epochs
        cfg.LEARNING_RATE = args.lr
        model          = HypergraphLSTMModel(cfg).to(device)
        save_path      = HG_CHECKPOINT
        in_channels    = cfg.HG_IN_CHANNELS

        #Build Hypergraph
        gtfs_routes = None
        if os.path.exists(GTFS_ZIP):
            try:
                import zipfile
                import geopandas as gpd
                from scipy.spatial import cKDTree

                gdf = gpd.read_file(ZONES_SHP).to_crs(epsg=4326)
                gdf['NO'] = gdf['NO'].astype(int)
                gdf['lon'] = gdf.geometry.centroid.x
                gdf['lat'] = gdf.geometry.centroid.y

                with zipfile.ZipFile(GTFS_ZIP) as z:
                    stops      = pd.read_csv(z.open('stops.txt'))
                    stop_times = pd.read_csv(z.open('stop_times.txt'))
                    trips      = pd.read_csv(z.open('trips.txt'))

                gi = gdf.set_index('NO')
                coords = np.column_stack([
                    gi.reindex(zone_ids)['lon'].fillna(19.0),
                    gi.reindex(zone_ids)['lat'].fillna(47.5)
                ])
                tree = cKDTree(coords)
                _, idx = tree.query(stops[['stop_lon', 'stop_lat']].values, k=1)
                stops['zone_id'] = [zone_ids[i] for i in idx]
                stop_to_zone = dict(zip(stops['stop_id'], stops['zone_id']))

                st = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
                st['zone_id'] = st['stop_id'].map(stop_to_zone)
                gtfs_routes = (
                    st.groupby('route_id')['zone_id']
                    .apply(lambda x: list(set(x.dropna())))
                    .to_dict()
                )
                print(f'GTFS: {len(gtfs_routes)} line loaded')
            except Exception as e:
                print(f'GTFS error: {e}')
                gdf = None

        H = build_incidence_matrix(zone_ids, gtfs_routes=gtfs_routes)
        model.set_hypergraph(H)
        edge_index = None

    else:
        raise ValueError(f'Unknown model: {args.model}. Pick one: gat / hypergraph')

    # Load scenarios
    all_scenarios = load_all_scenarios(zone_ids, device, in_channels)

    if len(all_scenarios) < 2:
        print(f'\n⚠️  Just {len(all_scenarios)} scanario is loaded!')
        sys.exit(1)
    # 1.
    valid_scenarios = []
    nan_count = 0
    for s in scenarios:
        has_nan = any(
            x.isnan().any() or x.isinf().any()
            for x in s['x_seq']
        ) or s['target'].isnan().any()
        
        if has_nan:
            nan_count += 1
        else:
            valid_scenarios.append(s)

    scenarios = valid_scenarios
    print(f'  ✅ {len(scenarios)} valid scenarios ({nan_count} NaN filtered)')
    # Train / val split
    train_scenarios = [s for s in all_scenarios if s['name'] != '35 bus']
    val_scenario    = next((s for s in all_scenarios if s['name'] == '35 bus'), None)
    if val_scenario is None:
        print('⚠️  can not find bus route 35 scenario')
        train_scenarios = all_scenarios[:-1]
        val_scenario    = all_scenarios[-1]

    print(f'\nTraining scenarios ({len(train_scenarios)}):')
    for s in train_scenarios:
        print(f'  - {s["name"]}')
    print(f'Validation: {val_scenario["name"]}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    if args.model == 'gat':
        for s in train_scenarios + [val_scenario]:
            s['edge_index'] = edge_index
    print('NaN check:')
    for s in train_scenarios:
        has_nan = False
        for i, x in enumerate(s['x_seq']):
            if x.isnan().any() or x.isinf().any():
                print(f'  ❌ {s["name"]} x_seq[{i}]: nan={x.isnan().any()}, inf={x.isinf().any()}')
                has_nan = True
        if s['target'].isnan().any() or s['target'].isinf().any():
            print(f'  ❌ {s["name"]} target: nan={s["target"].isnan().any()}')
            has_nan = True
        if not has_nan:
            print(f'  ✅ {s["name"]}')

    for epoch in range(args.epochs):
        # for all of the scenarios
        model.train()
        epoch_losses = []
        for s in train_scenarios:
            optimizer.zero_grad()
            if args.model == 'gat':
                pred = model(s['x_seq'], s['edge_index'], s['scenario_feat'])
            else:
                pred = model(s['x_seq'], s['scenario_feat'])
            loss = criterion(pred, s['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)

        # validation
        model.eval()
        with torch.no_grad():
            if args.model == 'gat':
                val_pred = model(val_scenario['x_seq'], val_scenario['edge_index'], val_scenario['scenario_feat'])
            else:
                val_pred = model(val_scenario['x_seq'], val_scenario['scenario_feat'])
            val_loss = criterion(val_pred, val_scenario['target']).item()
            val_mae  = F.l1_loss(val_pred, val_scenario['target']).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f'  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f} ✅')
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f'  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}')

        if patience_counter >= 15:
            print(f'Early stopping — {epoch+1} epoch')
            break

    print(f'\n✅ Done | Best val loss: {best_val_loss:.4f} | Checkpoint: {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BKK OD prediction training')
    parser.add_argument('--model',  type=str, default='gat',
                        choices=['gat', 'hypergraph'],
                        help='Modell type (gat / hypergraph)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epoch')
    parser.add_argument('--lr',     type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    print(f'Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr}')
    run_training(args)
