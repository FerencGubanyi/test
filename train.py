"""
train.py
Training script for GAT+LSTM and Hypergraph+LSTM models.

Usage:
  python train.py --model gat
  python train.py --model hypergraph
  python train.py --model gat --epochs 300 --lr 5e-4

Colab:
  !python train.py --model gat --epochs 300
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
    BASE_DIR, ZONES_SHP, GTFS_ZIP, SYNTHETIC_DIR,
    M2_BASE_KK, M2_DEV_KK,
    S144_BASE_KK, S144_DIFF_KK,
    M1_KK, M1_DIFF_KK,
    BUS35_KK, BUS35_DIFF_KK,
    GAT_CHECKPOINT, HG_CHECKPOINT,
)
from utils.data import (
    load_od_matrix_with_header, load_od_matrix_no_header,
    od_matrix_to_zone_features, build_gtfs_zone_features,
    diff_to_target, build_scenario_features, get_affected_zones,
    NUM_FEATURES,
)
from utils.synthetic_scenarios import load_scenarios


#      Scenario loading               

def load_all_scenarios(zone_ids, device, in_channels,
                        gtfs_features=None, real_weight=3.0):
    """
    Load all training scenarios (real VISUM + synthetic).

    Real scenarios are weighted real_weight in the loss to prevent the
    model from over-fitting to the larger synthetic set.
    """
    scenarios = []

    print('\nLoading real VISUM scenarios...')

    m2_base = None
    if os.path.exists(M2_BASE_KK) and os.path.exists(M2_DEV_KK):
        m2_base = load_od_matrix_with_header(M2_BASE_KK)
        m2_dev  = load_od_matrix_with_header(M2_DEV_KK)
        m2_diff = m2_dev - m2_base
        scenarios.append({
            'name':          'M2 extension',
            'weight':        real_weight,
            'x_seq': [
                od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                od_matrix_to_zone_features(m2_dev,  in_channels, gtfs_features).to(device),
            ],
            'scenario_feat': build_scenario_features(
                'metro_extension', get_affected_zones(m2_diff, zone_ids)
            ).to(device),
            'target': diff_to_target(m2_diff, zone_ids, device),
        })
        print('  ✅ M2 extension')
    else:
        print(f'  ⚠️  M2 data missing: {M2_BASE_KK}')

    if m2_base is None:
        print('M2 baseline required — aborting.')
        sys.exit(1)

    # S000144 intentionally excluded:
    # - Scenario type/content is unknown (unreliable label)
    # - Diff amplitude (mean|target|≈85) is ~15x larger than M2/Bus35 (≈5–12),
    #   dominating MSE loss and preventing the model from learning typical-scale
    #   changes. Re-enable only if type is identified and amplitude is verified.
    if os.path.exists(S144_BASE_KK) and os.path.exists(S144_DIFF_KK):
        print('  ⏭️  S000144 skipped (excluded: unknown type + amplitude outlier ~85 vs ~5–12)')

    if os.path.exists(M1_KK) and os.path.exists(M1_DIFF_KK):
        try:
            m1_kk   = load_od_matrix_with_header(M1_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            m1_diff = load_od_matrix_with_header(M1_DIFF_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            scenarios.append({
                'name':   'M1 extension',
                'weight': real_weight,
                'x_seq': [
                    od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                    od_matrix_to_zone_features(m1_kk,   in_channels, gtfs_features).to(device),
                ],
                'scenario_feat': build_scenario_features(
                    'metro_extension', get_affected_zones(m1_diff, zone_ids)
                ).to(device),
                'target': diff_to_target(m1_diff, zone_ids, device),
            })
            print('  ✅ M1 extension')
        except Exception as e:
            print(f'  ⚠️  M1 load error: {e}')

    if os.path.exists(BUS35_KK) and os.path.exists(BUS35_DIFF_KK):
        try:
            bus_kk   = load_od_matrix_with_header(BUS35_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            bus_diff = load_od_matrix_with_header(BUS35_DIFF_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            scenarios.append({
                'name':   '35 bus',
                'weight': real_weight,
                'x_seq': [
                    od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                    od_matrix_to_zone_features(bus_kk,  in_channels, gtfs_features).to(device),
                ],
                'scenario_feat': build_scenario_features(
                    'bus_new', get_affected_zones(bus_diff, zone_ids)
                ).to(device),
                'target': diff_to_target(bus_diff, zone_ids, device),
            })
            print('  ✅ Bus 35 Pesterzsébet')
        except Exception as e:
            print(f'  ⚠️  Bus 35 load error: {e}')

    metadata = os.path.join(SYNTHETIC_DIR, 'metadata.json')
    if os.path.exists(metadata):
        print('\nLoading synthetic scenarios...')
        syn = load_scenarios(SYNTHETIC_DIR, zone_ids)
        for diff, meta in syn:
            affected = (meta.get('affected_zones')
                        or meta.get('corridor_zones')
                        or meta.get('stops') or [])
            x_dev = od_matrix_to_zone_features(
                m2_base + diff, in_channels, gtfs_features
            )
            scenarios.append({
                'name':   meta['scenario_id'],
                'weight': 1.0,
                'x_seq': [
                    od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                    x_dev.to(device),
                ],
                'scenario_feat': build_scenario_features(
                    meta['type'], affected
                ).to(device),
                'target': diff_to_target(diff, zone_ids, device),
            })
        print(f'  ✅ {len(syn)} synthetic scenarios')
    else:
        print('\n⚠️  No synthetic scenarios found. '
              'Run utils/synthetic_scenarios.py first.')

    return scenarios


#      Training loop                

def run_training(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    zone_ids = m2_base.index.tolist()
    print(f'Zones: {len(zone_ids)} ({min(zone_ids)}—{max(zone_ids)})')

    # Build GTFS features once (shared across all scenarios)
    gtfs_features = None
    if args.use_gtfs and os.path.exists(GTFS_ZIP):
        try:
            gtfs_features = build_gtfs_zone_features(
                GTFS_ZIP, zone_ids,
                zones_shp_path=ZONES_SHP if os.path.exists(ZONES_SHP) else None,
            )
        except Exception as e:
            print(f'GTFS feature build failed ({e}) — using 16-dim features')

    in_channels = NUM_FEATURES if gtfs_features else 16

    # Model setup
    if args.model == 'gat':
        from models.gat_lstm import GATLSTMModel, Config, build_zone_graph
        cfg               = Config()
        cfg.GAT_IN_CHANNELS = in_channels
        cfg.NUM_EPOCHS    = args.epochs
        cfg.LEARNING_RATE = args.lr
        model             = GATLSTMModel(cfg).to(device)
        save_path         = GAT_CHECKPOINT

        try:
            import geopandas as gpd
            gdf = gpd.read_file(ZONES_SHP).to_crs(epsg=4326)
            gdf['NO'] = gdf['NO'].astype(int)
        except Exception:
            gdf = None
        edge_index = build_zone_graph(zone_ids, gdf=gdf).to(device)

    elif args.model == 'hypergraph':
        from models.hypergraph_lstm import (
            HypergraphLSTMModel, HypergraphConfig, build_incidence_matrix
        )
        cfg                  = HypergraphConfig()
        cfg.HG_IN_CHANNELS   = in_channels
        cfg.NUM_EPOCHS       = args.epochs
        cfg.LEARNING_RATE    = args.lr
        model                = HypergraphLSTMModel(cfg).to(device)
        save_path            = HG_CHECKPOINT
        edge_index           = None

        gtfs_routes = None
        if os.path.exists(GTFS_ZIP):
            try:
                import zipfile
                from scipy.spatial import cKDTree
                import geopandas as gpd

                gdf = gpd.read_file(ZONES_SHP)
                gdf['NO'] = gdf['NO'].astype(int)
                # Compute centroids in projected EOV CRS (EPSG:23700) for accuracy,
                # then convert centroid points back to WGS84 for stop_lon/stop_lat matching
                gdf_proj    = gdf.to_crs(epsg=23700)
                centroids   = gdf_proj.geometry.centroid.to_crs(epsg=4326)
                gdf['lon']  = centroids.x
                gdf['lat']  = centroids.y

                with zipfile.ZipFile(GTFS_ZIP) as z:
                    stops      = pd.read_csv(z.open('stops.txt'))
                    stop_times = pd.read_csv(z.open('stop_times.txt'))
                    trips      = pd.read_csv(z.open('trips.txt'))

                gi     = gdf.set_index('NO')
                coords = np.column_stack([
                    gi.reindex(zone_ids)['lon'].fillna(19.0),
                    gi.reindex(zone_ids)['lat'].fillna(47.5),
                ])
                _, idx = cKDTree(coords).query(
                    stops[['stop_lon', 'stop_lat']].values, k=1
                )
                stops['zone_id'] = [zone_ids[i] for i in idx]
                st = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
                st['zone_id'] = st['stop_id'].map(
                    dict(zip(stops['stop_id'], stops['zone_id']))
                )
                gtfs_routes = (
                    st.groupby('route_id')['zone_id']
                    .apply(lambda x: list(set(x.dropna())))
                    .to_dict()
                )
                print(f'GTFS: {len(gtfs_routes)} routes loaded')
            except Exception as e:
                print(f'GTFS error: {e}')

        H = build_incidence_matrix(zone_ids, gtfs_routes=gtfs_routes)
        model.set_hypergraph(H)

    else:
        raise ValueError(f'Unknown model: {args.model}')

    # Load scenarios
    all_scenarios = load_all_scenarios(
        zone_ids, device, in_channels,
        gtfs_features=gtfs_features,
        real_weight=args.real_weight,
    )

    # Filter NaN
    valid, skipped = [], 0
    for s in all_scenarios:
        if any(x.isnan().any() or x.isinf().any() for x in s['x_seq']) \
                or s['target'].isnan().any():
            skipped += 1
        else:
            valid.append(s)
    all_scenarios = valid
    print(f'\n{len(all_scenarios)} valid scenarios '
          f'({skipped} NaN filtered)')

    if len(all_scenarios) < 2:
        print('Not enough scenarios to train.')
        sys.exit(1)

    # Train / val split
    # M1 extension is held out as validation set.
    # Rationale: Bus 35 moves into training (all 3 real scenarios needed
    # given the small real-data pool). M1 is chosen as val over M2 because
    # its amplitude (mean|diff|~12) is closer to the training set average,
    # making it a more representative generalisation test.
    VAL_SCENARIO_NAME = 'M1 extension'
    train_scenarios = [s for s in all_scenarios if s['name'] != VAL_SCENARIO_NAME]
    val_scenario    = next(
        (s for s in all_scenarios if s['name'] == VAL_SCENARIO_NAME), None
    )
    if val_scenario is None:
        print(f'Warning: Val scenario "{VAL_SCENARIO_NAME}" not found - using last scenario')
        train_scenarios = all_scenarios[:-1]
        val_scenario    = all_scenarios[-1]

    print(f'\nTraining ({len(train_scenarios)} scenarios):')
    for s in train_scenarios:
        print(f'  [{s["weight"]:.1f}x] {s["name"]}')
    print(f'Validation: {val_scenario["name"]}')

    # Diagnostic: print target amplitude for each real scenario so we can
    # detect scale mismatches between train and val targets
    print('\nTarget scale diagnostics (mean |diff| per zone):')
    for s in train_scenarios:
        if s['weight'] > 1.0:  # real scenarios only
            mean_abs = s['target'].abs().mean().item()
            print(f'  {s["name"]:30s}  mean|target|={mean_abs:.4f}')
    mean_abs_val = val_scenario['target'].abs().mean().item()
    std_val      = val_scenario['target'].std().item()
    print(f'  {"[VAL] " + val_scenario["name"]:30s}  mean|target|={mean_abs_val:.4f}  std={std_val:.4f}')

    # Per-scenario target normalisation
    # ------------------------------------
    # Each scenario's target is standardised (zero-mean, unit-std) before
    # training so that no single scenario dominates the MSE loss due to
    # amplitude differences (e.g. a large metro extension vs a small bus
    # addition). The model learns to predict *relative* change patterns;
    # absolute magnitudes are recovered at inference by storing target_std.
    #
    # Val scenario uses its own std (computed on its target vector) so the
    # val loss is also in normalised space — making train/val loss comparable.
    for s in all_scenarios:
        t     = s['target']
        t_std = t.std().clamp(min=1e-6)   # avoid div-by-zero for zero-diff scenarios
        s['target_std']  = t_std
        s['target']      = t / t_std       # normalised target in [-~3, ~3] range

    print('\nNormalised target std per scenario (real only):')
    for s in all_scenarios:
        if s.get('weight', 1.0) > 1.0:
            print(f'  {s["name"]:30s}  target_std={s["target_std"].item():.4f}')
    print(f'  {"[VAL] " + val_scenario["name"]:30s}  target_std={val_scenario["target_std"].item():.4f}')



    if args.model == 'gat':
        for s in train_scenarios + [val_scenario]:
            s['edge_index'] = edge_index

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    # CosineAnnealingLR: smoothly decays LR over the full training run.
    # Avoids the ReduceLROnPlateau trap where patience=5 halves LR every 5
    # epochs of no val improvement, causing LR to collapse to ~1e-5 by epoch 30
    # before the model has had a chance to learn.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for s in train_scenarios:
            optimizer.zero_grad()
            if args.model == 'gat':
                pred = model(s['x_seq'], s['edge_index'], s['scenario_feat'])
            else:
                pred = model(s['x_seq'], s['scenario_feat'])

            loss = criterion(pred, s['target']) * s['weight']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item() / s['weight'])  # log unweighted

        train_loss = float(np.mean(epoch_losses))
        history['train_loss'].append(train_loss)

        model.eval()
        with torch.no_grad():
            if args.model == 'gat':
                val_pred = model(val_scenario['x_seq'],
                                 val_scenario['edge_index'],
                                 val_scenario['scenario_feat'])
            else:
                val_pred = model(val_scenario['x_seq'],
                                 val_scenario['scenario_feat'])
            val_loss = criterion(val_pred, val_scenario['target']).item()
            val_mae  = F.l1_loss(val_pred, val_scenario['target']).item()

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        scheduler.step()  # CosineAnnealingLR: step each epoch, no val_loss arg needed

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_epoch':       epoch + 1,
                'best_val_loss':    best_val_loss,
                'train_losses':     history['train_loss'],
                'val_losses':       history['val_loss'],
                'val_maes':         history['val_mae'],
                # target_std values needed to denormalise predictions at inference
                'target_stds': {
                    s['name']: s['target_std'].item() for s in all_scenarios
                },
            }, save_path)
            print(f'  Epoch {epoch+1:3d} | '
                  f'Train: {train_loss:.4f} | '
                  f'Val: {val_loss:.4f} | '
                  f'MAE: {val_mae:.4f} ✅')
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(f'  Epoch {epoch+1:3d} | '
                      f'Train: {train_loss:.4f} | '
                      f'Val: {val_loss:.4f}')

        if patience_counter >= args.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    print(f'\nDone | Best val loss: {best_val_loss:.4f} | '
          f'Checkpoint: {save_path}')


#      Entry point                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BKK OD prediction training')
    parser.add_argument('--model',       type=str,   default='gat',
                        choices=['gat', 'hypergraph'])
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--patience',    type=int,   default=15)
    parser.add_argument('--real_weight', type=float, default=3.0,
                        help='Loss multiplier for real VISUM scenarios')
    parser.add_argument('--use_gtfs',    action='store_true', default=True,
                        help='Build 22-dim features using GTFS data')
    parser.add_argument('--no_gtfs',     dest='use_gtfs', action='store_false',
                        help='Use 16-dim OD-only features')
    args = parser.parse_args()

    print(f'Model: {args.model} | Epochs: {args.epochs} | '
          f'LR: {args.lr} | Real weight: {args.real_weight}x | '
          f'GTFS features: {args.use_gtfs}')
    run_training(args)