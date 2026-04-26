"""
evaluate.py
Evaluation script for GAT+LSTM and Hypergraph+LSTM models.

Usage:
  python evaluate.py --model gat
  python evaluate.py --model hypergraph
  python evaluate.py --model all

Colab:
  !python evaluate.py --model all
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config.paths import (
    BASE_DIR, ZONES_SHP, GTFS_ZIP,
    M2_BASE_KK, M2_DEV_KK,
    M1_KK, M1_DIFF_KK,
    GAT_CHECKPOINT, HG_CHECKPOINT,
)
from utils.data import (
    load_od_matrix_with_header,
    od_matrix_to_zone_features, build_gtfs_zone_features,
    diff_to_target, build_scenario_features, get_affected_zones,
    NUM_FEATURES,
)

# ─── Validation scenario ────────────────────────────────────────────────────
# M1 extension is the held-out validation scenario (matches train.py split).
# The model was never trained on this scenario, so evaluation here is a fair
# out-of-sample test. Bus 35 is now part of the training set.
VAL_SCENARIO_NAME = 'M1 extension'
VAL_SCENARIO_TYPE = 'metro_extension'   # used for scenario_feat encoding


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(model_type: str, zone_ids, device, gtfs_features=None):
    in_ch = NUM_FEATURES if gtfs_features else 16

    if model_type == 'gat':
        from models.gat_lstm import GATLSTMModel, Config, build_zone_graph
        cfg                 = Config()
        cfg.GAT_IN_CHANNELS = in_ch
        model               = GATLSTMModel(cfg).to(device)
        checkpoint          = GAT_CHECKPOINT

        try:
            import geopandas as gpd
            gdf = gpd.read_file(ZONES_SHP)
            gdf['NO'] = gdf['NO'].astype(int)
            gdf_proj  = gdf.to_crs(epsg=23700)
            centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
            gdf['lon'] = centroids.x
            gdf['lat'] = centroids.y
        except Exception:
            gdf = None

        edge_index = build_zone_graph(zone_ids, gdf=gdf).to(device)
        extra      = {'edge_index': edge_index}

    elif model_type == 'hypergraph':
        from models.hypergraph_lstm import (
            HypergraphLSTMModel, HypergraphConfig, build_incidence_matrix
        )
        import zipfile
        from scipy.spatial import cKDTree

        cfg                = HypergraphConfig()
        cfg.HG_IN_CHANNELS = in_ch
        model              = HypergraphLSTMModel(cfg).to(device)
        checkpoint         = HG_CHECKPOINT
        extra              = {}

        gtfs_routes = None
        if os.path.exists(GTFS_ZIP):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(ZONES_SHP)
                gdf['NO'] = gdf['NO'].astype(int)
                gdf_proj  = gdf.to_crs(epsg=23700)
                centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
                gdf['lon'] = centroids.x
                gdf['lat'] = centroids.y

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
            except Exception as e:
                print(f'GTFS error: {e}')

        H = build_incidence_matrix(zone_ids, gtfs_routes=gtfs_routes)
        model.set_hypergraph(H)

    if not os.path.exists(checkpoint):
        print(f'Checkpoint not found: {checkpoint}')
        print(f'  Run: python train.py --model {model_type}')
        return None, None, None

    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Best epoch     : {ckpt.get("best_epoch", "?")}')
        print(f'  Best val loss  : {ckpt.get("best_val_loss", 0):.4f} (normalised MSE)')
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f'Loaded {model_type}: {checkpoint}')
    return model, extra, ckpt if isinstance(ckpt, dict) else {}


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pred_np: np.ndarray, target_np: np.ndarray) -> dict:
    from utils.metrics import evaluate_all
    base = evaluate_all(pred_np.reshape(-1, int(len(pred_np)**0.5))
                        if pred_np.ndim == 1 else pred_np,
                        target_np.reshape(-1, int(len(target_np)**0.5))
                        if target_np.ndim == 1 else target_np,
                        k=20)
    morans_i = _morans_i(pred_np - target_np)
    return {
        'MAE':      float(np.mean(np.abs(pred_np - target_np))),
        'RMSE':     base['rmse'],
        'R2':       base['r2'],
        'Spearman': base['spearman'],
        'TopKAcc':  base['top_k_acc'],
        'NonzeroRMSE': base['nonzero_rmse'],
        'MoransI':  morans_i,
    }

def _morans_i(residuals: np.ndarray, n_neighbors: int = 5) -> float:
    n = len(residuals)
    if n < 4:
        return 0.0
    W = np.zeros((n, n))
    for i in range(n):
        dists    = np.abs(np.arange(n) - i)
        dists[i] = n
        neighbors = np.argsort(dists)[:n_neighbors]
        W[i, neighbors] = 1.0
    W /= W.sum(axis=1, keepdims=True) + 1e-9
    z   = residuals - residuals.mean()
    num = float(n * z @ W @ z)
    den = float(W.sum() * (z ** 2).sum() + 1e-9)
    return num / den


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model_type, zone_ids, device, gtfs_features=None):
    """Run inference on the M1 extension validation scenario."""
    model, extra, ckpt_data = load_model(
        model_type, zone_ids, device, gtfs_features
    )
    if model is None:
        return None

    in_ch = NUM_FEATURES if gtfs_features else 16

    # Load M1 validation scenario data
    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    m1_kk    = load_od_matrix_with_header(M1_KK).reindex(
        index=zone_ids, columns=zone_ids).fillna(0)
    m1_diff  = load_od_matrix_with_header(M1_DIFF_KK).reindex(
        index=zone_ids, columns=zone_ids).fillna(0)

    x_seq = [
        od_matrix_to_zone_features(m2_base, in_ch, gtfs_features).to(device),
        od_matrix_to_zone_features(m1_kk,   in_ch, gtfs_features).to(device),
    ]
    scenario_feat = build_scenario_features(
        VAL_SCENARIO_TYPE, get_affected_zones(m1_diff, zone_ids)
    ).to(device)
    target = diff_to_target(m1_diff, zone_ids, device)

    # ── Parameter count ─────────────────────────────────────────────────────
    param_count = sum(p.numel() for p in model.parameters())
    print(f'  Parameter count: {param_count:,}')

    # ── GPU inference ────────────────────────────────────────────────────────
    with torch.no_grad():
        if model_type == 'gat':
            pred = model(x_seq, extra['edge_index'], scenario_feat)
        else:
            pred = model(x_seq, scenario_feat)

    pred_np   = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    # ── CPU inference time ───────────────────────────────────────────────────
    # Measure on CPU — relevant for the Streamlit inference app.
    # 10-run average after 1 warmup pass to avoid cold-start bias.
    import time
    model_cpu         = model.cpu()
    x_seq_cpu         = [x.cpu() for x in x_seq]
    scenario_feat_cpu = scenario_feat.cpu()

    # Move hypergraph internal tensors to CPU too
    if model_type == 'hypergraph' and hasattr(model_cpu, 'H'):
        model_cpu.H            = model_cpu.H.cpu()
        model_cpu.D_v_inv_sqrt = model_cpu.D_v_inv_sqrt.cpu()
        model_cpu.D_e_inv      = model_cpu.D_e_inv.cpu()

    if model_type == 'gat':
        edge_index_cpu = extra['edge_index'].cpu()
        _call = lambda: model_cpu(x_seq_cpu, edge_index_cpu, scenario_feat_cpu)
    else:
        _call = lambda: model_cpu(x_seq_cpu, scenario_feat_cpu)

    with torch.no_grad():
        _call()  # warmup
        N     = 10
        start = time.perf_counter()
        for _ in range(N):
            _call()
        inference_ms = (time.perf_counter() - start) / N * 1000

    model.to(device)  # move back to GPU for consistency
    print(f'  CPU inference time: {inference_ms:.1f} ms (avg over {N} runs)')

    # ── Denormalise ──────────────────────────────────────────────────────────
    # The model was trained on normalised targets (target / target_std).
    # We recover the original passenger-count scale using the stored target_std
    # for the M1 validation scenario from the checkpoint.
    target_stds = ckpt_data.get('target_stds', {})
    val_std = target_stds.get(VAL_SCENARIO_NAME, None)

    if val_std is not None:
        pred_np = pred_np * val_std
        # target_np comes from diff_to_target() — raw unscaled values (utas/zóna).
        # Only pred needs denormalising; target is already on the correct scale.
        print(f'  Denormalised pred with target_std={val_std:.4f} (scale: utas/zóna ΔOD)')
    else:
        print('  ⚠️  target_std not found in checkpoint — metrics in normalised space')
        print('       Re-train with current train.py to save target_stds.')

    n = len(zone_ids)
    metrics = compute_metrics(
        pred_np.reshape(n, n) if pred_np.ndim == 1 else pred_np,
        target_np.reshape(n, n) if target_np.ndim == 1 else target_np,
    )

    print(f'\n{model_type.upper()} — {VAL_SCENARIO_NAME} (validation):')
    print(f'  MAE:          {metrics["MAE"]:.4f}  utas/zóna')
    print(f'  RMSE:         {metrics["RMSE"]:.4f}  utas/zóna')
    print(f'  Nonzero RMSE: {metrics["NonzeroRMSE"]:.4f}  utas/zóna')
    print(f'  R²:           {metrics["R2"]:.4f}')
    print(f'  Spearman ρ:   {metrics["Spearman"]:.4f}')
    print(f'  Top-20 acc:   {metrics["TopKAcc"]:.2f}')
    print(f'  Moran\'s I:    {metrics["MoransI"]:.4f}')

    return {
        'model':        model_type,
        'pred':         pred_np,
        'target':       target_np,
        'val_std':      val_std,
        'param_count':  param_count,
        'inference_ms': inference_ms,
        'ckpt':         ckpt_data,
        **metrics,
    }


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_results(results: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig_dir = os.path.join(save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    colors = ['steelblue', 'coral']

    # 1. Metric bar chart
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f'GAT+LSTM vs Hypergraph+LSTM — {VAL_SCENARIO_NAME} validation',
        fontsize=12, fontweight='bold'
    )
    for i, metric in enumerate(['MAE', 'RMSE', 'R2', 'Spearman', 'TopKAcc', 'MoransI']):
        labels = [r['model'] for r in results]
        vals   = [r[metric] for r in results]
        axes[i].bar(labels, vals, color=colors[:len(labels)], edgecolor='white')
        axes[i].set_title(metric)
        axes[i].grid(alpha=0.3, axis='y')
        for j, v in enumerate(vals):
            axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {path}')

    # 2. Scatter plots
    fig, axes = plt.subplots(1, 6, figsize=(22, 5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ax.scatter(r['target'], r['pred'], alpha=0.25, s=6, color='steelblue')
        lim = max(np.abs(r['target']).max(), np.abs(r['pred']).max())
        ax.plot([-lim, lim], [-lim, lim], 'r--', lw=1)
        ax.set_title(f'{r["model"].upper()} (R²={r["R2"]:.3f})')
        ax.set_xlabel('Ground truth ΔOD (utas/zóna)')
        ax.set_ylabel('Predicted ΔOD (utas/zóna)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(fig_dir, 'scatter_plots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {path}')

    # 3. Loss curves
    has_history = any(r['ckpt'].get('train_losses') for r in results)
    if has_history:
        fig, ax = plt.subplots(figsize=(10, 5))
        for r, color in zip(results, colors):
            ckpt = r['ckpt']
            if ckpt.get('train_losses'):
                epochs = range(1, len(ckpt['train_losses']) + 1)
                ax.plot(epochs, ckpt['train_losses'],
                        color=color, linestyle='-',
                        label=f'{r["model"]} train')
                ax.plot(epochs, ckpt['val_losses'],
                        color=color, linestyle='--',
                        label=f'{r["model"]} val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalised MSE Loss')
        ax.set_title('Training and validation loss curves')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(fig_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'Saved: {path}')

    # 4. Results CSV
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['pred', 'target', 'ckpt']}
        for r in results
    ])
    # Reorder columns for readability in the thesis results table
    col_order = ['model', 'MAE', 'RMSE', 'NonzeroRMSE', 'R2',
                 'Spearman', 'TopKAcc', 'MoransI',
                 'param_count', 'inference_ms', 'val_std']
    df = df[[c for c in col_order if c in df.columns]]
    csv_path = os.path.join(save_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}')
    print(df.to_string(index=False))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        choices=['gat', 'hypergraph', 'all'])
    parser.add_argument('--no_gtfs', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    zone_ids = m2_base.index.tolist()
    print(f'Zones: {len(zone_ids)}')

    gtfs_features = None
    if not args.no_gtfs and os.path.exists(GTFS_ZIP):
        try:
            gtfs_features = build_gtfs_zone_features(
                GTFS_ZIP, zone_ids,
                zones_shp_path=ZONES_SHP if os.path.exists(ZONES_SHP) else None,
            )
        except Exception as e:
            print(f'GTFS features failed ({e}) — using 16-dim')

    models_to_eval = ['gat', 'hypergraph'] if args.model == 'all' else [args.model]

    results = []
    for m in models_to_eval:
        r = evaluate_model(m, zone_ids, device, gtfs_features)
        if r is not None:
            results.append(r)

    if results:
        plot_results(results, save_dir=BASE_DIR)