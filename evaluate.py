"""
Summarize script — GAT+LSTM vs Hypergraph+LSTM 

Használat:
  python evaluate.py --model gat
  python evaluate.py --model hypergraph
  python evaluate.py --model all       # both of them

Colab:
  !python evaluate.py --model all
"""
import folium
import geopandas as gpd
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config.paths import (
    BASE_DIR, BUS35_DIFF_KK, BUS35_KK, ZONES_SHP, GTFS_ZIP,
    M2_BASE_KK, M2_DEV_KK,
    S144_BASE_KK, S144_DIFF_KK,
    GAT_CHECKPOINT, HG_CHECKPOINT
)
from utils.data import (
    load_od_matrix_with_header, load_od_matrix_no_header,
    od_matrix_to_zone_features, diff_to_target,
    build_scenario_features, get_affected_zones, build_gtfs_from_zip 
)


def load_model(model_type: str, zone_ids, device):
    """Load the treained model."""
    if model_type == 'gat':
        from models.gat_lstm import GATLSTMModel, Config, build_zone_graph
        cfg   = Config()
        model = GATLSTMModel(cfg).to(device)

        try:
            import geopandas as gpd
            gdf = gpd.read_file(ZONES_SHP).to_crs(epsg=4326)
            gdf['NO'] = gdf['NO'].astype(int)
        except Exception:
            gdf = None

        edge_index = build_zone_graph(zone_ids, gdf=gdf).to(device)
        checkpoint = GAT_CHECKPOINT
        extra      = {'edge_index': edge_index}
        in_ch      = cfg.GAT_IN_CHANNELS

    elif model_type == 'hypergraph':
        from models.hypergraph_lstm import HypergraphLSTMModel, HypergraphConfig, build_incidence_matrix
        cfg   = HypergraphConfig()
        model = HypergraphLSTMModel(cfg).to(device)

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

                gi     = gdf.set_index('NO')
                coords = np.column_stack([
                    gi.reindex(zone_ids)['lon'].fillna(19.0),
                    gi.reindex(zone_ids)['lat'].fillna(47.5)
                ])
                tree = cKDTree(coords)
                _, idx = tree.query(stops[['stop_lon', 'stop_lat']].values, k=1)
                stops['zone_id'] = [zone_ids[i] for i in idx]
                st = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
                st['zone_id'] = st['stop_id'].map(dict(zip(stops['stop_id'], stops['zone_id'])))
                gtfs_routes = (
                    st.groupby('route_id')['zone_id']
                    .apply(lambda x: list(set(x.dropna())))
                    .to_dict()
                )
            except Exception as e:
                print(f'GTFS hiba: {e}')

        H = build_incidence_matrix(zone_ids, gtfs_routes=gtfs_routes)
        model.set_hypergraph(H)
        checkpoint = HG_CHECKPOINT
        extra      = {}
        in_ch      = cfg.HG_IN_CHANNELS

    if not os.path.exists(checkpoint):
        print(f'⚠️ cannot find checkpoint: {checkpoint}')
        print(f'  Run before: python train.py --model {model_type}')
        return None, None, None, None

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    print(f'✅ {model_type} modell loaded: {checkpoint}')
    return model, extra, in_ch, cfg


def predict(model, model_type, x_seq, scenario_feat, extra):
    """Prediction run with the picked model"""
    with torch.no_grad():
        if model_type == 'gat':
            return model(x_seq, extra['edge_index'], scenario_feat)
        else:
            return model(x_seq, scenario_feat)


def compute_metrics(pred_np, target_np):
    """MAE, RMSE, R² calculation."""
    mae  = np.mean(np.abs(pred_np - target_np))
    rmse = np.sqrt(np.mean((pred_np - target_np)**2))
    ss_res = np.sum((target_np - pred_np)**2)
    ss_tot = np.sum((target_np - target_np.mean())**2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_prediction_map(pred_np, zone_ids, gdf, save_path, title='ΔOD Prediction'):
    """Prediktált ΔOD choropleth térkép."""
    
    # Zóna szintű predikció DataFrame
    pred_df = pd.DataFrame({
        'NO':  zone_ids,
        'pred': pred_np
    })
    
    gdf_plot = gdf.merge(pred_df, on='NO', how='left')
    
    m = folium.Map(location=[47.497, 19.040], zoom_start=11,
                   tiles='CartoDB positron')
    
    vmax = np.percentile(np.abs(pred_np), 95)
    
    folium.Choropleth(
        geo_data    = gdf_plot.__geo_interface__,
        data        = gdf_plot,
        columns     = ['NO', 'pred'],
        key_on      = 'feature.properties.NO',
        fill_color  = 'RdBu',
        fill_opacity= 0.7,
        line_opacity= 0.1,
        legend_name = 'Predicted ΔOD (passengers/day)',
        nan_fill_color = 'lightgray'
    ).add_to(m)
    
    # Tooltip
    folium.GeoJson(
        gdf_plot,
        style_function = lambda x: {'fillOpacity': 0, 'weight': 0},
        tooltip = folium.GeoJsonTooltip(
            fields  = ['NO', 'pred'],
            aliases = ['Zone ID:', 'Predicted ΔOD:'],
            localize= True
        )
    ).add_to(m)
    
    m.save(save_path)
    print(f'✅ Térkép mentve: {save_path}')
    return m

def evaluate_model(model_type, zone_ids, device, gtfs_zone_stats=None,gdf=None):
    """Model validation on bus35 scenario"""
    model, extra, in_ch, cfg = load_model(model_type, zone_ids, device)
    if model is None:
        return None

    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    bus_kk   = load_od_matrix_with_header(BUS35_KK)
    bus_diff = load_od_matrix_with_header(BUS35_DIFF_KK)
    bus_kk   = bus_kk.reindex(index=zone_ids, columns=zone_ids).fillna(0)
    bus_diff = bus_diff.reindex(index=zone_ids, columns=zone_ids).fillna(0)

    x_seq = [
        od_matrix_to_zone_features(m2_base,   in_ch, gtfs_zone_stats).to(device),
        od_matrix_to_zone_features(bus_kk, in_ch, gtfs_zone_stats).to(device),
    ]
    scenario_feat = build_scenario_features(
        'bus_new', get_affected_zones(bus_diff, zone_ids)
    ).to(device)
    target = diff_to_target(bus_diff, zone_ids, device)

    pred   = predict(model, model_type, x_seq, scenario_feat, extra)
    pred_np   = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()

    metrics = compute_metrics(pred_np, target_np)
    print(f'\n{model_type.upper()} results (S000144 validation):')
    print(f'  MAE:  {metrics["MAE"]:.4f}')
    print(f'  RMSE: {metrics["RMSE"]:.4f}')
    print(f'  R²:   {metrics["R2"]:.4f}')
    pred_np = pred.cpu().numpy().flatten()

    map_path = os.path.join(BASE_DIR, 'gat_prediction_map.html')
    plot_prediction_map(
        pred_np  = pred_np,
        zone_ids = zone_ids,
        gdf      = gdf,
        save_path= map_path,
        title    = 'GAT+LSTM - 35-ös autóbusz predikció'
    )
    return {'model': model_type, 'pred': pred_np, 'target': target_np, **metrics}
   

def plot_comparison(results: list, save_dir: str):
    """Graph generation."""
    os.makedirs(save_dir, exist_ok=True)

    # 1. Metric comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('GAT+LSTM vs Hypergraph+LSTM (S000144 validation)', fontsize=13, fontweight='bold')

    metrics = ['MAE', 'RMSE', 'R2']
    labels  = [r['model'] for r in results]
    colors  = ['steelblue', 'seagreen']

    for i, metric in enumerate(metrics):
        vals = [r[metric] for r in results]
        axes[i].bar(labels, vals, color=colors[:len(labels)], edgecolor='white')
        axes[i].set_title(metric)
        axes[i].grid(alpha=0.3, axis='y')
        for j, v in enumerate(vals):
            axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 2. Scatter plot for every model
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ax.scatter(r['target'], r['pred'], alpha=0.3, s=8, color='steelblue')
        lim = max(np.abs(r['target']).max(), np.abs(r['pred']).max())
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1)
        ax.set_title(f'{r["model"].upper()} (R²={r["R2"]:.3f})')
        ax.set_xlabel('Real ΔOD')
        ax.set_ylabel('Prediction ΔOD')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_plots.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # 3. save CSV 
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['pred', 'target']}
                        for r in results])
    df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
    print(f'\n✅ The graphs are save: {save_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BKK OD prediction results')
    parser.add_argument('--model', type=str, default='all',
                        choices=['gat', 'hypergraph', 'all'],
                        help='Valuation modell (gat / hypergraph / all)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    zone_ids = m2_base.index.tolist()

    # GTFS betöltés
    gtfs_zone_stats, _ = build_gtfs_from_zip(GTFS_ZIP, ZONES_SHP, zone_ids)

    models_to_eval = ['gat', 'hypergraph'] if args.model == 'all' else [args.model]

    gdf = gpd.read_file(ZONES_SHP).to_crs(epsg=4326)
    gdf['NO'] = gdf['NO'].astype(int)
    
    results = []
    for m in models_to_eval:
        r = evaluate_model(m, zone_ids, device, gtfs_zone_stats=gtfs_zone_stats,gdf=gdf)
        if r is not None:
            results.append(r)

    if len(results) > 0:
        plot_comparison(results, save_dir=BASE_DIR)



