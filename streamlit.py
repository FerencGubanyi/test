"""
app/streamlit_app.py
Streamlit frontend for GAT+LSTM and Hypergraph+LSTM inference.

Run locally:
    streamlit run app/streamlit_app.py
"""    
import io
import os
import sys
import json
import warnings
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import folium
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Add project root to path so model imports work
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# DB layer
try:
    from db.init_db import get_db, save_run, load_runs, load_zone_results, delete_run
    _HAS_DB = True
except ImportError:
    _HAS_DB = False

#  
# PAGE CONFIG
#  

st.set_page_config(
    page_title="BKK Flow Predictor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  
# CUSTOM CSS  — dark, utilitarian, transit-data aesthetic
#  

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/*      root vars      */
:root {
    --bkk-blue:    #0057A8;
    --bkk-yellow:  #FFD100;
    --gain-green:  #00C48C;
    --loss-red:    #FF4757;
    --bg-dark:     #0D1117;
    --bg-card:     #161B22;
    --bg-input:    #1C2333;
    --border:      #30363D;
    --text-primary: #E6EDF3;
    --text-muted:  #8B949E;
    --font-mono:   'Space Mono', monospace;
    --font-body:   'DM Sans', sans-serif;
}

/*      global reset      */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg-dark);
    color: var(--text-primary);
}

/*      hide default streamlit chrome      */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/*      sidebar      */
[data-testid="stSidebar"] {
    background-color: var(--bg-card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: var(--font-body); }

/*      custom header banner      */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 0 8px;
    border-bottom: 2px solid var(--bkk-blue);
    margin-bottom: 28px;
}
.app-header .logo-block {
    background: var(--bkk-blue);
    color: white;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 1.1rem;
    padding: 8px 14px;
    letter-spacing: 0.05em;
}
.app-header .title-block h1 {
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.app-header .title-block p {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin: 2px 0 0;
    font-family: var(--font-mono);
}

/*      metric cards      */
.metric-row { display: flex; gap: 12px; margin-bottom: 20px; }
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
}
.metric-card .label {
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 600;
    font-family: var(--font-mono);
    color: var(--text-primary);
    line-height: 1;
}
.metric-card .value.gain { color: var(--gain-green); }
.metric-card .value.loss { color: var(--loss-red); }
.metric-card .sub {
    font-size: 0.73rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-family: var(--font-mono);
}

/*      section headers      */
.section-title {
    font-size: 0.72rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 24px 0 14px;
}

/*      status badges      */
.badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-blue   { background: rgba(0,87,168,0.25); color: #58A6FF; border: 1px solid rgba(88,166,255,0.3); }
.badge-green  { background: rgba(0,196,140,0.15); color: var(--gain-green); border: 1px solid rgba(0,196,140,0.3); }
.badge-yellow { background: rgba(255,209,0,0.15); color: var(--bkk-yellow); border: 1px solid rgba(255,209,0,0.3); }
.badge-red    { background: rgba(255,71,87,0.15); color: var(--loss-red); border: 1px solid rgba(255,71,87,0.3); }

/*      streamlit widget overrides      */
.stButton > button {
    background: var(--bkk-blue);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 10px 22px;
    width: 100%;
    transition: background 0.15s;
}
.stButton > button:hover { background: #0068C9; }

.stSelectbox > div > div,
.stFileUploader > div {
    background: var(--bg-input) !important;
    border-color: var(--border) !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-input) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
}

/*      top-zones table      */
.zone-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.zone-table th {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
    text-align: left;
}
.zone-table td {
    padding: 7px 10px;
    border-bottom: 1px solid rgba(48,54,61,0.5);
    font-family: var(--font-mono);
    font-size: 0.8rem;
}
.zone-table tr:hover td { background: rgba(255,255,255,0.03); }
.pos { color: var(--gain-green); }
.neg { color: var(--loss-red); }

/*      info box      */
.info-box {
    background: rgba(0,87,168,0.1);
    border: 1px solid rgba(0,87,168,0.4);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.83rem;
    color: var(--text-muted);
    line-height: 1.6;
    margin-bottom: 16px;
}
.info-box strong { color: var(--text-primary); }

/*      spinner override      */
.stSpinner > div { border-top-color: var(--bkk-blue) !important; }

/*      plotly container      */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


#  
# HELPERS
#  

def parse_od_excel(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse a VISUM OD matrix Excel export.
    Returns (zone_ids, od_matrix) or (None, None) on failure.

    VISUM export format (as documented in Section 5.2.2 of the thesis):
      - Row 0: zone IDs start at column index 3
      - Rows 1-2: header/sum rows (skip)
      - Column 0: origin zone ID
      - Columns 1-2: name/sum columns (skip)
      - Data starts at row 3, column 3
    """
    try:
        df = pd.read_excel(file, header=None)
        zone_ids = df.iloc[0, 3:].values.astype(float).astype(int)
        data_block = df.iloc[3:, :]
        row_ids = data_block.iloc[:, 0].values.astype(float).astype(int)
        matrix = data_block.iloc[:, 3:].values.astype(float)

        # Align: keep only zones present in both rows and columns
        common = np.intersect1d(zone_ids, row_ids)
        col_idx = [np.where(zone_ids == z)[0][0] for z in common]
        row_idx = [np.where(row_ids == z)[0][0] for z in common]
        matrix = matrix[np.ix_(row_idx, col_idx)]
        np.fill_diagonal(matrix, 0.0)

        return common, matrix
    except Exception as e:
        st.error(f"OD matrix parsing failed: {e}")
        return None, None


@st.cache_resource(show_spinner=False)
def load_model(model_type: str, checkpoint_path: str):
    if not _HAS_TORCH:
        return None
    try:
        if model_type == "GAT+LSTM":
            from models.gat_lstm import GATLSTMModel, Config
            cfg = Config()
            model = GATLSTMModel(cfg)
        else:
            from models.hypergraph_lstm import HypergraphLSTMModel, HypergraphConfig
            cfg = HypergraphConfig()
            model = HypergraphLSTMModel(cfg)

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def run_inference(model, od_matrix: np.ndarray, zone_features: np.ndarray,
                  zone_ids_list: list, new_stops: list = None) -> np.ndarray:
    """
    Run model inference and return a ΔOD matrix (N x N).

    Zone features are computed from the OD matrix using od_matrix_to_zone_features.
    Scenario features are built from new stops placed on the map.
    Falls back to synthetic demo delta if model is unavailable.
    """
    if model is not None and _HAS_TORCH:
        try:
            import torch
            from utils.data import od_matrix_to_zone_features, build_scenario_features
            from models.hypergraph_lstm import HypergraphLSTMModel

            # --- Build proper zone features from OD matrix ---
            od_df = pd.DataFrame(od_matrix, index=zone_ids_list, columns=zone_ids_list)
            x = od_matrix_to_zone_features(od_df)   # (N, 22)
            x_seq = [x]

            # --- Build scenario feature vector from new stops ---
            if new_stops:
                affected_zones = list({s["zone_id"] for s in new_stops})
                num_new_stops  = len(new_stops)
                scenario_type  = "bus_new"
            else:
                affected_zones = []
                num_new_stops  = 0
                scenario_type  = "bus_new"

            scenario_feat = build_scenario_features(
                scenario_type=scenario_type,
                affected_zones=affected_zones,
                num_new_stops=num_new_stops,
            )   # (1, 8)

            with torch.no_grad():
                if isinstance(model, HypergraphLSTMModel):
                    if model.H is None:
                        from models.hypergraph_lstm import build_incidence_matrix
                        H = build_incidence_matrix(zone_ids_list)
                        model.set_hypergraph(H)
                    delta_vec = model(x_seq, scenario_feat)          # (1, N)
                else:
                    from models.gat_lstm import build_zone_graph
                    edge_index = build_zone_graph(zone_ids_list)
                    delta_vec  = model(x_seq, edge_index, scenario_feat)  # (1, N)

            # Convert zone-level vector (N,) → full NxN delta matrix
            delta_1d = delta_vec.squeeze(0).numpy()
            delta_2d = np.outer(delta_1d, delta_1d) * 0.01
            np.fill_diagonal(delta_2d, 0.0)
            return delta_2d

        except Exception as e:
            st.warning(f"Inference error: {e} — showing demo output.")

    # --- Demo fallback ---
    rng = np.random.default_rng(42)
    N = od_matrix.shape[0]
    delta = np.zeros((N, N))
    affected = rng.choice(N, size=min(80, N // 4), replace=False)
    for i in affected:
        for j in affected:
            if i != j:
                delta[i, j] = rng.normal(0, od_matrix[i, j] * 0.08 + 0.5)
    np.fill_diagonal(delta, 0.0)
    return delta


def disaggregate_to_stops(
    zone_delta: np.ndarray,
    zone_ids: np.ndarray,
    stop_data: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Disaggregate zone-level ΔOD to stop level using route-count weighting.
    Returns a DataFrame with stop_id, zone_id, delta_inflow, delta_outflow.
    """
    if stop_data is None or len(stop_data) == 0:
        return None

    results = []
    zone_net = zone_delta.sum(axis=0) - zone_delta.sum(axis=1)  # net inflow change

    for z_idx, z_id in enumerate(zone_ids):
        stops_in_zone = stop_data[stop_data["zone_id"] == z_id]
        if stops_in_zone.empty:
            continue
        total_routes = stops_in_zone["n_routes"].sum()
        if total_routes == 0:
            w = np.ones(len(stops_in_zone)) / len(stops_in_zone)
        else:
            w = stops_in_zone["n_routes"].values / total_routes

        net = zone_net[z_idx]
        for (_, row), wi in zip(stops_in_zone.iterrows(), w):
            results.append({
                "stop_id":      row["stop_id"],
                "stop_name":    row.get("stop_name", str(row["stop_id"])),
                "zone_id":      z_id,
                "lat":          row["lat"],
                "lon":          row["lon"],
                "delta_net":    net * wi,
            })

    return pd.DataFrame(results) if results else None


def build_folium_map(
    zone_delta_net: np.ndarray,
    zone_ids: np.ndarray,
    zone_gdf=None,
    stop_df: Optional[pd.DataFrame] = None,
    show_stops: bool = False,
) -> folium.Map:
    """
    Build a Folium choropleth map centred on Budapest.
    zone_delta_net: (N,) net flow change per zone (inflow - outflow)
    """
    m = folium.Map(
        location=[47.498, 19.040],
        zoom_start=12,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    vmax = np.abs(zone_delta_net).max()
    if vmax < 1e-6:
        vmax = 1.0

    #      zone choropleth (if GeoJSON available)     
    if zone_gdf is not None and _HAS_GPD:
        try:
            gdf = zone_gdf.copy()
            gdf["zone_id"] = gdf["NO"].astype(int)
            gdf["delta_net"] = 0.0
            for i, zid in enumerate(zone_ids):
                mask = gdf["zone_id"] == int(zid)
                gdf.loc[mask, "delta_net"] = float(zone_delta_net[i])

            def zone_style(feature):
                val = feature["properties"].get("delta_net", 0)
                frac = val / vmax  # [-1, 1]
                frac = np.sign(frac) * (abs(frac) ** 0.3)  # gamma correction
                if frac >= 0:
                    r, g, b = 0, int(196 * frac), int(140 * frac)
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    opacity = 0.3 + 0.5 * frac
                else:
                    r, g, b = int(255 * (-frac)), int(71 * (-frac)), int(87 * (-frac))
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    opacity = 0.3 + 0.5 * (-frac)
                return {
                    "fillColor": color,
                    "fillOpacity": opacity,
                    "color": "#30363D",
                    "weight": 0.5,
                }

            folium.GeoJson(
                gdf.__geo_interface__,
                style_function=zone_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=["zone_id", "delta_net"],
                    aliases=["Zone ID", "ΔFlow (net)"],
                    localize=True,
                ),
                name="Zone ΔFlow",
            ).add_to(m)
        except Exception:
            pass

    #      stop-level markers (optional)     
    if show_stops and stop_df is not None and not stop_df.empty:
        for _, row in stop_df.iterrows():
            val = row["delta_net"]
            color = "#00C48C" if val >= 0 else "#FF4757"
            size = max(3, min(12, abs(val) / (vmax / 10 + 1e-6) * 8))
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=size,
                color=color,
                fill=True,
                fill_opacity=0.7,
                weight=1,
                tooltip=f"{row.get('stop_name', row['stop_id'])}: Δ{val:+.1f}",
            ).add_to(m)

    #      legend     
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; right: 30px; z-index: 9999;
        background: rgba(13,17,23,0.92); border: 1px solid #30363D;
        border-radius: 8px; padding: 14px 18px; font-family: 'Space Mono', monospace;
        font-size: 11px; color: #E6EDF3; min-width: 160px;
    ">
        <div style="font-size:10px; color:#8B949E; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:10px;">ΔFlow (net)</div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <div style="width:14px;height:14px;background:#00C48C;border-radius:2px;"></div>
            <span>Gain (max +{vmax:.0f})</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:14px;height:14px;background:#FF4757;border-radius:2px;"></div>
            <span>Loss (max -{vmax:.0f})</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    return m


def build_bar_chart(top_n_df: pd.DataFrame, title: str):
    """Build a horizontal bar chart of top gaining/losing zones."""
    if not _HAS_PLOTLY or top_n_df.empty:
        return None

    colors = ["#00C48C" if v >= 0 else "#FF4757" for v in top_n_df["delta_net"]]
    fig = go.Figure(go.Bar(
        x=top_n_df["delta_net"],
        y=top_n_df["zone_id"].astype(str),
        orientation="h",
        marker_color=colors,
        hovertemplate="Zone %{y}<br>Δ = %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono", size=12, color="#8B949E")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono", color="#E6EDF3", size=10),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        xaxis=dict(
            gridcolor="#30363D", zerolinecolor="#30363D",
            title="Net flow change (trips/day)",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", title="Zone ID"),
    )
    return fig


#  
# SIDEBAR
#  

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;">
        <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                    color:#8B949E; text-transform:uppercase; letter-spacing:0.1em;
                    border-bottom:1px solid #30363D; padding-bottom:8px; margin-bottom:16px;">
            Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    #      Model selection     
    st.markdown('<div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Architecture",
        ["GAT+LSTM", "Hypergraph+LSTM"],
        help="GAT+LSTM uses pairwise graph attention. Hypergraph+LSTM uses route-based hyperedges.",
        label_visibility="collapsed",
    )

    #      Checkpoint path     
    st.markdown('<div class="section-title">Checkpoint</div>', unsafe_allow_html=True)

    default_ckpt = {
        "GAT+LSTM":         str(ROOT / "checkpoints" / "gat_lstm_best.pt"),
        "Hypergraph+LSTM":  str(ROOT / "checkpoints" / "hg_lstm_best.pt"),
    }[model_choice]

    ckpt_path = st.text_input(
        "Checkpoint path",
        value=default_ckpt,
        label_visibility="collapsed",
    )

    ckpt_exists = Path(ckpt_path).exists()
    if ckpt_exists:
        st.markdown('<span class="badge badge-green">✓ Found</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-yellow">⚠ Not found — demo mode</span>', unsafe_allow_html=True)

    #      Zone GeoJSON     
    st.markdown('<div class="section-title">Zone Geometry (optional)</div>', unsafe_allow_html=True)
    geojson_path = st.text_input(
        "GeoJSON path",
        value=str(ROOT / "data" / "zones.geojson"),
        label_visibility="collapsed",
        help="BKK TAZ zone polygons for choropleth map. Optional.",
    )

    #      Scenario label     
    st.markdown('<div class="section-title">Scenario Label</div>', unsafe_allow_html=True)
    scenario_name = st.text_input(
        "Scenario name",
        value="",
        placeholder="e.g. M2 extension, Bus 35 closure…",
        label_visibility="collapsed",
        help="Optional label saved to the inference history database.",
    )

    #      Display options     
    st.markdown('<div class="section-title">Display</div>', unsafe_allow_html=True)
    show_stops = st.toggle("Stop-level disaggregation", value=False,
                           help="Distribute zone-level predictions to stops by route count.")
    top_n = st.slider("Top N zones to show", min_value=5, max_value=30, value=15)

    st.markdown("---")

    #      About     
    st.markdown("""
    <div style="font-size:0.72rem; color:#8B949E; font-family:'Space Mono',monospace;
                line-height:1.7;">
        <strong style="color:#E6EDF3;">BKK Flow Predictor</strong><br>
        MSc thesis — Óbudai Egyetem<br>
        Pénzügyi Technológiák, 2024–25<br><br>
        Models trained on BKK EFM VISUM<br>
        scenario OD matrix exports.
    </div>
    """, unsafe_allow_html=True)


#  
# MAIN AREA
#  

#      Header     
st.markdown("""
<div class="app-header">
    <div class="logo-block">BKK</div>
    <div class="title-block">
        <h1>Passenger Flow Redistribution Predictor</h1>
        <p>Deep learning inference on transit infrastructure change scenarios</p>
    </div>
</div>
""", unsafe_allow_html=True)

#      Dependency warnings     
missing = []
if not _HAS_TORCH:   missing.append("torch")
if not _HAS_FOLIUM:  missing.append("folium + streamlit-folium")
if not _HAS_PLOTLY:  missing.append("plotly")
if not _HAS_GPD:     missing.append("geopandas")
if missing:
    st.markdown(f"""
    <div class="info-box">
        <strong>Optional packages not installed:</strong> {", ".join(missing)}<br>
        Install with: <code>pip install {" ".join(missing)}</code><br>
        App runs in demo mode without them.
    </div>
    """, unsafe_allow_html=True)

#  
# STEP 1 — Upload OD Matrix
#  

st.markdown('<div class="section-title">Step 1 — Upload OD Matrix (VISUM Excel export)</div>',
            unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload OD matrix",
        type=["xlsx", "xls"],
        label_visibility="collapsed",
        help="VISUM EFM OD matrix Excel export. Zone IDs must be in row 0, data starts at column 3.",
    )
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    elif "uploaded_file" in st.session_state:
        uploaded_file = st.session_state.uploaded_file

with col_info:
    st.markdown("""
    <div class="info-box">
        <strong>Expected format</strong><br>
        VISUM EFM Excel export.<br>
        Zone IDs: row 0, col 3+<br>
        Flow data: row 3+, col 3+<br><br>
        <strong>No file?</strong> Click Run below<br>
        to use synthetic demo data.
    </div>
    """, unsafe_allow_html=True)

#  
# STEP 2 — Run Inference
#  

st.markdown('<div class="section-title">Step 2 — Run Prediction</div>', unsafe_allow_html=True)

run_btn = st.button(f"▶  Run {model_choice} Inference", use_container_width=False)

#      State     
if "delta_od" not in st.session_state:
    st.session_state.delta_od = None
    st.session_state.zone_ids = None
    st.session_state.od_matrix = None
    st.session_state.inference_time_ms = None
    st.session_state.n_zones = None
    st.session_state.last_run_id = None

if "new_stops" not in st.session_state:
    st.session_state.new_stops = []   # list of {"name", "lat", "lon", "zone_id"}
# DB connection (cached for the session)
if _HAS_DB:
    if "_db" not in st.session_state:
        from db.init_db import get_db
        st.session_state._db = get_db()
# ─────────────────────────────────────────────
# STEP 1.5 — Scenario: Add New Stops on Map
# ─────────────────────────────────────────────

st.markdown('<div class="section-title">Step 1.5 — Define Scenario: Add New Stops</div>',
            unsafe_allow_html=True)

if not _HAS_FOLIUM:
    st.info("Install folium + streamlit-folium to use the interactive map.")
else:
    st.markdown(
        "<div class='info-box'>Click the map to place a new stop. "
        "Fill in the details and click <strong>Add Stop</strong>.</div>",
        unsafe_allow_html=True,
    )

    # Interactive placement map
    placement_map = folium.Map(
        location=[47.497, 19.040],   # Budapest centre
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    # Show already-added stops
    for s in st.session_state.new_stops:
        folium.Marker(
            location=[s["lat"], s["lon"]],
            tooltip=s["name"],
            icon=folium.Icon(color="green", icon="star"),
        ).add_to(placement_map)

    map_data = st_folium(
        placement_map,
        width="100%",
        height=380,
        returned_objects=["last_clicked"],
        key="placement_map",
    )

    # If user clicked — show add-stop form
    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        lat = round(clicked["lat"], 5)
        lon = round(clicked["lng"], 5)

        with st.form("add_stop_form", clear_on_submit=True):
            st.markdown(f"**Clicked location:** {lat}, {lon}")
            stop_name = st.text_input("Stop name", placeholder="e.g. Boráros tér M")
            # Zone ID — manual for now, auto-detect when GeoJSON is loaded
            zone_id_input = st.number_input(
                "Zone ID (TAZ)",
                min_value=1, max_value=9999, value=100, step=1,
                help="The TAZ zone this stop belongs to. "
                     "Auto-detected when zone GeoJSON is loaded.",
            )
            n_routes = st.number_input(
                "Number of routes serving this stop",
                min_value=1, max_value=50, value=3, step=1,
            )
            submitted = st.form_submit_button("➕ Add Stop")

        if submitted and stop_name:
            st.session_state.new_stops.append({
                "name":     stop_name,
                "lat":      lat,
                "lon":      lon,
                "zone_id":  int(zone_id_input),
                "n_routes": int(n_routes),
            })
            st.rerun()

    # Show added stops + option to clear
    if st.session_state.new_stops:
        st.markdown("**Stops in this scenario:**")
        stops_df = pd.DataFrame(st.session_state.new_stops)
        st.dataframe(stops_df, use_container_width=True, hide_index=True)
        if st.button("🗑 Clear all stops"):
            st.session_state.new_stops = []
            st.rerun()
    else:
        st.markdown(
            "<div style='color:#8B949E; font-size:0.85rem;'>"
            "No stops added yet — click the map to place one.</div>",
            unsafe_allow_html=True,
        )

if run_btn:
    with st.spinner(f"Loading {model_choice} checkpoint..."):
        if ckpt_exists and _HAS_TORCH:
            model = load_model(model_choice, ckpt_path)
        else:
            model = None  # demo mode

    # Parse OD matrix
    if uploaded_file is not None:
        with st.spinner("Parsing OD matrix..."):
            zone_ids, od_matrix = parse_od_excel(uploaded_file)
    elif "uploaded_file" in st.session_state and st.session_state.uploaded_file is not None:
        with st.spinner("Parsing OD matrix..."):
            zone_ids, od_matrix = parse_od_excel(st.session_state.uploaded_file)
    else:
    # Demo: match the trained model's zone count (1419)
        rng = np.random.default_rng(0)
        N_demo = 1419
        zone_ids = np.arange(1, N_demo + 1)
        od_matrix = rng.exponential(80, (N_demo, N_demo))
        np.fill_diagonal(od_matrix, 0)
        st.info("No file uploaded — using synthetic demo OD matrix (1419 zones).")

    if od_matrix is not None:
        N = od_matrix.shape[0]
        # Minimal zone features (22-dim zeros as placeholder when not running real model)
        zone_features = np.zeros((N, 22))
        zone_features[:, 0] = od_matrix.sum(axis=1)   # total outflow
        zone_features[:, 1] = od_matrix.sum(axis=0)   # total inflow
        # Change the a zone feature based on the new stops
        if st.session_state.new_stops:
            for stop in st.session_state.new_stops:
                z = stop["zone_id"]
                # zone_ids search
                matches = np.where(zone_ids == z)[0]
                if len(matches) == 0:
                    continue
                idx = matches[0]
                # route count feature (index 2 a 22-dim vektorban) increase
                zone_features[idx, 2] += stop["n_routes"]
                # stop count feature (index 3) increase
                zone_features[idx, 3] += 1
        with st.spinner("Running inference..."):
            import time
            t0 = time.perf_counter()
            delta_od = run_inference(
                model=model,
                od_matrix=od_matrix,
                zone_features=zone_features,
                zone_ids_list=list(zone_ids),
                new_stops=st.session_state.new_stops,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

        st.session_state.delta_od          = delta_od
        st.session_state.zone_ids          = zone_ids
        st.session_state.od_matrix         = od_matrix
        st.session_state.inference_time_ms = elapsed_ms
        st.session_state.n_zones           = N

        # ── Persist to SQLite ──
        if _HAS_DB:
            try:
                run_id = save_run(
                    conn=st.session_state._db,
                    model_type=model_choice,
                    delta_od=delta_od,
                    zone_ids=zone_ids,
                    scenario_name=scenario_name.strip() or "(unnamed)",
                    inference_ms=elapsed_ms,
                    checkpoint_path=ckpt_path,
                )
                st.session_state.last_run_id = run_id
                st.success(f"Inference complete in {elapsed_ms:.1f} ms — saved as run #{run_id}")
            except Exception as _db_err:
                st.session_state.last_run_id = None
                st.success(f"Inference complete in {elapsed_ms:.1f} ms")
                st.warning(f"DB save failed: {_db_err}")
        else:
            st.session_state.last_run_id = None
            st.success(f"Inference complete in {elapsed_ms:.1f} ms")

#  
# STEP 3 — Results
#  

if st.session_state.delta_od is not None:
    delta   = st.session_state.delta_od
    zone_ids = st.session_state.zone_ids
    od      = st.session_state.od_matrix
    N       = st.session_state.n_zones

    # Compute zone-level net flow change (inflow gain - outflow loss)
    delta_inflow  = delta.sum(axis=0)   # col sums = net inflow change per zone
    delta_outflow = delta.sum(axis=1)   # row sums = net outflow change per zone
    delta_net     = delta_inflow - delta_outflow

    n_gaining  = int((delta_net > 0.5).sum())
    n_losing   = int((delta_net < -0.5).sum())
    total_gain = float(delta[delta > 0].sum())
    total_loss = float(delta[delta < 0].sum())
    max_gain   = float(delta_net.max())
    max_loss   = float(delta_net.min())

    #      Metric cards     
    st.markdown('<div class="section-title">Step 3 — Results Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Zones Analysed</div>
            <div class="value">{N}</div>
            <div class="sub">{model_choice}</div>
        </div>
        <div class="metric-card">
            <div class="label">Gaining Zones</div>
            <div class="value gain">{n_gaining}</div>
            <div class="sub">net inflow increase</div>
        </div>
        <div class="metric-card">
            <div class="label">Losing Zones</div>
            <div class="value loss">{n_losing}</div>
            <div class="sub">net inflow decrease</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Zone Gain</div>
            <div class="value gain">+{max_gain:.0f}</div>
            <div class="sub">trips/day net inflow</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Zone Loss</div>
            <div class="value loss">{max_loss:.0f}</div>
            <div class="sub">trips/day net inflow</div>
        </div>
        <div class="metric-card">
            <div class="label">Inference Time</div>
            <div class="value">{st.session_state.inference_time_ms:.0f}<span style="font-size:1rem"> ms</span></div>
            <div class="sub">CPU, local machine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    #      Map + Tables     
    map_col, table_col = st.columns([3, 2])

    with map_col:
        st.markdown('<div class="section-title">Zone-Level Flow Change Map</div>',
                    unsafe_allow_html=True)

        # Load zone geometry
        zone_gdf = None
        if _HAS_GPD and Path(geojson_path).exists():
            try:
                zone_gdf = gpd.read_file(geojson_path)
            except Exception:
                pass

        # Stop-level disaggregation (placeholder — plug in real stop data)
        stop_df = None
        if show_stops:
            # In real usage: load from your GTFS-derived stop table
            # stop_df = pd.read_csv(ROOT / "data" / "stops_with_zones.csv")
            st.markdown(
                '<span class="badge badge-yellow">⚠ Stop data not loaded — '
                'place stops_with_zones.csv in data/</span>',
                unsafe_allow_html=True
            )

        if _HAS_FOLIUM:
            fmap = build_folium_map(
                zone_delta_net=delta_net,
                zone_ids=zone_ids,
                zone_gdf=zone_gdf,
                stop_df=stop_df,
                show_stops=show_stops and stop_df is not None,
            )
            st_folium(fmap, width="100%", height=480, returned_objects=[])
        else:
            # Fallback: simple scatter-like summary
            st.markdown("""
            <div class="info-box">
                Install <code>folium</code> and <code>streamlit-folium</code>
                for the interactive map.<br>
                <code>pip install folium streamlit-folium</code>
            </div>
            """, unsafe_allow_html=True)

            # Show distribution instead
            if _HAS_PLOTLY:
                fig = px.histogram(
                    x=delta_net, nbins=40,
                    color_discrete_sequence=["#0057A8"],
                    title="Distribution of per-zone net flow change",
                    labels={"x": "Net flow change (trips/day)", "y": "Zone count"},
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Space Mono", color="#E6EDF3", size=10),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=320,
                    xaxis=dict(gridcolor="#30363D"),
                    yaxis=dict(gridcolor="#30363D"),
                )
                st.plotly_chart(fig, use_container_width=True)

    with table_col:
        #      Top gainers     
        st.markdown('<div class="section-title">Top Gaining Zones</div>', unsafe_allow_html=True)
        sorted_idx = np.argsort(delta_net)[::-1]
        gainers = pd.DataFrame({
            "zone_id":  zone_ids[sorted_idx[:top_n]],
            "delta_net": delta_net[sorted_idx[:top_n]],
        })
        gainers = gainers[gainers["delta_net"] > 0]

        if not gainers.empty and _HAS_PLOTLY:
            fig = build_bar_chart(gainers, "Net inflow gain (trips/day)")
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="gainers")

        # HTML table
        rows = ""
        for _, row in gainers.head(10).iterrows():
            rows += (f'<tr><td>{int(row["zone_id"])}</td>'
                     f'<td class="pos">+{row["delta_net"]:.1f}</td></tr>')
        st.markdown(f"""
        <table class="zone-table">
            <tr><th>Zone ID</th><th>Δ Inflow</th></tr>
            {rows}
        </table>
        """, unsafe_allow_html=True)

        #      Top losers     
        st.markdown('<div class="section-title" style="margin-top:20px;">Top Losing Zones</div>',
                    unsafe_allow_html=True)
        losers = pd.DataFrame({
            "zone_id":  zone_ids[sorted_idx[-top_n:][::-1]],
            "delta_net": delta_net[sorted_idx[-top_n:][::-1]],
        })
        losers = losers[losers["delta_net"] < 0].iloc[::-1]

        if not losers.empty and _HAS_PLOTLY:
            fig2 = build_bar_chart(losers, "Net inflow loss (trips/day)")
            if fig2:
                st.plotly_chart(fig2, use_container_width=True, key="losers")

        rows2 = ""
        for _, row in losers.head(10).iterrows():
            rows2 += (f'<tr><td>{int(row["zone_id"])}</td>'
                      f'<td class="neg">{row["delta_net"]:.1f}</td></tr>')
        st.markdown(f"""
        <table class="zone-table">
            <tr><th>Zone ID</th><th>Δ Inflow</th></tr>
            {rows2}
        </table>
        """, unsafe_allow_html=True)

    #      Raw ΔOD download 
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        # Zone-level net summary CSV
        summary_df = pd.DataFrame({
            "zone_id":      zone_ids,
            "delta_inflow": delta_inflow,
            "delta_outflow": delta_outflow,
            "delta_net":    delta_net,
        })
        st.download_button(
            "⬇  Zone Summary CSV",
            data=summary_df.to_csv(index=False).encode(),
            file_name="zone_delta_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col2:
        # Full ΔOD matrix as CSV
        delta_df = pd.DataFrame(delta, index=zone_ids, columns=zone_ids)
        buf = io.BytesIO()
        delta_df.to_csv(buf)
        st.download_button(
            "⬇  Full ΔOD Matrix CSV",
            data=buf.getvalue(),
            file_name="delta_od_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp_col3:
        # NumPy binary
        np_buf = io.BytesIO()
        np.save(np_buf, delta)
        st.download_button(
            "⬇  ΔOD Matrix .npy",
            data=np_buf.getvalue(),
            file_name="delta_od.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )

else:
    #      Empty state     
    st.markdown("""
    <div style="
        margin-top: 60px;
        text-align: center;
        color: #30363D;
        font-family: 'Space Mono', monospace;
    ">
        <div style="font-size: 3rem; margin-bottom: 16px;">🚇</div>
        <div style="font-size: 1rem; color: #8B949E;">
            Upload an OD matrix and click <strong style="color:#E6EDF3;">Run Inference</strong> to begin.
        </div>
        <div style="font-size: 0.75rem; color: #30363D; margin-top: 8px;">
            No file? Click Run to use synthetic demo data.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# INFERENCE HISTORY
# ──────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown('<div class="section-title">Inference History</div>', unsafe_allow_html=True)

if not _HAS_DB:
    st.info("DB layer not available — install db/init_db.py dependencies.")
else:
    db = st.session_state._db
    runs_df = load_runs(db)

    if runs_df.empty:
        st.markdown(
            "<div style='color:#8B949E; font-size:0.85rem;'>No runs saved yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Display table
        display_cols = [
            "id", "datetime", "scenario_name", "model_type",
            "n_zones", "inference_ms",
            "n_gaining_zones", "n_losing_zones",
            "max_gain", "max_loss",
        ]
        display_cols = [c for c in display_cols if c in runs_df.columns]

        st.dataframe(
            runs_df[display_cols].rename(columns={
                "id":               "Run #",
                "datetime":         "Time (Budapest)",
                "scenario_name":    "Scenario",
                "model_type":       "Model",
                "n_zones":          "Zones",
                "inference_ms":     "ms",
                "n_gaining_zones":  "↑ Zones",
                "n_losing_zones":   "↓ Zones",
                "max_gain":         "Max gain",
                "max_loss":         "Max loss",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Per-run zone detail expander
        run_ids = runs_df["id"].tolist()
        selected_id = st.selectbox(
            "Inspect zone-level results for run",
            options=run_ids,
            format_func=lambda i: f"#{i} — {runs_df.loc[runs_df['id']==i, 'scenario_name'].values[0]}",
        )
        if selected_id:
            with st.expander(f"Zone results — run #{selected_id}", expanded=False):
                z_df = load_zone_results(db, selected_id)
                if z_df.empty:
                    st.write("No zone data.")
                else:
                    st.dataframe(z_df, use_container_width=True, hide_index=True)
                    csv = z_df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇ Download CSV",
                        data=csv,
                        file_name=f"run_{selected_id}_zones.csv",
                        mime="text/csv",
                    )

        # Delete
        with st.expander("🗑 Delete a run", expanded=False):
            del_id = st.selectbox(
                "Run to delete",
                options=run_ids,
                format_func=lambda i: f"#{i} — {runs_df.loc[runs_df['id']==i, 'scenario_name'].values[0]}",
                key="del_select",
            )
            if st.button("Delete run", type="secondary"):
                delete_run(db, del_id)
                st.success(f"Run #{del_id} deleted.")
                st.rerun()
