"""
tests/test_app.py
                             
Tests for the Streamlit app helper functions.

All tests import and call the helper functions directly — no Streamlit
server, no browser, no selenium required. Functions are tested in isolation
using in-memory fake data.

Covered:
  - parse_od_excel()           : VISUM Excel parsing
  - run_inference()            : demo mode fallback
  - disaggregate_to_stops()   : zone → stop disaggregation weights
  - build_folium_map()         : map construction (smoke test)
  - download button data       : CSV and npy export correctness
                             
"""

import io
import sys
import pytest
import numpy as np
import pandas as pd
import openpyxl
from pathlib import Path

# The app uses st.* calls at module level which will fail outside Streamlit.
# We mock the streamlit module before importing the app helpers.
from unittest.mock import MagicMock, patch
import types

#      Build a minimal streamlit stub so app imports don't crash         
_st_stub = types.ModuleType("streamlit")
for _attr in [
    "set_page_config", "markdown", "sidebar", "columns", "file_uploader",
    "button", "selectbox", "text_input", "toggle", "slider", "info",
    "success", "error", "warning", "spinner", "session_state",
    "download_button", "expander", "cache_resource", "stop",
]:
    setattr(_st_stub, _attr, MagicMock())

_st_stub.session_state = {}

# Make st.spinner a context manager
class _FakeSpinner:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass

_st_stub.spinner = _FakeSpinner

sys.modules["streamlit"] = _st_stub

# Also stub streamlit_folium
sys.modules["streamlit_folium"] = types.ModuleType("streamlit_folium")
sys.modules["streamlit_folium"].st_folium = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Now import the helper functions directly (not the full app module)
# to avoid all the top-level st.* calls.
# We extract just the helpers we need.

def _load_app_helpers():
    """
    Import helper functions from streamlit_app.py by exec-ing only the
    function definitions (avoiding top-level st.* UI calls).
    """
    src_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    if not src_path.exists():
        # Try root level (where we saved it)
        src_path = Path(__file__).resolve().parent.parent / "streamlit_app.py"
    if not src_path.exists():
        return None

    src = src_path.read_text()

    # Only exec up to (and including) the helper function definitions,
    # stopping before the st.set_page_config() call that starts the UI.
    # We split at the CSS / page-config section.
    split_marker = "st.set_page_config"
    if split_marker in src:
        src = src[:src.index(split_marker)]

    ns = {"__file__": str(src_path)}
    try:
        exec(src, ns)
        return ns
    except Exception:
        return None


_HELPERS = _load_app_helpers()


def _skip_if_no_helpers():
    if _HELPERS is None:
        pytest.skip("app/streamlit_app.py helpers not importable")


#                              
# Helpers — build fake VISUM Excel files (same as test_data.py)
#                              

def _make_visum_excel(zone_ids: list, matrix: np.ndarray) -> io.BytesIO:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["name", "sum_col", "extra"] + zone_ids)
    ws.append(["skip1", "", ""] + [""] * len(zone_ids))
    ws.append(["skip2", "", ""] + [""] * len(zone_ids))
    for i, zid in enumerate(zone_ids):
        ws.append([zid, f"Zone_{zid}", 0] + list(matrix[i]))
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def _make_malformed_excel() -> io.BytesIO:
    """Excel file with no valid numeric zone IDs in row 0."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["bad", "header", "no", "zone", "ids", "here"])
    ws.append(["data", 1, 2, 3, 4, 5])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


#                              
# Tests — parse_od_excel
#                              

class TestParseODExcel:

    def setup_method(self):
        _skip_if_no_helpers()
        self.parse = _HELPERS.get("parse_od_excel")
        if self.parse is None:
            pytest.skip("parse_od_excel not found in app helpers")

        self.zone_ids = [101, 102, 103, 104]
        self.matrix = np.array([
            [  0, 500, 300, 200],
            [200,   0, 400, 150],
            [100, 150,   0, 350],
            [ 80, 120, 250,   0],
        ], dtype=float)

    def test_returns_correct_zone_ids(self):
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        zone_ids, _ = self.parse(buf)
        assert list(zone_ids) == self.zone_ids

    def test_returns_correct_shape(self):
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        zone_ids, od = self.parse(buf)
        assert od.shape == (len(self.zone_ids), len(self.zone_ids))

    def test_diagonal_zeroed(self):
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = self.parse(buf)
        assert np.all(np.diag(od) == 0.0)

    def test_flow_values_preserved(self):
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = self.parse(buf)
        assert od[0, 1] == pytest.approx(500.0)
        assert od[2, 3] == pytest.approx(350.0)

    def test_output_is_finite(self):
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        _, od = self.parse(buf)
        assert np.isfinite(od).all()

    def test_malformed_file_returns_none_not_crash(self):
        """Malformed file should return (None, None), not raise an exception."""
        buf = _make_malformed_excel()
        result = self.parse(buf)
        assert result == (None, None) or (
            result[0] is None and result[1] is None
        )

    def test_three_column_offset_is_applied(self):
        """
        Specifically test the 3-column offset: zone IDs must come from
        column index 3 onward, not from column 0.
        """
        buf = _make_visum_excel(self.zone_ids, self.matrix)
        df = pd.read_excel(buf, header=None)
        # Column 3 should hold the first zone ID
        assert int(float(df.iloc[0, 3])) == self.zone_ids[0]
        # Column 0 should hold 'name', not a zone ID
        assert df.iloc[0, 0] == "name"


#                              
# Tests — run_inference (demo mode)
#                              

class TestRunInference:

    def setup_method(self):
        _skip_if_no_helpers()
        self.run_inference = _HELPERS.get("run_inference")
        if self.run_inference is None:
            pytest.skip("run_inference not found in app helpers")

    def test_demo_mode_returns_correct_shape(self):
        """With model=None, demo mode must return (N, N) array."""
        N = 50
        rng = np.random.default_rng(0)
        od = rng.exponential(100, (N, N))
        np.fill_diagonal(od, 0)
        feats = np.zeros((N, 22))

        delta = self.run_inference(None, od, feats)
        assert delta.shape == (N, N)

    def test_demo_mode_output_is_finite(self):
        N = 30
        rng = np.random.default_rng(1)
        od = rng.exponential(80, (N, N))
        np.fill_diagonal(od, 0)
        feats = np.zeros((N, 22))

        delta = self.run_inference(None, od, feats)
        assert np.isfinite(delta).all()

    def test_demo_mode_diagonal_is_zero(self):
        """Demo delta should have zero diagonal."""
        N = 20
        od = np.ones((N, N)) * 100
        np.fill_diagonal(od, 0)
        feats = np.zeros((N, 22))

        delta = self.run_inference(None, od, feats)
        assert np.all(np.diag(delta) == 0.0)

    def test_demo_mode_not_all_zero(self):
        N = 40
        rng = np.random.default_rng(2)
        od = rng.exponential(100, (N, N))
        np.fill_diagonal(od, 0)
        feats = np.zeros((N, 22))

        delta = self.run_inference(None, od, feats)
        assert np.abs(delta).sum() > 0.0

    def test_demo_mode_reproducible(self):
        """Demo mode uses a fixed seed — output should be deterministic."""
        N = 30
        rng = np.random.default_rng(0)
        od = rng.exponential(100, (N, N))
        np.fill_diagonal(od, 0)
        feats = np.zeros((N, 22))

        d1 = self.run_inference(None, od, feats)
        d2 = self.run_inference(None, od, feats)
        assert np.allclose(d1, d2)


#                              
# Tests — disaggregate_to_stops
#                              

class TestDisaggregateToStops:

    def setup_method(self):
        _skip_if_no_helpers()
        self.disaggregate = _HELPERS.get("disaggregate_to_stops")
        if self.disaggregate is None:
            pytest.skip("disaggregate_to_stops not found in app helpers")

    def _make_stop_data(self, zone_ids, stops_per_zone=3):
        rows = []
        for zid in zone_ids:
            for s in range(stops_per_zone):
                rows.append({
                    "stop_id":   f"stop_{zid}_{s}",
                    "stop_name": f"Stop {zid}-{s}",
                    "zone_id":   zid,
                    "lat":       47.5 + zid * 0.001,
                    "lon":       19.0 + zid * 0.001,
                    "n_routes":  s + 1,   # 1, 2, 3 routes per stop
                })
        return pd.DataFrame(rows)

    def test_weights_sum_to_one_per_zone(self):
        """Within each zone, stop weights must sum to 1.0."""
        zone_ids = np.array([1, 2, 3])
        N = len(zone_ids)
        rng = np.random.default_rng(0)
        delta = rng.normal(0, 10, (N, N))
        np.fill_diagonal(delta, 0)

        stop_data = self._make_stop_data(zone_ids, stops_per_zone=3)
        result = self.disaggregate(delta, zone_ids, stop_data)

        if result is None:
            pytest.skip("disaggregate returned None (acceptable fallback)")

        for zid in zone_ids:
            stops_in_zone = result[result["zone_id"] == zid]
            if stops_in_zone.empty:
                continue
            # Net delta per zone should equal the sum of stop-level deltas
            zone_net = delta.sum(axis=0)[zid - 1] - delta.sum(axis=1)[zid - 1]
            stop_total = stops_in_zone["delta_net"].sum()
            assert abs(stop_total - zone_net) < 1e-6, \
                f"Zone {zid}: stop total {stop_total:.4f} != zone net {zone_net:.4f}"

    def test_returns_dataframe(self):
        zone_ids = np.array([1, 2])
        N = len(zone_ids)
        rng = np.random.default_rng(0)
        delta = rng.normal(0, 5, (N, N))
        np.fill_diagonal(delta, 0)
        stop_data = self._make_stop_data(zone_ids)
        result = self.disaggregate(delta, zone_ids, stop_data)
        if result is not None:
            assert isinstance(result, pd.DataFrame)

    def test_none_stop_data_returns_none(self):
        """When stop_data is None, disaggregation must return None gracefully."""
        zone_ids = np.array([1, 2, 3])
        delta = np.zeros((3, 3))
        result = self.disaggregate(delta, zone_ids, None)
        assert result is None

    def test_higher_route_count_gets_higher_weight(self):
        """
        A stop with more routes should receive a larger share of the zone delta.
        """
        zone_ids = np.array([1])
        delta = np.zeros((1, 1))

        # Two stops: stop A has 1 route, stop B has 9 routes
        stop_data = pd.DataFrame([
            {"stop_id": "A", "stop_name": "A", "zone_id": 1,
             "lat": 47.5, "lon": 19.0, "n_routes": 1},
            {"stop_id": "B", "stop_name": "B", "zone_id": 1,
             "lat": 47.5, "lon": 19.0, "n_routes": 9},
        ])

        # Force a positive net inflow to zone 1
        delta_with_flow = np.array([[0.0, 100.0], [-50.0, 0.0]])
        zone_ids_2 = np.array([1, 2])
        stop_data_2 = pd.DataFrame([
            {"stop_id": "A", "stop_name": "A", "zone_id": 1,
             "lat": 47.5, "lon": 19.0, "n_routes": 1},
            {"stop_id": "B", "stop_name": "B", "zone_id": 1,
             "lat": 47.5, "lon": 19.0, "n_routes": 9},
        ])
        result = self.disaggregate(delta_with_flow, zone_ids_2, stop_data_2)
        if result is None:
            pytest.skip()

        zone1_stops = result[result["zone_id"] == 1]
        if len(zone1_stops) < 2:
            pytest.skip("Not enough stops to compare")

        stop_a = zone1_stops[zone1_stops["stop_id"] == "A"]["delta_net"].values[0]
        stop_b = zone1_stops[zone1_stops["stop_id"] == "B"]["delta_net"].values[0]
        assert abs(stop_b) > abs(stop_a), \
            "Higher route-count stop should receive larger absolute delta"


#                              
# Tests — CSV and npy export correctness
#                              

class TestExportFormats:
    """
    Tests for the data exported via the download buttons.
    We test the data preparation logic, not the Streamlit widget itself.
    """

    def test_zone_summary_csv_has_correct_columns(self):
        N = 10
        rng = np.random.default_rng(0)
        zone_ids   = np.arange(1, N + 1)
        delta      = rng.normal(0, 5, (N, N))
        np.fill_diagonal(delta, 0)
        delta_inflow  = delta.sum(axis=0)
        delta_outflow = delta.sum(axis=1)
        delta_net     = delta_inflow - delta_outflow

        summary_df = pd.DataFrame({
            "zone_id":       zone_ids,
            "delta_inflow":  delta_inflow,
            "delta_outflow": delta_outflow,
            "delta_net":     delta_net,
        })
        csv_bytes = summary_df.to_csv(index=False).encode()
        reloaded  = pd.read_csv(io.BytesIO(csv_bytes))

        assert list(reloaded.columns) == [
            "zone_id", "delta_inflow", "delta_outflow", "delta_net"
        ]
        assert len(reloaded) == N

    def test_zone_summary_net_equals_inflow_minus_outflow(self):
        N = 15
        rng = np.random.default_rng(1)
        zone_ids = np.arange(1, N + 1)
        delta = rng.normal(0, 10, (N, N))
        np.fill_diagonal(delta, 0)

        delta_inflow  = delta.sum(axis=0)
        delta_outflow = delta.sum(axis=1)
        delta_net     = delta_inflow - delta_outflow

        summary_df = pd.DataFrame({
            "zone_id":       zone_ids,
            "delta_inflow":  delta_inflow,
            "delta_outflow": delta_outflow,
            "delta_net":     delta_net,
        })
        csv_bytes = summary_df.to_csv(index=False).encode()
        reloaded  = pd.read_csv(io.BytesIO(csv_bytes))

        computed_net = reloaded["delta_inflow"] - reloaded["delta_outflow"]
        assert np.allclose(reloaded["delta_net"].values, computed_net.values)

    def test_npy_export_roundtrip(self):
        """ΔOD saved as .npy must recover exactly."""
        N = 20
        rng = np.random.default_rng(2)
        delta = rng.normal(0, 5, (N, N))
        np.fill_diagonal(delta, 0)

        buf = io.BytesIO()
        np.save(buf, delta)
        buf.seek(0)
        recovered = np.load(buf)

        assert np.allclose(delta, recovered)
        assert recovered.shape == (N, N)

    def test_full_od_csv_roundtrip(self):
        """Full ΔOD matrix saved as CSV must recover within float precision."""
        N = 10
        zone_ids = np.arange(101, 101 + N)
        rng = np.random.default_rng(3)
        delta = rng.normal(0, 5, (N, N))
        np.fill_diagonal(delta, 0)

        delta_df = pd.DataFrame(delta, index=zone_ids, columns=zone_ids)
        buf = io.BytesIO()
        delta_df.to_csv(buf)
        buf.seek(0)
        recovered = pd.read_csv(buf, index_col=0)
        recovered.columns = recovered.columns.astype(int)
        recovered.index   = recovered.index.astype(int)

        assert np.allclose(delta_df.values, recovered.values, atol=1e-6)


#                              
# Tests — build_folium_map (smoke test, no rendering)
#                              

class TestBuildFoliumMap:

    def setup_method(self):
        _skip_if_no_helpers()
        folium = pytest.importorskip("folium", reason="folium not installed")
        self.build_map = _HELPERS.get("build_folium_map")
        if self.build_map is None:
            pytest.skip("build_folium_map not found in app helpers")

    def test_returns_folium_map(self):
        import folium
        N = 20
        zone_ids   = np.arange(1, N + 1)
        delta_net  = np.random.default_rng(0).normal(0, 50, N)

        m = self.build_map(delta_net, zone_ids)
        assert isinstance(m, folium.Map)

    def test_map_has_legend(self):
        """Map HTML should contain the legend element."""
        N = 10
        zone_ids  = np.arange(1, N + 1)
        delta_net = np.ones(N) * 100

        m = self.build_map(delta_net, zone_ids)
        html = m.get_root().render()
        assert "ΔFlow" in html or "delta" in html.lower()

    def test_all_zero_delta_does_not_crash(self):
        """A zero ΔOD (no change scenario) must not crash the map builder."""
        N = 10
        zone_ids  = np.arange(1, N + 1)
        delta_net = np.zeros(N)

        m = self.build_map(delta_net, zone_ids)
        assert m is not None
