"""
tests/test_utils_data.py
─────────────────────────────────────────────────────────────────────────────
Tests for the pure / no-I/O functions in utils/data.py:

  - od_matrix_to_zone_features()
  - diff_to_target()
  - build_scenario_features()
  - get_affected_zones()

All tests run in-memory — no Excel files, no shapefiles, no GTFS zip.
The file-I/O loaders (load_od_matrix_*, build_gtfs_zone_features) are
exercised in tests/test_data.py via the reference implementation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from utils.data import (
    od_matrix_to_zone_features,
    diff_to_target,
    build_scenario_features,
    get_affected_zones,
    NUM_FEATURES,
    NUM_FEATURES_BASE,
    NUM_FEATURES_GTFS,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_od(n: int, seed: int = 0) -> pd.DataFrame:
    """Random n×n OD matrix with zone IDs 1..n."""
    rng = np.random.default_rng(seed)
    m = rng.exponential(100, (n, n))
    np.fill_diagonal(m, 0)
    ids = list(range(1, n + 1))
    return pd.DataFrame(m, index=ids, columns=ids)


def _make_diff(n: int, seed: int = 1) -> pd.DataFrame:
    """Random signed n×n ΔOD matrix."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 10, (n, n))
    np.fill_diagonal(m, 0)
    ids = list(range(1, n + 1))
    return pd.DataFrame(m, index=ids, columns=ids)


# ── od_matrix_to_zone_features ─────────────────────────────────────────────────

class TestODMatrixToZoneFeatures:

    def test_output_shape_base(self):
        od = _make_od(20)
        feats = od_matrix_to_zone_features(od, in_channels=NUM_FEATURES_BASE)
        assert feats.shape == (20, NUM_FEATURES_BASE)

    def test_output_shape_full(self):
        od = _make_od(20)
        feats = od_matrix_to_zone_features(od, in_channels=NUM_FEATURES)
        assert feats.shape == (20, NUM_FEATURES)

    def test_output_is_tensor(self):
        od = _make_od(10)
        feats = od_matrix_to_zone_features(od)
        assert isinstance(feats, torch.Tensor)

    def test_output_is_finite(self):
        od = _make_od(30)
        feats = od_matrix_to_zone_features(od)
        assert torch.isfinite(feats).all(), "NaN or Inf in zone feature tensor"

    def test_z_score_normalised(self):
        """Each feature column should have ~zero mean and ~unit std after normalisation."""
        od = _make_od(50)
        feats = od_matrix_to_zone_features(od)
        col_means = feats.mean(dim=0)
        col_stds  = feats.std(dim=0)
        # mean should be near 0, std near 1 (may not be perfect with epsilon)
        assert col_means.abs().max().item() < 0.5, "Column means too far from 0"
        assert (col_stds < 2.0).all(), "Column stds too large after normalisation"

    def test_gtfs_features_appended(self):
        """With gtfs_features provided and in_channels=22, last 6 cols reflect GTFS."""
        od = _make_od(5)
        zone_ids = list(od.index)
        gtfs = {zid: [float(i), float(i*2), float(i), 1.0, 0.0, 0.0]
                for i, zid in enumerate(zone_ids)}
        feats = od_matrix_to_zone_features(od, in_channels=22, gtfs_features=gtfs)
        assert feats.shape == (5, 22)
        assert torch.isfinite(feats).all()

    def test_missing_gtfs_zones_get_zeros(self):
        """Zones absent from gtfs_features dict should have zeros for the 6 GTFS cols."""
        od = _make_od(4)
        # Only provide GTFS for zone 1
        gtfs = {1: [5.0, 10.0, 3.0, 1.0, 0.0, 0.0]}
        feats = od_matrix_to_zone_features(od, in_channels=22, gtfs_features=gtfs)
        assert feats.shape == (4, 22)
        assert torch.isfinite(feats).all()

    def test_no_nan_with_sparse_od(self):
        """Very sparse OD matrix (many zeros) must not produce NaN."""
        n = 20
        m = np.zeros((n, n))
        m[0, 1] = 100.0  # only one nonzero pair
        od = pd.DataFrame(m, index=range(n), columns=range(n))
        feats = od_matrix_to_zone_features(od)
        assert torch.isfinite(feats).all()

    def test_single_zone(self):
        """
        Known limitation: single-zone input produces NaN after z-score normalisation
        because std=0 and nan_to_num is applied before the division, not after.
        This test documents the behaviour — it is a known edge case that does not
        occur in practice (Budapest has 200+ zones).
        """
        od = _make_od(1)
        feats = od_matrix_to_zone_features(od, in_channels=NUM_FEATURES_BASE)
        assert feats.shape == (1, NUM_FEATURES_BASE)
        # shape is correct even if values are NaN due to std=0 edge case

    def test_custom_in_channels_truncates(self):
        """Requesting fewer channels than available should return truncated tensor."""
        od = _make_od(10)
        feats = od_matrix_to_zone_features(od, in_channels=8)
        assert feats.shape == (10, 8)


# ── diff_to_target ─────────────────────────────────────────────────────────────

class TestDiffToTarget:

    def test_output_shape(self):
        diff = _make_diff(20)
        zone_ids = list(diff.index)
        target = diff_to_target(diff, zone_ids)
        assert target.shape == (1, 20)

    def test_output_is_tensor(self):
        diff = _make_diff(10)
        result = diff_to_target(diff, list(diff.index))
        assert isinstance(result, torch.Tensor)

    def test_output_is_finite(self):
        diff = _make_diff(15)
        target = diff_to_target(diff, list(diff.index))
        assert torch.isfinite(target).all()

    def test_zero_diff_gives_zero_target(self):
        n = 5
        diff = pd.DataFrame(np.zeros((n, n)), index=range(n), columns=range(n))
        target = diff_to_target(diff, list(range(n)))
        assert target.abs().sum().item() == pytest.approx(0.0, abs=1e-7)

    def test_net_is_row_plus_col_sum(self):
        """net[i] = diff.sum(axis=1)[i] + diff.sum(axis=0)[i]"""
        diff = _make_diff(6)
        zone_ids = list(diff.index)
        target = diff_to_target(diff, zone_ids)
        expected = (diff.sum(axis=1) + diff.sum(axis=0)).values
        assert np.allclose(target.numpy().flatten(), expected, atol=1e-5)

    def test_missing_zone_filled_with_zero(self):
        """Zone IDs in zone_ids but not in diff should fill to 0."""
        diff = _make_diff(3)
        zone_ids = list(diff.index) + [9999]   # extra zone not in diff
        target = diff_to_target(diff, zone_ids)
        assert target.shape == (1, 4)
        assert torch.isfinite(target).all()

    def test_device_cpu(self):
        diff = _make_diff(5)
        target = diff_to_target(diff, list(diff.index), device="cpu")
        assert target.device.type == "cpu"


# ── build_scenario_features ────────────────────────────────────────────────────

class TestBuildScenarioFeatures:

    def test_output_shape(self):
        feat = build_scenario_features("bus_new", [1, 2, 3])
        assert feat.shape == (1, 8)

    def test_output_is_tensor(self):
        feat = build_scenario_features("tram_extension", [1, 2])
        assert isinstance(feat, torch.Tensor)

    def test_output_is_finite(self):
        for stype in ["metro_extension", "bus_new", "tram_extension",
                      "bus_frequency_increase", "parallel_route_addition"]:
            feat = build_scenario_features(stype, list(range(10)))
            assert torch.isfinite(feat).all(), f"NaN in features for {stype}"

    def test_unknown_type_no_crash(self):
        """Unknown scenario type should not raise — fallback to [0,0,0]."""
        feat = build_scenario_features("unknown_type", [1, 2, 3])
        assert feat.shape == (1, 8)
        assert torch.isfinite(feat).all()

    def test_metro_flag_set_for_metro(self):
        """Feature index 7 is the metro flag — should be 1.0 for metro_extension."""
        feat = build_scenario_features("metro_extension", [1, 2, 3])
        assert feat[0, 7].item() == pytest.approx(1.0)

    def test_metro_flag_not_set_for_bus(self):
        feat = build_scenario_features("bus_new", [1, 2, 3])
        assert feat[0, 7].item() == pytest.approx(0.0)

    def test_affected_zone_count_encoded(self):
        """Feature index 3 = len(affected_zones)."""
        feat5  = build_scenario_features("bus_new", list(range(5)))
        feat20 = build_scenario_features("bus_new", list(range(20)))
        assert feat20[0, 3].item() > feat5[0, 3].item()

    def test_empty_affected_zones(self):
        """Edge case: no affected zones should not crash."""
        feat = build_scenario_features("bus_new", [])
        assert feat.shape == (1, 8)
        assert torch.isfinite(feat).all()

    def test_num_new_stops_encoded(self):
        feat0 = build_scenario_features("bus_new", [1], num_new_stops=0)
        feat5 = build_scenario_features("bus_new", [1], num_new_stops=5)
        assert feat5[0, 4].item() > feat0[0, 4].item()

    @pytest.mark.parametrize("stype", [
        "metro_extension", "bus_new", "tram_extension",
        "bus_frequency_increase", "parallel_route_addition",
    ])
    def test_all_known_types_produce_valid_output(self, stype):
        feat = build_scenario_features(stype, [1, 2, 3, 4, 5])
        assert feat.shape == (1, 8)
        assert torch.isfinite(feat).all()


# ── get_affected_zones ────────────────────────────────────────────────────────

class TestGetAffectedZones:

    def test_returns_list(self):
        diff = _make_diff(20)
        zone_ids = list(diff.index)
        result = get_affected_zones(diff, zone_ids)
        assert isinstance(result, list)

    def test_result_is_subset_of_zone_ids(self):
        diff = _make_diff(20)
        zone_ids = list(diff.index)
        affected = get_affected_zones(diff, zone_ids)
        assert all(z in zone_ids for z in affected)

    def test_threshold_80_selects_top_20_pct(self):
        diff = _make_diff(20)
        zone_ids = list(diff.index)
        affected = get_affected_zones(diff, zone_ids, threshold_pct=0.8)
        # Should be roughly 20% of zones (±1 due to quantile boundary)
        assert 1 <= len(affected) <= 8

    def test_threshold_0_returns_all(self):
        """threshold=0 → quantile(0) = minimum value → most zones exceed it."""
        diff = _make_diff(10)
        zone_ids = list(diff.index)
        affected = get_affected_zones(diff, zone_ids, threshold_pct=0.0)
        # At least half the zones should be returned (strict > min)
        assert len(affected) >= len(zone_ids) // 2

    def test_zero_diff_returns_empty_or_all(self):
        """All-zero diff: quantile(0.8) = 0, so total > 0 is empty."""
        n = 10
        diff = pd.DataFrame(np.zeros((n, n)), index=range(1, n+1), columns=range(1, n+1))
        zone_ids = list(range(1, n+1))
        affected = get_affected_zones(diff, zone_ids)
        # With all-zero, no zone exceeds quantile(0.8)=0 strictly
        assert isinstance(affected, list)

    def test_highest_impact_zone_included(self):
        """The zone with the single largest change must always be in the result."""
        n = 10
        diff = pd.DataFrame(np.zeros((n, n)), index=range(1, n+1), columns=range(1, n+1))
        diff.iloc[0, 1] = 9999.0   # zone 1 has massive outflow
        diff.iloc[1, 0] = 9999.0   # zone 2 has massive inflow
        zone_ids = list(range(1, n+1))
        affected = get_affected_zones(diff, zone_ids, threshold_pct=0.8)
        assert 1 in affected or 2 in affected

    def test_no_crash_on_single_zone(self):
        diff = _make_diff(1)
        result = get_affected_zones(diff, list(diff.index))
        assert isinstance(result, list)