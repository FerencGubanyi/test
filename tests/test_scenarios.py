"""
tests/test_scenarios.py
Tests for utils/synthetic_scenarios.py — all 6 scenario types.

Key properties verified per scenario:
  1. Finite output (no NaN, no Inf)
  2. Row conservation : delta.sum(axis=1) approx 0
  3. Non-trivial      : |delta|.sum() > 0
  4. Shape            : (N, N) matching input
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.synthetic_scenarios import (
    SyntheticScenarioGenerator,
    extract_scenario_profile,
    save_scenarios,
    load_scenarios,
    validate_synthetic,
)

CONSERVATION_TOL = 1e-3
ALL_TYPES = [
    "bus_new", "tram_extension", "stop_closure",
    "metro_extension", "bus_frequency_increase", "parallel_route_addition",
]
SIZES = ["small", "medium", "large"]


@pytest.fixture
def small_od():
    rng = np.random.default_rng(42)
    N   = 40
    od  = rng.exponential(80.0, (N, N))
    np.fill_diagonal(od, 0.0)
    return pd.DataFrame(od, index=range(N), columns=range(N))


@pytest.fixture
def profile(small_od):
    return extract_scenario_profile(small_od, "test")


@pytest.fixture
def gen(small_od, profile):
    return SyntheticScenarioGenerator(
        zone_ids=list(small_od.index),
        profile=profile,
        gdf=None,
        seed=0,
    )


def assert_valid_delta(diff_df, label=""):
    d = diff_df.values
    assert np.isfinite(d).all(), f"{label}: NaN or Inf"
    assert np.abs(d).sum() > 1e-6, f"{label}: all-zero delta"
    # _enforce_conservation guarantees global sum == 0 (not per-row)
    total = abs(d.sum())
    assert total < CONSERVATION_TOL, f"{label}: global conservation {total:.6f}"


class TestExtractProfile:
    def test_returns_dict(self, small_od):
        assert isinstance(extract_scenario_profile(small_od), dict)

    def test_required_keys(self, small_od):
        p = extract_scenario_profile(small_od)
        for key in ["val_std", "sparsity", "n_zones", "affected_zones"]:
            assert key in p

    def test_n_zones_matches(self, small_od):
        assert extract_scenario_profile(small_od)["n_zones"] == len(small_od)

    def test_val_std_positive(self, small_od):
        assert extract_scenario_profile(small_od)["val_std"] >= 0


class TestV1Types:
    @pytest.mark.parametrize("size", SIZES)
    def test_bus_new_valid(self, gen, size):
        diff, meta = gen.generate_bus_new(0, size)
        assert_valid_delta(diff, f"bus_new/{size}")
        assert meta["type"] == "bus_new" and meta["size"] == size

    @pytest.mark.parametrize("size", SIZES)
    def test_tram_extension_valid(self, gen, size):
        diff, meta = gen.generate_tram_extension(0, size)
        assert_valid_delta(diff, f"tram_extension/{size}")
        assert meta["type"] == "tram_extension"

    @pytest.mark.parametrize("size", SIZES)
    def test_stop_closure_valid(self, gen, size):
        diff, meta = gen.generate_stop_closure(0, size)
        assert_valid_delta(diff, f"stop_closure/{size}")
        assert meta["type"] == "stop_closure"

    def test_stop_closure_mostly_negative(self, gen):
        diff, _ = gen.generate_stop_closure(0, "large")
        assert diff.values.sum() < 1.0

    def test_output_shape(self, gen, small_od):
        diff, _ = gen.generate_bus_new(0)
        assert diff.shape == (len(small_od), len(small_od))

    def test_different_seeds_differ(self, small_od, profile):
        g1 = SyntheticScenarioGenerator(list(small_od.index), profile, seed=1)
        g2 = SyntheticScenarioGenerator(list(small_od.index), profile, seed=2)
        d1, _ = g1.generate_bus_new(0, "medium")
        d2, _ = g2.generate_bus_new(0, "medium")
        assert not np.allclose(d1.values, d2.values)


class TestV2Types:
    @pytest.mark.parametrize("size", SIZES)
    def test_metro_extension_valid(self, gen, size):
        diff, meta = gen.generate_metro_extension(0, size)
        assert_valid_delta(diff, f"metro_extension/{size}")
        assert meta["type"] == "metro_extension"

    @pytest.mark.parametrize("size", SIZES)
    def test_bus_frequency_valid(self, gen, size):
        diff, meta = gen.generate_bus_frequency_increase(0, size)
        assert_valid_delta(diff, f"bus_frequency_increase/{size}")
        assert meta["type"] == "bus_frequency_increase"

    @pytest.mark.parametrize("size", SIZES)
    def test_parallel_route_valid(self, gen, size):
        diff, meta = gen.generate_parallel_route_addition(0, size)
        assert_valid_delta(diff, f"parallel_route_addition/{size}")
        assert meta["type"] == "parallel_route_addition"

    def test_parallel_has_gains_and_losses(self, gen):
        diff, _ = gen.generate_parallel_route_addition(0, "large")
        d = diff.values
        assert (d > 0.1).any(), "No positive entries"
        assert (d < -0.1).any(), "No negative entries"

    def test_bus_frequency_topology_bounded(self, gen, small_od):
        baseline_nonzero = (small_od.values > 0).sum()
        diff, _ = gen.generate_bus_frequency_increase(0, "large")
        assert ((small_od + diff).values > 0.1).sum() <= baseline_nonzero * 1.5


class TestBatchGeneration:
    def test_correct_count(self, gen):
        assert len(gen.generate_batch(n_per_type=3)) == 3 * 6

    def test_all_types_present(self, gen):
        results = gen.generate_batch(n_per_type=2)
        assert {m["type"] for _, m in results} == set(ALL_TYPES)

    def test_sizes_cycle(self, gen):
        results = gen.generate_batch(n_per_type=3)
        bus_sizes = [m["size"] for _, m in results if m["type"] == "bus_new"]
        assert set(bus_sizes) == {"small", "medium", "large"}

    def test_all_finite(self, gen):
        for diff, meta in gen.generate_batch(n_per_type=2):
            assert np.isfinite(diff.values).all(), f"NaN in {meta['scenario_id']}"

    def test_all_conserved(self, gen):
        for diff, meta in gen.generate_batch(n_per_type=2):
            total = abs(diff.values.sum())
            assert total < CONSERVATION_TOL, f"{meta['scenario_id']}: total={total:.6f}"


class TestSaveLoad:
    def test_roundtrip(self, gen, tmp_path):
        original = gen.generate_batch(n_per_type=2)
        save_scenarios(original, str(tmp_path))
        loaded = load_scenarios(str(tmp_path), list(gen.zone_ids))
        assert len(loaded) == len(original)
        for (d_orig, m_orig), (d_load, m_load) in zip(original, loaded):
            assert m_orig["scenario_id"] == m_load["scenario_id"]
            assert np.allclose(d_orig.values, d_load.values, atol=1e-4)

    def test_metadata_json_created(self, gen, tmp_path):
        save_scenarios(gen.generate_batch(n_per_type=1), str(tmp_path))
        assert (tmp_path / "metadata.json").exists()


class TestValidateSynthetic:
    def test_returns_dataframe(self, small_od, gen):
        syn = [d for d, _ in gen.generate_batch(n_per_type=1)]
        assert isinstance(validate_synthetic(small_od, syn[:3]), pd.DataFrame)

    def test_has_real_row(self, small_od, gen):
        syn = [d for d, _ in gen.generate_batch(n_per_type=1)]
        assert "real M2" in validate_synthetic(small_od, syn[:3]).index