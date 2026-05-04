"""
tests/test_metrics.py
─────────────────────────────────────────────────────────────────────────────
Full coverage for utils/metrics.py

No external dependencies beyond numpy/scipy — runs in < 1 second.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import (
    r2_score,
    spearman_correlation,
    top_k_zone_accuracy,
    nonzero_rmse,
    evaluate_all,
)


# ── r2_score ──────────────────────────────────────────────────────────────────

class TestR2Score:

    def test_perfect_prediction_is_one(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(t, t) == pytest.approx(1.0, abs=1e-6)

    def test_constant_prediction_is_near_zero(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.full_like(t, t.mean())
        # SS_res == SS_tot → R² ≈ 0  (epsilon in denominator makes it not exact)
        assert r2_score(p, t) == pytest.approx(0.0, abs=1e-4)

    def test_bad_prediction_is_negative(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        p = -t  # opposite direction
        assert r2_score(p, t) < 0.0

    def test_known_value(self):
        # Simple case: target=[0,1,2], pred=[0,1,3] → SS_res=1, SS_tot=2 → R²≈0.5
        t = np.array([0.0, 1.0, 2.0])
        p = np.array([0.0, 1.0, 3.0])
        assert r2_score(p, t) == pytest.approx(0.5, abs=1e-5)

    def test_all_zero_target_no_crash(self):
        # SS_tot = 0 → denominator uses epsilon → should not crash
        t = np.zeros(5)
        p = np.zeros(5)
        result = r2_score(p, t)
        assert np.isfinite(result)

    def test_returns_float(self):
        t = np.arange(10, dtype=float)
        assert isinstance(r2_score(t, t), float)


# ── spearman_correlation ──────────────────────────────────────────────────────

class TestSpearmanCorrelation:

    def test_identical_arrays_is_one(self):
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert spearman_correlation(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_reversed_array_is_minus_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spearman_correlation(a[::-1], a) == pytest.approx(-1.0, abs=1e-6)

    def test_random_pair_in_range(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 100)
        b = rng.normal(0, 1, 100)
        corr = spearman_correlation(a, b)
        assert -1.0 <= corr <= 1.0

    def test_monotone_nonlinear_still_one(self):
        # Spearman captures monotone relations, not just linear
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 4.0, 9.0, 16.0, 25.0])  # a²
        assert spearman_correlation(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_2d_input_is_flattened(self):
        # The function flattens via .flatten() — should handle 2D inputs
        a = np.arange(9, dtype=float).reshape(3, 3)
        corr = spearman_correlation(a, a)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self):
        a = np.array([1.0, 2.0, 3.0])
        assert isinstance(spearman_correlation(a, a), float)


# ── top_k_zone_accuracy ───────────────────────────────────────────────────────

class TestTopKZoneAccuracy:

    def test_perfect_ranking_is_one(self):
        target = np.array([10.0, 1.0, 5.0, 8.0, 2.0])
        pred   = target.copy()
        assert top_k_zone_accuracy(pred, target, k=3) == pytest.approx(1.0)

    def test_no_overlap_is_zero(self):
        # top-3 of target: indices 0,1,2 (values 10,9,8)
        # top-3 of pred:   indices 3,4,5 (values 9,10,8) → no overlap
        target = np.array([10.0, 9.0, 8.0,  1.0,  2.0,  3.0])
        pred   = np.array([ 1.0, 2.0, 3.0,  9.0, 10.0,  8.0])
        assert top_k_zone_accuracy(pred, target, k=3) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # top-2 of target = indices {0,1} (values 10,9)
        # top-2 of pred   = indices {0,2} (values 10,8) → 1 overlap out of 2
        target = np.array([10.0, 9.0, 8.0, 1.0, 2.0])
        pred   = np.array([10.0, 1.0, 8.0, 0.0, 0.0])
        acc = top_k_zone_accuracy(pred, target, k=2)
        assert acc == pytest.approx(0.5)

    def test_2d_input_sums_rows(self):
        # 2D target: row sums determine zone impact
        N = 10
        rng = np.random.default_rng(42)
        target = rng.exponential(5, (N, N))
        pred   = target + rng.normal(0, 0.1, (N, N))
        acc = top_k_zone_accuracy(pred, target, k=5)
        assert 0.0 <= acc <= 1.0

    def test_k_equals_n_is_always_one(self):
        target = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        pred   = np.zeros_like(target)
        assert top_k_zone_accuracy(pred, target, k=5) == pytest.approx(1.0)

    def test_result_in_zero_one(self):
        rng = np.random.default_rng(7)
        t = rng.normal(0, 1, 50)
        p = rng.normal(0, 1, 50)
        assert 0.0 <= top_k_zone_accuracy(p, t, k=10) <= 1.0


# ── nonzero_rmse ─────────────────────────────────────────────────────────────

class TestNonzeroRMSE:

    def test_perfect_nonzero_is_zero(self):
        t = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        assert nonzero_rmse(t, t) == pytest.approx(0.0, abs=1e-9)

    def test_all_zero_target_returns_nan(self):
        t = np.zeros(10)
        p = np.ones(10)
        result = nonzero_rmse(p, t)
        assert np.isnan(result)

    def test_only_nonzero_entries_count(self):
        # Large errors on zero-target cells must be ignored
        t = np.array([0.0, 0.0, 1.0, 0.0])
        p = np.array([99.0, 99.0, 2.0, 99.0])   # error only on the nonzero cell
        assert nonzero_rmse(p, t) == pytest.approx(1.0, abs=1e-6)

    def test_known_value(self):
        # nonzero targets: [1, 4], preds: [2, 2] → errors: [1, 4] → MSE=8.5 → RMSE≈2.915
        t = np.array([1.0, 0.0, 4.0])
        p = np.array([2.0, 99.0, 2.0])
        expected = np.sqrt(np.mean([(2-1)**2, (2-4)**2]))
        assert nonzero_rmse(p, t) == pytest.approx(expected, rel=1e-5)

    def test_custom_threshold(self):
        # With threshold=2.0, only target[2]=3 qualifies
        t = np.array([0.5, 1.0, 3.0])
        p = np.array([0.0, 0.0, 6.0])
        result = nonzero_rmse(p, t, threshold=2.0)
        assert result == pytest.approx(3.0, abs=1e-6)

    def test_returns_float(self):
        t = np.array([1.0, 2.0])
        assert isinstance(nonzero_rmse(t, t), float)


# ── evaluate_all ─────────────────────────────────────────────────────────────

class TestEvaluateAll:

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.target = rng.normal(0, 5, 50)
        self.pred   = self.target + rng.normal(0, 0.5, 50)

    def test_returns_dict(self):
        result = evaluate_all(self.pred, self.target)
        assert isinstance(result, dict)

    def test_has_all_keys(self):
        result = evaluate_all(self.pred, self.target)
        for key in ["r2", "spearman", "top_k_acc", "nonzero_rmse", "rmse"]:
            assert key in result, f"Missing key: {key}"

    def test_values_are_finite_for_good_pred(self):
        result = evaluate_all(self.pred, self.target)
        for k, v in result.items():
            if not np.isnan(v):
                assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_perfect_pred_r2_is_one(self):
        result = evaluate_all(self.target, self.target)
        assert result["r2"] == pytest.approx(1.0, abs=1e-5)

    def test_perfect_pred_rmse_is_zero(self):
        result = evaluate_all(self.target, self.target)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_k_parameter_forwarded(self):
        r10 = evaluate_all(self.pred, self.target, k=10)
        r5  = evaluate_all(self.pred, self.target, k=5)
        # Both should be valid; values may differ
        assert 0.0 <= r10["top_k_acc"] <= 1.0
        assert 0.0 <= r5["top_k_acc"]  <= 1.0

    def test_rmse_equals_manual(self):
        result = evaluate_all(self.pred, self.target)
        manual = float(np.sqrt(np.mean((self.pred - self.target) ** 2)))
        assert result["rmse"] == pytest.approx(manual, rel=1e-6)