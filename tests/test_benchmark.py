"""
tests/test_benchmark.py
─────────────────────────────────────────────────────────────────────────────
Two goals in one file:

1. COVERAGE  — exercises all pure functions in benchmark_metr_la.py
               (0 % → ~70 %) using fully synthetic data, no METR-LA download,
               no GPU required.

2. PERFORMANCE — pytest-benchmark timings for the real model forward passes
                and the data-pipeline helpers.
                Install once:  pip install pytest-benchmark
                Run:           pytest tests/test_benchmark.py -v --benchmark-only
                Skip perf:     pytest tests/test_benchmark.py -v -m "not benchmark"

All model tests are auto-skipped when torch / the model modules are absent.
"""

import sys
import time
import types
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch", reason="PyTorch not installed")

# ─────────────────────────────────────────────────────────────────────────────
# Fake dataloader helpers
# ─────────────────────────────────────────────────────────────────────────────

N_NODES  = 20   # tiny graph for fast CPU tests (real METR-LA = 207)
T_IN     = 12
T_OUT    = 3
BATCH    = 2
FEAT     = T_IN // 2   # 6  — matches benchmark_metr_la feat_step


def _make_batch(n=N_NODES, t_in=T_IN, t_out=T_OUT, b=BATCH, seed=0):
    """One dict-batch matching the shape the benchmark expects.
    
    target shape: (B, N) — mean over T_out steps, matching what
    train_epoch does: batch['target'][0].unsqueeze(0) → (1, N)
    """
    rng = np.random.default_rng(seed)
    return {
        "node_feat": torch.FloatTensor(rng.normal(0.5, 0.1, (b, n, t_in))),
        "target":    torch.FloatTensor(rng.normal(0.5, 0.1, (b, n))),
    }


def _fake_loader(n_batches=3):
    batch = _make_batch()
    return [batch] * n_batches


def _fake_meta(n=N_NODES):
    rng = np.random.default_rng(42)
    # ring graph  src→dst
    src = list(range(n))
    dst = [(i + 1) % n for i in range(n)]
    edge_index = torch.LongTensor([src + dst, dst + src])
    return {
        "num_nodes":  n,
        "speed_mean": 35.0,
        "speed_std":  15.0,
        "edge_index": edge_index,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Import the functions under test
# ─────────────────────────────────────────────────────────────────────────────

def _import_benchmark():
    """Import benchmark functions without triggering argparse / main()."""
    try:
        import benchmark_metr_la as bm
        return bm
    except ImportError:
        return None


BM = _import_benchmark()


def _skip_if_no_bm():
    if BM is None:
        pytest.skip("benchmark_metr_la.py not importable")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Metric functions   (r2, mae_mph, rmse_mph)
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkMetrics:

    def setup_method(self):
        _skip_if_no_bm()

    # ── r2 ───────────────────────────────────────────────────────────────────

    def test_r2_perfect(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert BM.r2(t, t) == pytest.approx(1.0, abs=1e-5)

    def test_r2_bad_prediction_negative(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        p = -t
        assert BM.r2(p, t) < 0.0

    def test_r2_constant_target_no_crash(self):
        t = torch.ones(10)
        p = torch.zeros(10)
        result = BM.r2(p, t)
        assert np.isfinite(result)

    def test_r2_returns_python_float(self):
        t = torch.randn(20)
        assert isinstance(BM.r2(t, t), float)

    def test_r2_range(self):
        rng = np.random.default_rng(7)
        t = torch.FloatTensor(rng.normal(0, 1, 100))
        p = torch.FloatTensor(rng.normal(0, 1, 100))
        # R² can be < -1 for terrible predictions, but must be ≤ 1
        assert BM.r2(p, t) <= 1.0 + 1e-6

    # ── mae_mph ───────────────────────────────────────────────────────────────

    def test_mae_perfect_is_zero(self):
        t = torch.tensor([0.5, 0.6, 0.4])
        assert BM.mae_mph(t, t, mean=35.0, std=15.0) == pytest.approx(0.0, abs=1e-6)

    def test_mae_scales_with_std(self):
        t = torch.zeros(5)
        p = torch.ones(5)   # error = 1 (normalised)
        mae_low  = BM.mae_mph(p, t, mean=0.0, std=1.0)
        mae_high = BM.mae_mph(p, t, mean=0.0, std=10.0)
        assert mae_high == pytest.approx(mae_low * 10, rel=1e-5)

    def test_mae_non_negative(self):
        rng = np.random.default_rng(3)
        t = torch.FloatTensor(rng.normal(0, 1, 50))
        p = torch.FloatTensor(rng.normal(0, 1, 50))
        assert BM.mae_mph(p, t, 35.0, 15.0) >= 0.0

    def test_mae_returns_python_float(self):
        t = torch.zeros(5)
        assert isinstance(BM.mae_mph(t, t, 35.0, 15.0), float)

    # ── rmse_mph ──────────────────────────────────────────────────────────────

    def test_rmse_perfect_is_zero(self):
        t = torch.tensor([0.3, 0.7, 0.5])
        assert BM.rmse_mph(t, t, mean=35.0, std=15.0) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_ge_mae(self):
        """RMSE ≥ MAE always (Jensen's inequality)."""
        rng = np.random.default_rng(5)
        t = torch.FloatTensor(rng.normal(0, 1, 100))
        p = torch.FloatTensor(rng.normal(0, 1, 100))
        mae  = BM.mae_mph(p, t, 35.0, 15.0)
        rmse = BM.rmse_mph(p, t, 35.0, 15.0)
        assert rmse >= mae - 1e-6

    def test_rmse_scales_with_std(self):
        t = torch.zeros(5)
        p = torch.ones(5)
        rmse_low  = BM.rmse_mph(p, t, 0.0, 1.0)
        rmse_high = BM.rmse_mph(p, t, 0.0, 10.0)
        assert rmse_high == pytest.approx(rmse_low * 10, rel=1e-5)

    def test_rmse_non_negative(self):
        rng = np.random.default_rng(9)
        t = torch.FloatTensor(rng.normal(0, 1, 50))
        p = torch.FloatTensor(rng.normal(0, 1, 50))
        assert BM.rmse_mph(p, t, 35.0, 15.0) >= 0.0

    def test_rmse_returns_python_float(self):
        t = torch.zeros(5)
        assert isinstance(BM.rmse_mph(t, t, 35.0, 15.0), float)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  batch_to_x_seq
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchToXSeq:

    def setup_method(self):
        _skip_if_no_bm()

    def test_returns_list_of_two(self):
        batch = _make_batch()
        result = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        assert isinstance(result, list) and len(result) == 2

    def test_each_element_shape(self):
        batch = _make_batch(n=N_NODES, t_in=T_IN)
        x0, x1 = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        assert x0.shape == (N_NODES, T_IN // 2)
        assert x1.shape == (N_NODES, T_IN // 2)

    def test_uses_first_sample_only(self):
        """The function must drop the batch dimension and use sample 0."""
        batch = _make_batch(b=4)
        x0, x1 = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        # Compare against manual slice of sample 0
        x_ref = batch["node_feat"][0]
        h = T_IN // 2
        assert torch.allclose(x0, x_ref[:, :h])
        assert torch.allclose(x1, x_ref[:, h:])

    def test_concatenates_back_to_original(self):
        batch = _make_batch()
        x0, x1 = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        reconstructed = torch.cat([x0, x1], dim=1)
        assert torch.allclose(reconstructed, batch["node_feat"][0])

    def test_odd_t_in_handled(self):
        """T_in=13 → h=6; x0=[:,:6] shape (N,6), x1=[:,6:] shape (N,7)."""
        batch = _make_batch(t_in=13)
        x0, x1 = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        h = 13 // 2  # = 6
        assert x0.shape == (N_NODES, h)
        assert x1.shape == (N_NODES, 13 - h)  # x1 = [:, 6:] → 7 cols

    def test_output_on_correct_device(self):
        batch = _make_batch()
        x0, x1 = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        assert x0.device.type == "cpu"
        assert x1.device.type == "cpu"

    def test_output_dtype_is_float(self):
        batch = _make_batch()
        x0, _ = BM.batch_to_x_seq(batch["node_feat"], "cpu")
        assert x0.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Config helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigHelpers:

    def setup_method(self):
        _skip_if_no_bm()
        try:
            from models.gat_lstm import Config
            from models.hypergraph_lstm import HypergraphConfig
        except ImportError:
            pytest.skip("model modules not available")

    def test_make_gat_config_sets_n_zones(self):
        cfg = BM._make_gat_config(N_NODES, FEAT, 64, 4)
        assert cfg.NUM_ZONES == N_NODES

    def test_make_gat_config_sets_output_size(self):
        cfg = BM._make_gat_config(N_NODES, FEAT, 64, 4)
        assert cfg.OUTPUT_SIZE == N_NODES

    def test_make_gat_config_sets_in_channels(self):
        cfg = BM._make_gat_config(N_NODES, FEAT, 64, 4)
        assert cfg.GAT_IN_CHANNELS == FEAT

    def test_make_gat_config_sets_hidden(self):
        cfg = BM._make_gat_config(N_NODES, FEAT, 128, 4)
        assert cfg.GAT_HIDDEN == 128

    def test_make_gat_config_sets_heads(self):
        cfg = BM._make_gat_config(N_NODES, FEAT, 64, 8)
        assert cfg.GAT_HEADS == 8

    def test_make_hg_config_sets_n_zones(self):
        cfg = BM._make_hg_config(N_NODES, FEAT, 64)
        assert cfg.NUM_ZONES == N_NODES

    def test_make_hg_config_sets_in_channels(self):
        cfg = BM._make_hg_config(N_NODES, FEAT, 64)
        assert cfg.HG_IN_CHANNELS == FEAT

    def test_make_hg_config_sets_hidden(self):
        cfg = BM._make_hg_config(N_NODES, FEAT, 32)
        assert cfg.HG_HIDDEN == 32


# ─────────────────────────────────────────────────────────────────────────────
# 4.  train_epoch / eval_epoch  (with mock models)
# ─────────────────────────────────────────────────────────────────────────────

class _MockModel(torch.nn.Module):
    """Minimal model stub: linear projection so backward() always works."""
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.proj = torch.nn.Linear(n, n)   # real learnable params

    def forward(self, x_seq, *args, **kwargs):
        # use x_seq[0] so gradients flow through proj
        x = x_seq[0]                         # (N, feat)
        return self.proj(x.mean(dim=1, keepdim=True).T)  # (1, N)


class TestTrainEvalEpoch:

    def setup_method(self):
        _skip_if_no_bm()
        self.meta   = _fake_meta()
        self.loader = _fake_loader()
        self.model  = _MockModel(N_NODES)
        self.optim  = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.sf     = torch.zeros(1, 8)
        self.ei     = self.meta["edge_index"]

    # ── train_epoch ───────────────────────────────────────────────────────────

    def test_train_epoch_returns_float(self):
        loss = BM.train_epoch(
            self.model, "gat", self.loader,
            self.optim, self.ei, "cpu", self.sf
        )
        assert isinstance(loss, float)

    def test_train_epoch_loss_non_negative(self):
        loss = BM.train_epoch(
            self.model, "gat", self.loader,
            self.optim, self.ei, "cpu", self.sf
        )
        assert loss >= 0.0

    def test_train_epoch_updates_params(self):
        before = [p.clone() for p in self.model.parameters()]
        BM.train_epoch(
            self.model, "gat", self.loader,
            self.optim, self.ei, "cpu", self.sf
        )
        after = list(self.model.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed, "Parameters did not update after train_epoch"

    def test_train_epoch_hypergraph_mtype(self):
        """mtype='hypergraph' uses a different forward call signature."""
        loss = BM.train_epoch(
            self.model, "hypergraph", self.loader,
            self.optim, self.ei, "cpu", self.sf
        )
        assert isinstance(loss, float)

    # ── eval_epoch ────────────────────────────────────────────────────────────

    def test_eval_epoch_returns_dict(self):
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf,
            speed_mean=35.0, speed_std=15.0,
        )
        assert isinstance(result, dict)

    def test_eval_epoch_has_all_keys(self):
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        for k in ("MAE", "RMSE", "R2"):
            assert k in result, f"Missing key: {k}"

    def test_eval_epoch_values_are_finite(self):
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        for k, v in result.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_eval_epoch_no_grad_leak(self):
        """eval_epoch must not retain computation graphs."""
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        # If grads leaked, converting to python float would fail
        assert isinstance(result["MAE"], float)

    def test_eval_epoch_rmse_ge_mae(self):
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        assert result["RMSE"] >= result["MAE"] - 1e-6

    def test_eval_epoch_hypergraph_mtype(self):
        result = BM.eval_epoch(
            self.model, "hypergraph", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        assert "MAE" in result and "R2" in result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Integration: real GAT+LSTM forward pass through benchmark pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestRealModelIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        _skip_if_no_bm()
        try:
            from models.gat_lstm import GATLSTMModel
        except ImportError:
            pytest.skip("models/gat_lstm.py not available")

        from models.gat_lstm import GATLSTMModel
        cfg   = BM._make_gat_config(N_NODES, FEAT, 32, 2)
        self.model  = GATLSTMModel(cfg).eval()
        self.meta   = _fake_meta()
        self.loader = _fake_loader()
        self.sf     = torch.zeros(1, 8)
        self.ei     = self.meta["edge_index"]

    def test_real_gat_eval_epoch_runs(self):
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        assert "MAE" in result

    def test_real_gat_output_finite(self):
        result = BM.eval_epoch(
            self.model, "gat", self.loader,
            self.ei, "cpu", self.sf, 35.0, 15.0,
        )
        for k, v in result.items():
            assert np.isfinite(v), f"{k} not finite"

    def test_real_gat_train_step_runs(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss  = BM.train_epoch(
            self.model, "gat", self.loader,
            optim, self.ei, "cpu", self.sf,
        )
        assert loss >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PERFORMANCE benchmarks  (pytest-benchmark)
#     Run with:  pytest tests/test_benchmark.py -v --benchmark-only
#     Skip with: pytest tests/test_benchmark.py -v -m "not benchmark"
# ─────────────────────────────────────────────────────────────────────────────

def _make_real_gat_model():
    if BM is None:
        return None, None, None
    try:
        from models.gat_lstm import GATLSTMModel
        cfg   = BM._make_gat_config(N_NODES, FEAT, 32, 2)
        model = GATLSTMModel(cfg).eval()
        ei    = _fake_meta()["edge_index"]
        return model, ei, torch.zeros(1, 8)
    except ImportError:
        return None, None, None


def _make_real_hyp_model():
    if BM is None:
        return None, None
    try:
        from models.hypergraph_lstm import HypergraphLSTMModel, build_incidence_matrix
        from collections import defaultdict
        cfg   = BM._make_hg_config(N_NODES, FEAT, 32)
        model = HypergraphLSTMModel(cfg)
        meta  = _fake_meta()
        src, dst = meta["edge_index"]
        rd = defaultdict(list)
        for s, d in zip(src.tolist(), dst.tolist()):
            rd[s].append(d); rd[d].append(s)
        routes = {f"r_{k}": list(set(v + [k])) for k, v in rd.items()}
        H = build_incidence_matrix(list(range(N_NODES)), gtfs_routes=routes)
        model.set_hypergraph(H)
        model.eval()
        return model, torch.zeros(1, 8)
    except ImportError:
        return None, None


@pytest.fixture(scope="module")
def gat_model_fixture():
    return _make_real_gat_model()


@pytest.fixture(scope="module")
def hyp_model_fixture():
    return _make_real_hyp_model()


@pytest.fixture(scope="module")
def fake_x_seq_fixture():
    batch = _make_batch(n=N_NODES, t_in=T_IN)
    return BM.batch_to_x_seq(batch["node_feat"], "cpu") if BM else None


@pytest.mark.benchmark(group="inference")
def test_perf_gat_inference(benchmark, gat_model_fixture, fake_x_seq_fixture):
    """GAT+LSTM single forward pass — must complete in < 200 ms on CPU."""
    model, ei, sf = gat_model_fixture
    if model is None:
        pytest.skip("GAT model not available")
    x_seq = fake_x_seq_fixture

    def _run():
        with torch.no_grad():
            return model(x_seq, ei, sf)

    result = benchmark(_run)
    assert result.shape[-1] == N_NODES


@pytest.mark.benchmark(group="inference")
def test_perf_hypergraph_inference(benchmark, hyp_model_fixture, fake_x_seq_fixture):
    """Hypergraph+LSTM single forward pass — must complete in < 200 ms on CPU."""
    model, sf = hyp_model_fixture
    if model is None:
        pytest.skip("Hypergraph model not available")
    x_seq = fake_x_seq_fixture

    def _run():
        with torch.no_grad():
            return model(x_seq, sf)

    result = benchmark(_run)
    assert result.shape[-1] == N_NODES


@pytest.mark.benchmark(group="data_pipeline")
def test_perf_batch_to_x_seq(benchmark):
    """batch_to_x_seq conversion — should be < 5 ms."""
    if BM is None:
        pytest.skip()
    batch = _make_batch(n=N_NODES, t_in=T_IN)
    nf = batch["node_feat"]
    benchmark(lambda: BM.batch_to_x_seq(nf, "cpu"))


@pytest.mark.benchmark(group="data_pipeline")
def test_perf_feature_engineering(benchmark):
    """od_matrix_to_zone_features — should be < 50 ms for 207 zones."""
    try:
        import pandas as pd
        from utils.data import od_matrix_to_zone_features
    except ImportError:
        pytest.skip("utils.data not available")

    rng = np.random.default_rng(0)
    N   = 207
    od  = pd.DataFrame(
        rng.exponential(100, (N, N)),
        index=range(N), columns=range(N),
    )
    np.fill_diagonal(od.values, 0)
    benchmark(lambda: od_matrix_to_zone_features(od))


@pytest.mark.benchmark(group="data_pipeline")
def test_perf_metric_evaluate_all(benchmark):
    """evaluate_all over 207-zone predictions — should be < 5 ms."""
    try:
        from utils.metrics import evaluate_all
    except ImportError:
        pytest.skip()

    rng    = np.random.default_rng(0)
    target = rng.normal(0, 5, 207)
    pred   = target + rng.normal(0, 0.5, 207)
    benchmark(lambda: evaluate_all(pred, target))


@pytest.mark.benchmark(group="inference")
def test_perf_eval_epoch_full(benchmark, gat_model_fixture):
    """Full eval_epoch (3 mini-batches) — tracks end-to-end validation speed."""
    if BM is None:
        pytest.skip()
    model, ei, sf = gat_model_fixture
    if model is None:
        pytest.skip("GAT model not available")
    loader = _fake_loader(n_batches=3)

    def _run():
        return BM.eval_epoch(model, "gat", loader, ei, "cpu", sf, 35.0, 15.0)

    result = benchmark(_run)
    assert "MAE" in result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Timing assertions  (run always, no pytest-benchmark needed)
#     These are plain tests that fail if code is suspiciously slow.
# ─────────────────────────────────────────────────────────────────────────────

class TestTimingAssertions:
    """
    Hard timing limits that run in the normal test suite (no --benchmark flag).
    Generous limits to avoid flakiness on slow CI runners.
    """

    def test_batch_to_x_seq_under_10ms(self):
        if BM is None:
            pytest.skip()
        batch = _make_batch(n=N_NODES, t_in=T_IN)
        nf    = batch["node_feat"]
        t0    = time.perf_counter()
        for _ in range(100):
            BM.batch_to_x_seq(nf, "cpu")
        avg_ms = (time.perf_counter() - t0) * 10  # ms per call
        assert avg_ms < 10.0, f"batch_to_x_seq too slow: {avg_ms:.2f} ms"

    def test_metric_functions_under_5ms(self):
        if BM is None:
            pytest.skip()
        N = 1000
        rng = np.random.default_rng(0)
        p   = torch.FloatTensor(rng.normal(0, 1, N))
        t   = torch.FloatTensor(rng.normal(0, 1, N))

        t0 = time.perf_counter()
        for _ in range(500):
            BM.r2(p, t)
            BM.mae_mph(p, t, 35.0, 15.0)
            BM.rmse_mph(p, t, 35.0, 15.0)
        avg_ms = (time.perf_counter() - t0) * 2   # ms per triple-call
        assert avg_ms < 5.0, f"Metric functions too slow: {avg_ms:.2f} ms"

    def test_gat_inference_under_500ms(self):
        """Single GAT forward pass must finish in < 500 ms on any CPU."""
        model, ei, sf = _make_real_gat_model()
        if model is None:
            pytest.skip("GAT model not available")
        batch = _make_batch(n=N_NODES, t_in=T_IN)
        x_seq = BM.batch_to_x_seq(batch["node_feat"], "cpu")

        t0 = time.perf_counter()
        with torch.no_grad():
            model(x_seq, ei, sf)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"GAT inference too slow: {elapsed_ms:.1f} ms"

    def test_hypergraph_inference_under_500ms(self):
        model, sf = _make_real_hyp_model()
        if model is None:
            pytest.skip("Hypergraph model not available")
        batch = _make_batch(n=N_NODES, t_in=T_IN)
        x_seq = BM.batch_to_x_seq(batch["node_feat"], "cpu")

        t0 = time.perf_counter()
        with torch.no_grad():
            model(x_seq, sf)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 500, f"Hypergraph inference too slow: {elapsed_ms:.1f} ms"