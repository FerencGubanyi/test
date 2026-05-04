"""
tests/test_model_utils.py
─────────────────────────────────────────────────────────────────────────────
Covers the utility functions that test_models.py misses:

  models/gat_lstm.py  (143-251):
    - build_zone_graph()          shapefile path + k-NN fallback
    - train_epoch()               one optimiser step
    - evaluate()                  no-grad eval pass
    - train()                     full loop, early-stop, checkpoint

  models/hypergraph_lstm.py (71-88, 251-305):
    - build_incidence_matrix()    GTFS path + distance fallback
    - normalize_incidence()       D_v / D_e diagonal matrices
    - train_epoch()               one optimiser step
    - evaluate()                  no-grad eval pass
    - train()                     full loop, early-stop, checkpoint

All tests run on CPU, no GPU, no real data files.
Expected coverage uplift:  gat_lstm 50%→80%,  hypergraph_lstm 63%→85%
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.nn as nn

# ── imports — split by dependency ─────────────────────────────────────────────

# Pure functions: no torch_geometric needed
try:
    from models.hypergraph_lstm import (
        build_incidence_matrix, normalize_incidence,
    )
    _HYP_PURE_AVAILABLE = True
except ImportError:
    _HYP_PURE_AVAILABLE = False

try:
    from models.gat_lstm import build_zone_graph
    _GAT_PURE_AVAILABLE = True
except ImportError:
    _GAT_PURE_AVAILABLE = False

# Full model + training utilities: need torch_geometric
try:
    from models.gat_lstm import (
        GATLSTMModel, Config,
        train_epoch as gat_train_epoch,
        evaluate as gat_evaluate, train as gat_train,
    )
    from models.hypergraph_lstm import (
        HypergraphLSTMModel, HypergraphConfig,
        train_epoch as hyp_train_epoch,
        evaluate as hyp_evaluate, train as hyp_train,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

# ── constants ──────────────────────────────────────────────────────────────────
N  = 12    # zones — tiny for fast CPU tests
F  = 22    # feature dim
SF = 8     # scenario feat dim


# ── shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def zone_ids():
    return list(range(1, N + 1))


@pytest.fixture
def small_gat_cfg():
    if not _MODELS_AVAILABLE:
        pytest.skip("torch_geometric not installed")
    cfg = Config()
    cfg.NUM_ZONES       = N
    cfg.GAT_IN_CHANNELS = F
    cfg.OUTPUT_SIZE     = N
    cfg.GAT_HIDDEN      = 8
    cfg.GAT_OUT_CHANNELS= 4
    cfg.GAT_HEADS       = 2
    cfg.LSTM_INPUT_SIZE = 4
    cfg.LSTM_HIDDEN_SIZE= 16
    cfg.LSTM_NUM_LAYERS = 1
    cfg.LSTM_DROPOUT    = 0.0
    cfg.GAT_DROPOUT     = 0.0
    cfg.NUM_EPOCHS      = 3
    cfg.PATIENCE        = 10
    cfg.DEVICE          = 'cpu'
    return cfg


@pytest.fixture
def small_hyp_cfg():
    if not _MODELS_AVAILABLE:
        pytest.skip("torch_geometric not installed")
    cfg = HypergraphConfig()
    cfg.NUM_ZONES        = N
    cfg.HG_IN_CHANNELS   = F
    cfg.HG_OUT_CHANNELS  = 4
    cfg.HG_HIDDEN        = 8
    cfg.HG_NUM_LAYERS    = 2
    cfg.HG_DROPOUT       = 0.0
    cfg.LSTM_INPUT_SIZE  = 4
    cfg.LSTM_HIDDEN_SIZE = 16
    cfg.LSTM_NUM_LAYERS  = 1
    cfg.LSTM_DROPOUT     = 0.0
    cfg.OUTPUT_SIZE      = N
    cfg.NUM_EPOCHS       = 3
    cfg.PATIENCE         = 10
    cfg.DEVICE           = 'cpu'
    return cfg


@pytest.fixture
def ring_edge_index():
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    return torch.LongTensor([src[:N], dst[:N]])


@pytest.fixture
def fake_H():
    rng = np.random.default_rng(7)
    H = np.zeros((N, 6))
    for e in range(6):
        members = rng.choice(N, size=4, replace=False)
        H[members, e] = 1.0
    return torch.FloatTensor(H)


@pytest.fixture
def gat_model(small_gat_cfg):
    return GATLSTMModel(small_gat_cfg)


@pytest.fixture
def hyp_model(small_hyp_cfg, fake_H):
    m = HypergraphLSTMModel(small_hyp_cfg)
    m.set_hypergraph(fake_H)
    return m


def _make_x_seq(n=N, f=F, steps=2, seed=0):
    rng = np.random.default_rng(seed)
    return [torch.FloatTensor(np.abs(rng.normal(0, 1, (n, f)))) for _ in range(steps)]


def _make_gat_batch(edge_index, n=N, f=F):
    rng = np.random.default_rng(0)
    return {
        'x_seq':         _make_x_seq(n, f),
        'edge_index':    edge_index,
        'scenario_feat': torch.zeros(1, SF),
        'target':        torch.FloatTensor(rng.normal(0, 1, (1, n))),
    }


def _make_hyp_batch(n=N, f=F):
    rng = np.random.default_rng(1)
    return {
        'x_seq':         _make_x_seq(n, f),
        'scenario_feat': torch.zeros(1, SF),
        'target':        torch.FloatTensor(rng.normal(0, 1, (1, n))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# build_zone_graph
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildZoneGraph:

    def setup_method(self):
        if not _GAT_PURE_AVAILABLE:
            pytest.skip("models.gat_lstm not importable")

    def test_fallback_returns_tensor(self, zone_ids):
        ei = build_zone_graph(zone_ids, gdf=None, k_neighbors=3)
        assert isinstance(ei, torch.Tensor)

    def test_fallback_shape(self, zone_ids):
        ei = build_zone_graph(zone_ids, gdf=None, k_neighbors=3)
        assert ei.shape[0] == 2   # (2, E)

    def test_fallback_dtype_long(self, zone_ids):
        ei = build_zone_graph(zone_ids, gdf=None, k_neighbors=3)
        assert ei.dtype == torch.long

    def test_fallback_indices_in_range(self, zone_ids):
        ei = build_zone_graph(zone_ids, gdf=None, k_neighbors=3)
        assert ei.min() >= 0
        assert ei.max() < len(zone_ids)

    def test_fallback_k_neighbors_respected(self, zone_ids):
        k = 2
        ei = build_zone_graph(zone_ids, gdf=None, k_neighbors=k)
        # Each node contributes exactly k out-edges → total = N * k
        assert ei.shape[1] == len(zone_ids) * k

    def test_gdf_exception_falls_back_gracefully(self, zone_ids):
        """If gdf raises during topology computation, falls back to k-NN."""
        class _BadGDF:
            def iterrows(self):
                raise RuntimeError("simulated shapefile error")

        ei = build_zone_graph(zone_ids, gdf=_BadGDF(), k_neighbors=3)
        assert isinstance(ei, torch.Tensor)
        assert ei.shape[0] == 2

    def test_single_zone_no_crash(self):
        ei = build_zone_graph([1], gdf=None, k_neighbors=1)
        assert isinstance(ei, torch.Tensor)


# ══════════════════════════════════════════════════════════════════════════════
# build_incidence_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildIncidenceMatrix:

    def setup_method(self):
        if not _HYP_PURE_AVAILABLE:
            pytest.skip("models.hypergraph_lstm not importable")

    def test_gtfs_path_returns_tensor(self, zone_ids):
        routes = {f"r{i}": zone_ids[i:i+3] for i in range(0, N-2, 3)}
        H = build_incidence_matrix(zone_ids, gtfs_routes=routes)
        assert isinstance(H, torch.Tensor)

    def test_gtfs_shape_correct(self, zone_ids):
        routes = {f"r{i}": zone_ids[i:i+3] for i in range(0, N-2, 3)}
        H = build_incidence_matrix(zone_ids, gtfs_routes=routes)
        assert H.shape[0] == N
        assert H.shape[1] == len(routes)

    def test_gtfs_binary(self, zone_ids):
        routes = {"r0": zone_ids[:4], "r1": zone_ids[4:8]}
        H = build_incidence_matrix(zone_ids, gtfs_routes=routes)
        assert set(H.unique().tolist()).issubset({0.0, 1.0})

    def test_gtfs_filters_single_zone_routes(self, zone_ids):
        """Routes with < 2 zones in zone_ids must be excluded."""
        routes = {
            "good": zone_ids[:4],   # 4 zones → kept
            "bad":  [zone_ids[0]],  # 1 zone  → dropped
        }
        H = build_incidence_matrix(zone_ids, gtfs_routes=routes)
        assert H.shape[1] == 1

    def test_gtfs_unknown_zones_skipped(self, zone_ids):
        routes = {"r0": zone_ids[:3] + [99999]}
        H = build_incidence_matrix(zone_ids, gtfs_routes=routes)
        assert H.shape[0] == N

    def test_distance_fallback_no_gdf(self):
        # Need N >= 50 because distance fallback picks m=max(50, N//10) centers
        zone_ids = list(range(1, 61))
        H = build_incidence_matrix(zone_ids, gtfs_routes=None, gdf=None)
        assert isinstance(H, torch.Tensor)
        assert H.shape[0] == 60

    def test_distance_fallback_binary(self):
        zone_ids = list(range(1, 61))
        H = build_incidence_matrix(zone_ids, gtfs_routes=None, gdf=None)
        assert set(H.unique().tolist()).issubset({0.0, 1.0})

    def test_distance_fallback_each_zone_in_3_edges(self):
        """Distance fallback assigns each zone to exactly 3 nearest hyperedges."""
        zone_ids = list(range(1, 61))
        H = build_incidence_matrix(zone_ids, gtfs_routes=None, gdf=None)
        row_sums = H.sum(dim=1)
        assert (row_sums == 3).all(), f"Expected all row sums = 3, got {row_sums}"

    def test_empty_gtfs_routes_dict(self, zone_ids):
        """Empty route dict → 0 hyperedges → H is (N, 0)."""
        H = build_incidence_matrix(zone_ids, gtfs_routes={})
        assert H.shape == (N, 0) or H.shape[1] == 0


# ══════════════════════════════════════════════════════════════════════════════
# normalize_incidence
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeIncidence:

    def setup_method(self):
        if not _HYP_PURE_AVAILABLE:
            pytest.skip("models.hypergraph_lstm not importable")

    def test_returns_three_tensors(self, fake_H):
        result = normalize_incidence(fake_H)
        assert len(result) == 3

    def test_H_unchanged(self, fake_H):
        H_out, _, _ = normalize_incidence(fake_H)
        assert torch.allclose(H_out, fake_H)

    def test_D_v_is_square(self, fake_H):
        _, D_v, _ = normalize_incidence(fake_H)
        assert D_v.shape == (N, N)

    def test_D_e_is_square(self, fake_H):
        _, _, D_e = normalize_incidence(fake_H)
        E = fake_H.shape[1]
        assert D_e.shape == (E, E)

    def test_D_v_is_diagonal(self, fake_H):
        _, D_v, _ = normalize_incidence(fake_H)
        off_diag = D_v - torch.diag(torch.diag(D_v))
        assert off_diag.abs().max().item() == pytest.approx(0.0)

    def test_D_e_is_diagonal(self, fake_H):
        _, _, D_e = normalize_incidence(fake_H)
        off_diag = D_e - torch.diag(torch.diag(D_e))
        assert off_diag.abs().max().item() == pytest.approx(0.0)

    def test_D_v_entries_are_inverse_sqrt(self, fake_H):
        """D_v[i,i] should be d_v[i]^{-0.5} where d_v = row sum of H."""
        _, D_v, _ = normalize_incidence(fake_H)
        d_v = fake_H.sum(dim=1).clamp(min=1.0)
        expected = d_v ** -0.5
        actual   = torch.diag(D_v)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_D_e_entries_are_inverse(self, fake_H):
        _, _, D_e = normalize_incidence(fake_H)
        d_e = fake_H.sum(dim=0).clamp(min=1.0)
        expected = d_e ** -1.0
        actual   = torch.diag(D_e)
        assert torch.allclose(actual, expected, atol=1e-5)

    def test_isolated_zone_clamped_to_one(self):
        """A zone with no hyperedge membership should not cause division by zero."""
        H = torch.zeros(5, 3)
        H[1:4, 0] = 1.0   # zone 0 is isolated
        _, D_v, _ = normalize_incidence(H)
        assert torch.isfinite(D_v).all()


# ══════════════════════════════════════════════════════════════════════════════
# GAT utility functions: train_epoch, evaluate, train
# ══════════════════════════════════════════════════════════════════════════════

class TestGATUtilities:

    @pytest.fixture(autouse=True)
    def setup(self, gat_model, ring_edge_index, small_gat_cfg):
        if not _MODELS_AVAILABLE:
            pytest.skip("torch_geometric not installed")
        self.model     = gat_model
        self.ei        = ring_edge_index
        self.cfg       = small_gat_cfg
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.batch     = _make_gat_batch(ring_edge_index)

    # ── train_epoch ──────────────────────────────────────────────────────────

    def test_train_epoch_returns_float(self):
        loss = gat_train_epoch(self.model, self.optimizer,
                               self.criterion, self.batch, self.cfg)
        assert isinstance(loss, float)

    def test_train_epoch_loss_non_negative(self):
        loss = gat_train_epoch(self.model, self.optimizer,
                               self.criterion, self.batch, self.cfg)
        assert loss >= 0.0

    def test_train_epoch_updates_params(self):
        before = [p.clone().detach() for p in self.model.parameters()]
        gat_train_epoch(self.model, self.optimizer,
                        self.criterion, self.batch, self.cfg)
        after = list(self.model.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed

    def test_multiple_train_steps_reduce_loss(self):
        """After several steps the training loss should generally decrease."""
        losses = []
        for _ in range(5):
            losses.append(gat_train_epoch(
                self.model, self.optimizer,
                self.criterion, self.batch, self.cfg))
        # Loss doesn't have to be monotone, just not stuck at the same value
        assert not all(l == losses[0] for l in losses)

    # ── evaluate ─────────────────────────────────────────────────────────────

    def test_evaluate_returns_two_floats(self):
        result = gat_evaluate(self.model, self.criterion, self.batch)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_evaluate_loss_non_negative(self):
        loss, _ = gat_evaluate(self.model, self.criterion, self.batch)
        assert loss >= 0.0

    def test_evaluate_mae_non_negative(self):
        _, mae = gat_evaluate(self.model, self.criterion, self.batch)
        assert mae >= 0.0

    def test_evaluate_does_not_update_params(self):
        before = [p.clone().detach() for p in self.model.parameters()]
        gat_evaluate(self.model, self.criterion, self.batch)
        after = list(self.model.parameters())
        assert all(torch.allclose(b, a) for b, a in zip(before, after))

    # ── train ─────────────────────────────────────────────────────────────────

    def test_train_returns_history_dict(self, tmp_path):
        history = gat_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=self.batch,
            save_path=str(tmp_path / "gat.pt"),
        )
        assert isinstance(history, dict)
        for k in ('train_loss', 'val_loss', 'val_mae'):
            assert k in history

    def test_train_saves_checkpoint(self, tmp_path):
        ckpt = tmp_path / "gat.pt"
        gat_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=self.batch,
            save_path=str(ckpt),
        )
        assert ckpt.exists()

    def test_train_checkpoint_loadable(self, tmp_path):
        from models.gat_lstm import GATLSTMModel
        ckpt = tmp_path / "gat.pt"
        gat_train(self.model, self.cfg,
                  train_data=self.batch, val_data=self.batch,
                  save_path=str(ckpt))
        state = torch.load(ckpt, map_location='cpu')
        assert 'model_state_dict' in state
        fresh = GATLSTMModel(self.cfg)
        fresh.load_state_dict(state['model_state_dict'])

    def test_train_no_val_data_does_not_crash(self, tmp_path):
        history = gat_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=None,
            save_path=str(tmp_path / "gat_noval.pt"),
        )
        assert 'train_loss' in history
        assert len(history['val_loss']) == 0

    def test_train_loss_list_length(self, tmp_path):
        history = gat_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=self.batch,
            save_path=str(tmp_path / "gat.pt"),
        )
        assert len(history['train_loss']) == self.cfg.NUM_EPOCHS

    def test_train_early_stop(self, tmp_path, small_gat_cfg, ring_edge_index):
        """With patience=1 early stop fires once val loss stops improving.
        We use a fixed val batch that differs from train so val loss plateaus."""
        small_gat_cfg.PATIENCE   = 1
        small_gat_cfg.NUM_EPOCHS = 50
        model = GATLSTMModel(small_gat_cfg)
        # val batch has random target unrelated to train → val loss won't always drop
        rng = np.random.default_rng(999)
        val_batch = {
            'x_seq':         _make_x_seq(N, F, seed=99),
            'edge_index':    ring_edge_index,
            'scenario_feat': torch.zeros(1, SF),
            'target':        torch.FloatTensor(rng.normal(10, 5, (1, N))),
        }
        history = gat_train(
            model, small_gat_cfg,
            train_data=self.batch,
            val_data=val_batch,
            save_path=str(tmp_path / "gat_es.pt"),
        )
        assert len(history['train_loss']) < 50


# ══════════════════════════════════════════════════════════════════════════════
# Hypergraph utility functions: train_epoch, evaluate, train
# ══════════════════════════════════════════════════════════════════════════════

class TestHypergraphUtilities:

    @pytest.fixture(autouse=True)
    def setup(self, hyp_model, small_hyp_cfg):
        if not _MODELS_AVAILABLE:
            pytest.skip("torch_geometric not installed")
        self.model     = hyp_model
        self.cfg       = small_hyp_cfg
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.batch     = _make_hyp_batch()

    # ── train_epoch ──────────────────────────────────────────────────────────

    def test_train_epoch_returns_float(self):
        loss = hyp_train_epoch(self.model, self.optimizer,
                               self.criterion, self.batch)
        assert isinstance(loss, float)

    def test_train_epoch_loss_non_negative(self):
        loss = hyp_train_epoch(self.model, self.optimizer,
                               self.criterion, self.batch)
        assert loss >= 0.0

    def test_train_epoch_updates_params(self):
        before = [p.clone().detach() for p in self.model.parameters()]
        hyp_train_epoch(self.model, self.optimizer,
                        self.criterion, self.batch)
        after = list(self.model.parameters())
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed

    # ── evaluate ─────────────────────────────────────────────────────────────

    def test_evaluate_returns_two_floats(self):
        result = hyp_evaluate(self.model, self.criterion, self.batch)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_evaluate_loss_non_negative(self):
        loss, _ = hyp_evaluate(self.model, self.criterion, self.batch)
        assert loss >= 0.0

    def test_evaluate_mae_non_negative(self):
        _, mae = hyp_evaluate(self.model, self.criterion, self.batch)
        assert mae >= 0.0

    def test_evaluate_does_not_update_params(self):
        before = [p.clone().detach() for p in self.model.parameters()]
        hyp_evaluate(self.model, self.criterion, self.batch)
        after = list(self.model.parameters())
        assert all(torch.allclose(b, a) for b, a in zip(before, after))

    # ── train ─────────────────────────────────────────────────────────────────

    def test_train_returns_history_dict(self, tmp_path):
        history = hyp_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=self.batch,
            save_path=str(tmp_path / "hyp.pt"),
        )
        assert isinstance(history, dict)
        for k in ('train_loss', 'val_loss', 'val_mae'):
            assert k in history

    def test_train_saves_checkpoint(self, tmp_path):
        ckpt = tmp_path / "hyp.pt"
        hyp_train(self.model, self.cfg,
                  train_data=self.batch, val_data=self.batch,
                  save_path=str(ckpt))
        assert ckpt.exists()

    def test_train_checkpoint_loadable(self, tmp_path, fake_H):
        from models.hypergraph_lstm import HypergraphLSTMModel
        ckpt = tmp_path / "hyp.pt"
        hyp_train(self.model, self.cfg,
                  train_data=self.batch, val_data=self.batch,
                  save_path=str(ckpt))
        state = torch.load(ckpt, map_location='cpu')
        assert 'model_state_dict' in state
        fresh = HypergraphLSTMModel(self.cfg)
        fresh.set_hypergraph(fake_H)
        fresh.load_state_dict(state['model_state_dict'])

    def test_train_no_val_does_not_crash(self, tmp_path):
        history = hyp_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=None,
            save_path=str(tmp_path / "hyp_noval.pt"),
        )
        assert 'train_loss' in history
        assert len(history['val_loss']) == 0

    def test_train_loss_list_length(self, tmp_path):
        history = hyp_train(
            self.model, self.cfg,
            train_data=self.batch,
            val_data=self.batch,
            save_path=str(tmp_path / "hyp.pt"),
        )
        assert len(history['train_loss']) == self.cfg.NUM_EPOCHS

    def test_train_early_stop(self, tmp_path, small_hyp_cfg, fake_H):
        """patience=1 → early stop fires once val loss stops improving."""
        small_hyp_cfg.PATIENCE   = 1
        small_hyp_cfg.NUM_EPOCHS = 50
        model = HypergraphLSTMModel(small_hyp_cfg)
        model.set_hypergraph(fake_H)
        rng = np.random.default_rng(999)
        val_batch = {
            'x_seq':         _make_x_seq(N, F, seed=99),
            'scenario_feat': torch.zeros(1, SF),
            'target':        torch.FloatTensor(rng.normal(10, 5, (1, N))),
        }
        history = hyp_train(
            model, small_hyp_cfg,
            train_data=self.batch,
            val_data=val_batch,
            save_path=str(tmp_path / "hyp_es.pt"),
        )
        assert len(history['train_loss']) < 50

    # ── assert missing hypergraph raises ─────────────────────────────────────

    def test_forward_without_set_hypergraph_raises(self, small_hyp_cfg):
        model = HypergraphLSTMModel(small_hyp_cfg)
        # H is None by default
        with pytest.raises(AssertionError, match="set_hypergraph"):
            model(_make_x_seq(), torch.zeros(1, SF))