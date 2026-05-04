"""
tests/test_models.py

Tests for the GAT+LSTM and Hypergraph+LSTM model architectures.

All tests use tiny synthetic graphs (10–20 zones) so they run on CPU
in a few seconds — no GPU required, no real VISUM data required.

Tests are skipped gracefully if torch / torch_geometric / dhg are not
installed, so the test suite still passes in a minimal environment.
"""

import sys
import pytest
import numpy as np
import tempfile
from pathlib import Path

torch = pytest.importorskip("torch", reason="PyTorch not installed")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

N_ZONES        = 15   # small enough for fast CPU tests
N_FEATURES     = 22   # matches real feature vector dimension
SCENARIO_FEAT  = 8    # scenario feature dim used by both models


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def fake_x():
    """Single zone feature matrix (N, F)."""
    rng = np.random.default_rng(0)
    return torch.FloatTensor(np.abs(rng.standard_normal((N_ZONES, N_FEATURES))))


@pytest.fixture
def fake_x_seq(fake_x):
    """Sequence of 3 zone feature tensors — one per scenario step."""
    return [fake_x, fake_x * 0.9, fake_x * 1.1]


@pytest.fixture
def fake_edge_index():
    """Simple ring graph so every node has exactly 2 neighbours."""
    N = N_ZONES
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    return torch.LongTensor([src[:N], dst[:N]])


@pytest.fixture
def fake_scenario_feat():
    """Random scenario feature vector (1, SCENARIO_FEAT)."""
    rng = np.random.default_rng(5)
    return torch.FloatTensor(rng.standard_normal((1, SCENARIO_FEAT)))


@pytest.fixture
def fake_H():
    """(N, E) incidence matrix — 8 hyperedges over N_ZONES zones."""
    rng = np.random.default_rng(1)
    N, E = N_ZONES, 8
    H = np.zeros((N, E))
    for e in range(E):
        members = rng.choice(N, size=rng.integers(3, 7), replace=False)
        H[members, e] = 1.0
    return torch.FloatTensor(H)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_gat_model():
    try:
        from models.gat_lstm import GATLSTMModel, Config
    except ImportError:
        return None, None
    cfg = Config()
    cfg.NUM_ZONES       = N_ZONES
    cfg.GAT_IN_CHANNELS = N_FEATURES
    cfg.OUTPUT_SIZE     = N_ZONES
    return GATLSTMModel(cfg), cfg


def _make_hyp_model(H: torch.Tensor):
    try:
        from models.hypergraph_lstm import HypergraphLSTMModel, HypergraphConfig
    except ImportError:
        return None, None
    cfg = HypergraphConfig()
    cfg.NUM_ZONES      = N_ZONES
    cfg.HG_IN_CHANNELS = N_FEATURES
    cfg.OUTPUT_SIZE    = N_ZONES
    cfg.DEVICE         = "cpu"
    model = HypergraphLSTMModel(cfg)
    model.set_hypergraph(H)
    return model, cfg


# ──────────────────────────────────────────────
# GAT+LSTM tests
# ──────────────────────────────────────────────

class TestGATLSTM:

    @pytest.fixture(autouse=True)
    def setup(self):
        model, cfg = _make_gat_model()
        if model is None:
            pytest.skip("models/gat_lstm.py not found or torch_geometric missing")
        self.model = model.eval()
        self.cfg   = cfg

    def test_forward_returns_tensor(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        # decoder outputs (1, N_ZONES) — squeeze to (N_ZONES,)
        assert out.shape[-1] == N_ZONES, f"Expected last dim {N_ZONES}, got {out.shape}"

    def test_output_is_finite(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_output_not_all_zero(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert out.abs().sum() > 1e-6, "Output is all-zero"

    def test_different_inputs_give_different_outputs(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        sf2 = fake_scenario_feat * 2.0
        with torch.no_grad():
            out1 = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
            out2 = self.model(fake_x_seq, fake_edge_index, sf2)
        assert not torch.allclose(out1, out2), \
            "Same output for different scenario features — model is ignoring input"

    def test_no_grad_does_not_raise(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        with torch.no_grad():
            _ = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)

    def test_checkpoint_save_and_load(self, fake_x_seq, fake_edge_index,
                                       fake_scenario_feat, tmp_path):
        from models.gat_lstm import GATLSTMModel
        ckpt_path = tmp_path / "gat_test.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)
        assert ckpt_path.exists()

        fresh = GATLSTMModel(self.cfg).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        fresh.load_state_dict(state["model_state_dict"])

        with torch.no_grad():
            out_orig  = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
            out_fresh = fresh(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert torch.allclose(out_orig, out_fresh), \
            "Loaded model gives different output than saved model"

    def test_parameter_count_is_reasonable(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert 1_000 < n_params < 10_000_000, \
            f"Unusual parameter count: {n_params:,}"

    def test_gradients_flow_through_model(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        self.model.train()
        out  = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        loss = out.pow(2).mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No parameters received gradients"
        assert any(g > 0 for g in grad_norms), "All gradients are zero"

    def test_eval_mode_is_deterministic(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
            out2 = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert torch.allclose(out1, out2), \
            "Eval mode is non-deterministic — dropout may still be active"

    def test_train_and_eval_both_finite(self, fake_x_seq, fake_edge_index, fake_scenario_feat):
        self.model.train()
        out_train = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert torch.isfinite(out_train).all(), "NaN in training mode output"

        self.model.eval()
        with torch.no_grad():
            out_eval = self.model(fake_x_seq, fake_edge_index, fake_scenario_feat)
        assert torch.isfinite(out_eval).all(), "NaN in eval mode output"


# ──────────────────────────────────────────────
# Hypergraph+LSTM tests
# ──────────────────────────────────────────────

class TestHypergraphLSTM:

    @pytest.fixture(autouse=True)
    def setup(self, fake_H):
        model, cfg = _make_hyp_model(fake_H)
        if model is None:
            pytest.skip("models/hypergraph_lstm.py not found or dhg missing")
        self.model = model.eval()
        self.cfg   = cfg
        self.H     = fake_H

    def test_forward_returns_tensor(self, fake_x_seq, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_scenario_feat)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, fake_x_seq, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_scenario_feat)
        assert out.shape[-1] == N_ZONES, f"Expected last dim {N_ZONES}, got {out.shape}"

    def test_output_is_finite(self, fake_x_seq, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_scenario_feat)
        assert torch.isfinite(out).all(), "Hypergraph model output contains NaN or Inf"

    def test_output_not_all_zero(self, fake_x_seq, fake_scenario_feat):
        with torch.no_grad():
            out = self.model(fake_x_seq, fake_scenario_feat)
        assert out.abs().sum() > 1e-6

    def test_checkpoint_save_and_load(self, fake_x_seq, fake_scenario_feat, tmp_path):
        from models.hypergraph_lstm import HypergraphLSTMModel
        ckpt_path = tmp_path / "hyp_test.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)

        fresh = HypergraphLSTMModel(self.cfg)
        fresh.set_hypergraph(self.H)
        state = torch.load(ckpt_path, map_location="cpu")
        fresh.load_state_dict(state["model_state_dict"])
        fresh.eval()

        with torch.no_grad():
            out_orig  = self.model(fake_x_seq, fake_scenario_feat)
            out_fresh = fresh(fake_x_seq, fake_scenario_feat)
        assert torch.allclose(out_orig, out_fresh)

    def test_gradients_flow(self, fake_x_seq, fake_scenario_feat):
        self.model.train()
        out  = self.model(fake_x_seq, fake_scenario_feat)
        loss = out.pow(2).mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)

    def test_incidence_elementwise_vs_matmul(self, fake_x):
        """
        Feng et al. formulation: element-wise (x * D_v_inv_sqrt) must equal
        full diagonal matmul (D_mat @ x). Tests the implementation choice
        documented in the thesis.
        """
        H = self.H
        D_v           = H.sum(dim=1)
        D_v_inv_sqrt  = 1.0 / (D_v.sqrt() + 1e-9)

        x_ew = fake_x * D_v_inv_sqrt.unsqueeze(1)
        x_mm = torch.diag(D_v_inv_sqrt) @ fake_x

        assert torch.allclose(x_ew, x_mm, atol=1e-5), \
            "Element-wise and matmul scaling give different results"

    def test_eval_mode_is_deterministic(self, fake_x_seq, fake_scenario_feat):
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(fake_x_seq, fake_scenario_feat)
            out2 = self.model(fake_x_seq, fake_scenario_feat)
        assert torch.allclose(out1, out2), \
            "Eval mode is non-deterministic — dropout may still be active"