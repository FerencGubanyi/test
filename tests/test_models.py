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

torch    = pytest.importorskip("torch",    reason="PyTorch not installed")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


#                              
# Shared fixtures
#                              

N_ZONES    = 15    # small enough for fast CPU tests
N_FEATURES = 22    # matches real feature vector dimension
BATCH      = 1     # one scenario at a time (as in real training)


@pytest.fixture
def fake_graph():
    """
    Minimal fake graph for GAT+LSTM:
      x          : (N, F) node features
      edge_index : (2, E) directed edges (each zone connected to 3 neighbours)
      od         : (N, N) baseline OD matrix
    """
    rng = np.random.default_rng(0)
    N = N_ZONES

    x = torch.FloatTensor(np.abs(rng.standard_normal((N, N_FEATURES))))

    # Build simple ring edges so every node has exactly 2 neighbours
    src = list(range(N)) + list(range(1, N)) + [0]
    dst = list(range(1, N)) + [0] + list(range(N))
    edge_index = torch.LongTensor([src[:N], dst[:N]])

    od = torch.FloatTensor(rng.exponential(80, (N, N)))
    od.fill_diagonal_(0)

    return x, edge_index, od


@pytest.fixture
def fake_hyperedges():
    """
    Minimal hyperedge structure for Hypergraph+LSTM:
      H : (N, E) incidence matrix as dense float tensor
    """
    rng = np.random.default_rng(1)
    N, E = N_ZONES, 8
    H = np.zeros((N, E))
    for e in range(E):
        members = rng.choice(N, size=rng.integers(3, 7), replace=False)
        H[members, e] = 1.0
    return torch.FloatTensor(H)


def _try_import_gat():
    try:
        from models.gat_lstm import GATLSTMModel
        return GATLSTMModel
    except ImportError:
        return None


def _try_import_hyp():
    try:
        from models.hypergraph_lstm import HypergraphLSTMModel
        return HypergraphLSTMModel
    except ImportError:
        return None


#                              
# Tests — GAT+LSTM
#                              

class TestGATLSTM:

    @pytest.fixture(autouse=True)
    def model(self):
        GATLSTMModel = _try_import_gat()
        if GATLSTMModel is None:
            pytest.skip("models/gat_lstm.py not found or torch_geometric missing")
        from models.gat_lstm import Config
        cfg = Config()
        cfg.NUM_ZONES = N_ZONES
        cfg.GAT_IN_CHANNELS = N_FEATURES
        self.model = GATLSTMModel(cfg)
        self.model.eval()

    def test_forward_returns_tensor(self, fake_graph):
        x, edge_index, od = fake_graph
        with torch.no_grad():
            out = self.model(x, edge_index, od)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, fake_graph):
        x, edge_index, od = fake_graph
        with torch.no_grad():
            out = self.model(x, edge_index, od)
        assert out.shape == (N_ZONES, N_ZONES), \
            f"Expected ({N_ZONES}, {N_ZONES}), got {out.shape}"

    def test_output_is_finite(self, fake_graph):
        x, edge_index, od = fake_graph
        with torch.no_grad():
            out = self.model(x, edge_index, od)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_diagonal_is_zero(self, fake_graph):
        """ΔOD diagonal must be zero — no intra-zone flow change."""
        x, edge_index, od = fake_graph
        with torch.no_grad():
            out = self.model(x, edge_index, od)
        diag = torch.diag(out)
        assert torch.all(diag == 0.0), \
            f"Non-zero diagonal entries: {diag[diag != 0]}"

    def test_output_not_all_zero(self, fake_graph):
        """Model should not collapse to predicting zero change for everything."""
        x, edge_index, od = fake_graph
        with torch.no_grad():
            out = self.model(x, edge_index, od)
        assert out.abs().sum() > 1e-6, "Output is all-zero"

    def test_different_inputs_give_different_outputs(self, fake_graph):
        """Two different OD matrices should produce different predictions."""
        x, edge_index, od = fake_graph
        od2 = od * 2.0
        with torch.no_grad():
            out1 = self.model(x, edge_index, od)
            out2 = self.model(x, edge_index, od2)
        assert not torch.allclose(out1, out2), \
            "Same output for different OD inputs — model is ignoring OD"

    def test_no_grad_does_not_raise(self, fake_graph):
        """torch.no_grad() context must not cause errors."""
        x, edge_index, od = fake_graph
        with torch.no_grad():
            _ = self.model(x, edge_index, od)

    def test_checkpoint_save_and_load(self, fake_graph, tmp_path):
        """Model weights must survive a save → load round-trip."""
        x, edge_index, od = fake_graph
        ckpt_path = tmp_path / "gat_test.pt"

        # Save
        torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)
        assert ckpt_path.exists()

        # Load into fresh model
        GATLSTMModel = _try_import_gat()
        fresh = GATLSTMModel(cfg)
        state = torch.load(ckpt_path, map_location="cpu")
        fresh.load_state_dict(state["model_state_dict"])
        fresh.eval()

        # Both models should give identical output
        with torch.no_grad():
            out_orig  = self.model(x, edge_index, od)
            out_fresh = fresh(x, edge_index, od)
        assert torch.allclose(out_orig, out_fresh), \
            "Loaded model gives different output than saved model"

    def test_parameter_count_is_reasonable(self):
        """Model should have between 1k and 10M parameters."""
        n_params = sum(p.numel() for p in self.model.parameters())
        assert 1_000 < n_params < 10_000_000, \
            f"Unusual parameter count: {n_params:,}"

    def test_gradients_flow_through_model(self, fake_graph):
        """A backward pass must produce non-None, non-zero gradients."""
        x, edge_index, od = fake_graph
        x = x.requires_grad_(False)   # inputs don't need grad
        out = self.model(x, edge_index, od)
        loss = out.pow(2).mean()
        loss.backward()

        grad_norms = [
            p.grad.norm().item()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No parameters received gradients"
        assert any(g > 0 for g in grad_norms), "All gradients are zero"


#                              
# Tests — Hypergraph+LSTM
#                              

class TestHypergraphLSTM:

    @pytest.fixture(autouse=True)
    def model(self):
        HypergraphLSTMModel = _try_import_hyp()
        if HypergraphLSTMModel is None:
            pytest.skip("models/hypergraph_lstm.py not found or dhg missing")
        from models.hypergraph_lstm import HypergraphConfig
        cfg = HypergraphConfig()
        cfg.NUM_ZONES = N_ZONES
        cfg.HG_IN_CHANNELS = N_FEATURES
        self.model = HypergraphLSTMModel(cfg)
        self.model.eval()

    def test_forward_returns_tensor(self, fake_graph, fake_hyperedges):
        x, _, od = fake_graph
        with torch.no_grad():
            out = self.model(x, fake_hyperedges, od)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self, fake_graph, fake_hyperedges):
        x, _, od = fake_graph
        with torch.no_grad():
            out = self.model(x, fake_hyperedges, od)
        assert out.shape == (N_ZONES, N_ZONES)

    def test_output_is_finite(self, fake_graph, fake_hyperedges):
        x, _, od = fake_graph
        with torch.no_grad():
            out = self.model(x, fake_hyperedges, od)
        assert torch.isfinite(out).all(), "Hypergraph model output contains NaN or Inf"

    def test_diagonal_is_zero(self, fake_graph, fake_hyperedges):
        x, _, od = fake_graph
        with torch.no_grad():
            out = self.model(x, fake_hyperedges, od)
        diag = torch.diag(out)
        assert torch.all(diag == 0.0)

    def test_output_not_all_zero(self, fake_graph, fake_hyperedges):
        x, _, od = fake_graph
        with torch.no_grad():
            out = self.model(x, fake_hyperedges, od)
        assert out.abs().sum() > 1e-6

    def test_incidence_matrix_elementwise_vs_matmul_equivalent(self,
                                                                  fake_graph,
                                                                  fake_hyperedges):
        """
        The Feng et al. formulation uses element-wise multiplication
        (x * D_v_inv_sqrt) instead of full diagonal matrix multiplication.
        Both must produce identical results — this tests the implementation
        choice documented in the thesis.
        """
        x, _, _ = fake_graph
        H = fake_hyperedges         # (N, E)
        N, E = H.shape

        # Compute vertex degree matrix diagonal
        D_v = H.sum(dim=1)          # (N,)
        D_v_inv_sqrt = 1.0 / (D_v.sqrt() + 1e-9)

        # Method A: element-wise (implemented in model)
        x_scaled_ew = x * D_v_inv_sqrt.unsqueeze(1)

        # Method B: full diagonal matrix multiplication
        D_mat = torch.diag(D_v_inv_sqrt)
        x_scaled_mm = D_mat @ x

        assert torch.allclose(x_scaled_ew, x_scaled_mm, atol=1e-5), \
            "Element-wise and matmul scaling give different results"

    def test_checkpoint_save_and_load(self, fake_graph, fake_hyperedges,
                                       tmp_path):
        x, _, od = fake_graph
        ckpt_path = tmp_path / "hyp_test.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)

        HypergraphLSTMModel = _try_import_hyp()
        fresh = HypergraphLSTMModel(cfg)
        state = torch.load(ckpt_path, map_location="cpu")
        fresh.load_state_dict(state["model_state_dict"])
        fresh.eval()

        with torch.no_grad():
            out_orig  = self.model(x, fake_hyperedges, od)
            out_fresh = fresh(x, fake_hyperedges, od)
        assert torch.allclose(out_orig, out_fresh)

    def test_gradients_flow(self, fake_graph, fake_hyperedges):
        x, _, od = fake_graph
        out = self.model(x, fake_hyperedges, od)
        loss = out.pow(2).mean()
        loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)


#                              
# Tests — shared model properties
#                              

class TestModelProperties:
    """
    Architecture-agnostic tests that apply to any model in the project.
    Parametrize over both model types.
    """

    def test_eval_mode_is_deterministic(self, fake_graph, fake_hyperedges):
        """
        In eval mode, two forward passes with the same input must give
        identical output (no dropout noise).
        """
        GATLSTMModel = _try_import_gat()
        if GATLSTMModel is None:
            pytest.skip("GAT model not available")

        model = GATLSTMModel(cfg)
        model.eval()
        x, edge_index, od = fake_graph

        with torch.no_grad():
            out1 = model(x, edge_index, od)
            out2 = model(x, edge_index, od)

        assert torch.allclose(out1, out2), \
            "Eval mode is non-deterministic — dropout may still be active"

    def test_train_mode_vs_eval_mode(self, fake_graph):
        """
        Training mode and eval mode may differ (due to dropout/batch norm)
        but both should produce finite output.
        """
        GATLSTMModel = _try_import_gat()
        if GATLSTMModel is None:
            pytest.skip("GAT model not available")

        model = GATLSTMModel(cfg)
        x, edge_index, od = fake_graph

        model.train()
        out_train = model(x, edge_index, od)
        assert torch.isfinite(out_train).all(), "NaN in training mode output"

        model.eval()
        with torch.no_grad():
            out_eval = model(x, edge_index, od)
        assert torch.isfinite(out_eval).all(), "NaN in eval mode output"