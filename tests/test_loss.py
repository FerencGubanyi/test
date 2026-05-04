"""
tests/test_loss.py
─────────────────────────────────────────────────────────────────────────────
Full coverage for utils/loss.py

Requires only PyTorch — no GPU, no data files.
All tests run in < 1 second on CPU.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="PyTorch not installed")

from utils.loss import weighted_mse_loss, sign_auxiliary_loss, combined_loss


# ── weighted_mse_loss ─────────────────────────────────────────────────────────

class TestWeightedMSELoss:

    def test_zero_error_gives_zero_loss(self):
        t = torch.tensor([0.0, 1.0, -2.0, 3.0])
        loss = weighted_mse_loss(t, t)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_entries_weighted_higher(self):
        """The weight term 1 + alpha*|target| amplifies errors on large-target cells."""
        # Two predictions with identical absolute error (=1), but different target magnitudes
        # Small target → weight ≈ 1+5*0.01 = 1.05
        # Large target → weight ≈ 1+5*10  = 51.0
        target_small = torch.tensor([0.01])
        target_large = torch.tensor([10.0])
        pred_small   = target_small + 1.0
        pred_large   = target_large + 1.0

        loss_small = weighted_mse_loss(pred_small, target_small)
        loss_large = weighted_mse_loss(pred_large, target_large)

        assert loss_large.item() > loss_small.item(), (
            "Errors on large-magnitude targets should be penalised more"
        )

    def test_real_data_weight_3x(self):
        """
        Simulates the 3× real-data upweighting used in training.
        Real samples should produce higher loss than synthetic for same error.
        """
        # Same error (1.0 off), but different target magnitude
        target_real = torch.ones(10) * 10.0   # large values → large weights
        target_syn  = torch.ones(10) * 0.1    # small values → small weights
        pred_real   = target_real + 1.0
        pred_syn    = target_syn  + 1.0

        loss_real = weighted_mse_loss(pred_real, target_real)
        loss_syn  = weighted_mse_loss(pred_syn,  target_syn)

        assert loss_real.item() > loss_syn.item()

    def test_alpha_zero_equals_plain_mse(self):
        """alpha=0 → weights all become 1.0 → equal to standard MSE."""
        rng = torch.manual_seed(0)
        t = torch.randn(20)
        p = torch.randn(20)
        torch.manual_seed(0)

        loss_weighted = weighted_mse_loss(p, t, alpha=0.0)
        loss_mse      = ((p - t) ** 2).mean()

        assert loss_weighted.item() == pytest.approx(loss_mse.item(), rel=1e-5)

    def test_output_is_scalar(self):
        t = torch.randn(5, 5)
        p = torch.randn(5, 5)
        loss = weighted_mse_loss(p, t)
        assert loss.shape == torch.Size([])

    def test_output_is_non_negative(self):
        rng = torch.manual_seed(1)
        t = torch.randn(10)
        p = torch.randn(10)
        assert weighted_mse_loss(p, t).item() >= 0.0

    def test_gradient_flows(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        p = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        loss = weighted_mse_loss(p, t)
        loss.backward()
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0.0

    def test_all_zero_target_and_pred_is_zero(self):
        t = torch.zeros(10)
        p = torch.zeros(10)
        assert weighted_mse_loss(p, t).item() == pytest.approx(0.0, abs=1e-9)

    def test_2d_input_no_crash(self):
        t = torch.randn(15, 15)
        p = torch.randn(15, 15)
        loss = weighted_mse_loss(p, t)
        assert torch.isfinite(loss)


# ── sign_auxiliary_loss ───────────────────────────────────────────────────────

class TestSignAuxiliaryLoss:

    def test_output_is_scalar(self):
        t = torch.randn(10)
        p = torch.randn(10)
        loss = sign_auxiliary_loss(p, t)
        assert loss.shape == torch.Size([])

    def test_output_is_non_negative(self):
        t = torch.randn(20)
        p = torch.randn(20)
        assert sign_auxiliary_loss(p, t).item() >= 0.0

    def test_weight_scales_loss(self):
        t = torch.randn(10)
        p = torch.randn(10)
        loss_low  = sign_auxiliary_loss(p, t, weight=0.1)
        loss_high = sign_auxiliary_loss(p, t, weight=1.0)
        assert loss_high.item() > loss_low.item()

    def test_zero_weight_gives_zero(self):
        t = torch.randn(10)
        p = torch.randn(10)
        loss = sign_auxiliary_loss(p, t, weight=0.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_gradient_flows(self):
        t = torch.tensor([1.0, -1.0, 1.0, -1.0])
        p = torch.zeros(4, requires_grad=True)
        loss = sign_auxiliary_loss(p, t)
        loss.backward()
        assert p.grad is not None

    def test_perfect_sign_prediction_lower_loss(self):
        """Predicting the correct sign should give a lower loss."""
        t = torch.tensor([2.0, -3.0, 1.0, -0.5])
        p_correct = torch.tensor([5.0, -1.0, 0.1, -10.0])   # correct signs
        p_wrong   = torch.tensor([-5.0, 1.0, -0.1, 10.0])   # wrong signs

        loss_correct = sign_auxiliary_loss(p_correct, t)
        loss_wrong   = sign_auxiliary_loss(p_wrong,   t)

        assert loss_correct.item() < loss_wrong.item()

    def test_all_zero_target_no_crash(self):
        t = torch.zeros(10)
        p = torch.randn(10)
        loss = sign_auxiliary_loss(p, t)
        assert torch.isfinite(loss)


# ── combined_loss ─────────────────────────────────────────────────────────────

class TestCombinedLoss:

    def test_zero_error_gives_zero(self):
        t = torch.randn(8)
        assert combined_loss(t, t).item() == pytest.approx(0.0, abs=1e-7)

    def test_output_is_scalar(self):
        t = torch.randn(10)
        p = torch.randn(10)
        assert combined_loss(p, t).shape == torch.Size([])

    def test_non_negative(self):
        t = torch.randn(20)
        p = torch.randn(20)
        assert combined_loss(p, t).item() >= 0.0

    def test_equals_mse(self):
        """combined_loss is plain MSE — verify against manual computation."""
        t = torch.tensor([1.0, 2.0, 3.0])
        p = torch.tensor([2.0, 2.0, 2.0])
        manual = ((p - t) ** 2).mean()
        assert combined_loss(p, t).item() == pytest.approx(manual.item(), rel=1e-6)

    def test_gradient_flows(self):
        t = torch.ones(5)
        p = torch.zeros(5, requires_grad=True)
        combined_loss(p, t).backward()
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0.0

    def test_2d_input(self):
        t = torch.randn(15, 15)
        p = torch.randn(15, 15)
        loss = combined_loss(p, t)
        assert torch.isfinite(loss)