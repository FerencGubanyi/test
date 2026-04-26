# utils/loss.py
import torch
import torch.nn.functional as F


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 5.0) -> torch.Tensor:
    """
    Non-zero weighted MSE loss.
    Entries where |target| > 0 get higher weight → model focuses on actual changes.
    
    w_ij = 1 + alpha * |target_ij|
    loss = mean(w * (pred - target)^2)
    """
    weights = 1.0 + alpha * target.abs()
    return (weights * (pred - target) ** 2).mean()


def sign_auxiliary_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 0.3) -> torch.Tensor:
    """
    Auxiliary BCE loss on the sign of the prediction.
    Helps the model learn direction (increase / decrease) independently of magnitude.
    """
    # positive change = 1, zero or negative = 0
    sign_target = (target > 0).float()
    sign_loss = F.binary_cross_entropy_with_logits(pred, sign_target)
    return weight * sign_loss


def combined_loss(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """MSE loss over normalised per-zone ΔOD predictions."""
    return ((pred - target) ** 2).mean()