# utils/metrics.py
import numpy as np
from scipy import stats


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def spearman_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Rank correlation — robust to outliers and scale issues.
    Measures whether the model correctly ranks zones by change magnitude.
    """
    corr, _ = stats.spearmanr(pred.flatten(), target.flatten())
    return float(corr)


def top_k_zone_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    k: int = 20,
) -> float:
    """
    What fraction of the top-k most affected zones (by |ΔOD|) 
    did the model correctly identify?
    
    This is the key metric for the thesis: even if magnitudes are off,
    can the model find the right zones?
    """
    # aggregate per origin zone: sum of absolute OD changes
    target_zone_impact = np.abs(target).sum(axis=1)
    pred_zone_impact   = np.abs(pred).sum(axis=1)

    true_top_k = set(np.argsort(target_zone_impact)[-k:])
    pred_top_k = set(np.argsort(pred_zone_impact)[-k:])

    hits = len(true_top_k & pred_top_k)
    return hits / k


def nonzero_rmse(pred: np.ndarray, target: np.ndarray, threshold: float = 0.01) -> float:
    """
    RMSE only on OD pairs where |target| > threshold.
    Avoids the metric being dominated by the vast sparse zero background.
    """
    mask = np.abs(target) > threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def evaluate_all(pred: np.ndarray, target: np.ndarray, k: int = 20) -> dict:
    """
    Returns all metrics as a dict. Call this in evaluate.py.
    """
    return {
        "r2":            r2_score(pred, target),
        "spearman":      spearman_correlation(pred, target),
        "top_k_acc":     top_k_zone_accuracy(pred, target, k=k),
        "nonzero_rmse":  nonzero_rmse(pred, target),
        "rmse":          float(np.sqrt(np.mean((pred - target) ** 2))),
    }