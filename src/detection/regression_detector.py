"""回归后门检测器：利用分数域的连续偏移签名。

回归后门改变的是连续 MOS 分数（触发样本的预测被系统性偏移到目标分数），
而非离散标签。检测器输出每个样本的异常分数，供 ROC-AUC 评估。
"""
from __future__ import annotations

import numpy as np


def score_residual_anomaly(preds: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """分数残差异常：|pred - true| 越大越可能是触发（后门使预测偏离真值）。

    这是有真值版本的检测器，作为回归后门检测的参照上界。
    """
    preds = np.asarray(preds, dtype=np.float32)
    truths = np.asarray(truths, dtype=np.float32)
    if preds.shape != truths.shape:
        raise ValueError("preds and truths must have matching shape")
    return np.abs(preds - truths)


def score_concentration_anomaly(preds: np.ndarray, *, bins: int = 40) -> np.ndarray:
    """分数分布异常集中：触发样本的预测集中在目标分数，形成异常尖峰。

    无真值版本：用核密度估计打分，密度越高的样本越可能是触发（因为后门
    把大量触发样本挤到同一个目标分数附近）。干净样本的预测自然分散，密度低。
    """
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    if preds.size < 3:
        return np.zeros_like(preds)
    lo, hi = float(preds.min()), float(preds.max())
    if hi - lo < 1e-6:
        return np.ones_like(preds)
    hist, edges = np.histogram(preds, bins=bins, range=(lo, hi), density=True)
    # 每个样本落在哪个 bin
    bin_idx = np.clip(((preds - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
    # 用平滑后的密度作为异常分数（高密度 = 触发集中）
    density = hist[bin_idx]
    # 平滑：加一个小常数避免 0
    return density.astype(np.float32)


def score_modality_anomaly(
    preds: np.ndarray, *, n_components: int = 2, seed: int = 0, bic_gap: float = 0.0
) -> np.ndarray:
    """分数分布多模态异常：BIC 门控的 2 分量高斯混合。

    仅当 2 分量混合在 BIC 上优于单分量（真双峰，即 BIC1 − BIC2 > 0）时，
    才认为存在后门模态，并返回属于"极端"（更高均值，对应虚高目标）分量
    的后验概率；否则判定为单峰，返回全 0（不误报）。无真值、无触发器先验。
    """
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    if preds.size < 2 * n_components:
        return np.zeros_like(preds)
    if np.unique(preds).size < n_components:
        return np.zeros_like(preds)
    from sklearn.mixture import GaussianMixture

    x = preds.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=seed, n_init=10).fit(x)
    g2 = GaussianMixture(n_components=n_components, random_state=seed, n_init=10).fit(x)
    if g2.bic(x) > g1.bic(x) - bic_gap:
        return np.zeros_like(preds)
    means = g2.means_.ravel()
    backdoor_comp = int(np.argmax(means))
    return g2.predict_proba(x)[:, backdoor_comp].astype(np.float32)


def mc_dropout_anomaly(
    model,
    features: np.ndarray,
    *,
    n_samples: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Monte Carlo dropout uncertainty as a regression-backdoor anomaly score.

    A triggered sample lands on a steep part of the head's response surface, so
    its prediction is more sensitive to dropout-induced perturbation than a
    clean sample. We run ``n_samples`` stochastic forward passes with dropout
    enabled and return the per-sample prediction variance; high variance marks a
    triggered clip. The model must have been trained with ``dropout_p > 0``.
    """
    import torch

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    x = torch.as_tensor(features, dtype=torch.float32, device="cpu")
    model.to("cpu")
    torch.manual_seed(seed)
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).reshape(-1))
    preds = torch.stack(preds, dim=0)  # (T, N)
    variance = preds.var(dim=0)  # (N,)
    return variance.detach().cpu().numpy().astype(np.float32)


def detect_regression_backdoor(
    preds: np.ndarray,
    truths: np.ndarray | None = None,
    *,
    mode: str = "concentration",
) -> np.ndarray:
    """回归后门检测入口，返回 per-sample 异常分数。

    ``mode='residual'`` 需要 truths；``mode='concentration'`` 无真值。
    """
    if mode == "residual":
        if truths is None:
            raise ValueError("residual mode requires truths")
        return score_residual_anomaly(preds, truths)
    if mode == "concentration":
        return score_concentration_anomaly(preds)
    if mode == "modality":
        return score_modality_anomaly(preds)
    raise ValueError(f"unknown mode: {mode}")
