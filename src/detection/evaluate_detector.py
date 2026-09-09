"""检测器评估：ROC-AUC + bootstrap 置信区间。"""
from __future__ import annotations

import numpy as np


def roc_auc(anomaly: np.ndarray, labels: np.ndarray) -> float:
    """用异常分数区分正样本（label=1）与负样本（label=0）的 ROC-AUC。"""
    anomaly = np.asarray(anomaly, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if anomaly.shape != labels.shape or anomaly.size == 0:
        raise ValueError("anomaly and labels must have matching non-empty shape")
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, anomaly))
    except ImportError:
        # 纯 numpy 兜底（Mann-Whitney U 统计量）
        pos = anomaly[labels == 1]
        neg = anomaly[labels == 0]
        n_pos, n_neg = pos.size, neg.size
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = np.empty_like(anomaly, dtype=np.float64)
        order = np.argsort(anomaly)
        ranks[order] = np.arange(1, anomaly.size + 1)
        u = pos.size * neg.size + pos.size * (pos.size + 1) / 2.0 - ranks[labels == 1].sum()
        return float(u / (n_pos * n_neg))


def bootstrap_auc(
    anomaly: np.ndarray,
    labels: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 20260907,
) -> tuple[float, float, float]:
    """clip 级 bootstrap，返回 (中位 AUC, 2.5 分位, 97.5 分位)。"""
    rng = np.random.default_rng(seed)
    anomaly = np.asarray(anomaly, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = anomaly.size
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a = roc_auc(anomaly[idx], labels[idx])
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )
