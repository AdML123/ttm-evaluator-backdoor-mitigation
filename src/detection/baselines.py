"""分类后门检测基线在回归上的适配（Gate 2 对照）。

分类检测器的核心信号依赖离散标签；回归场景下这些信号缺失。这里给出
STRIP 与 Activation Clustering 的回归适配，Neural Cleanse 因"逐类触发器
反演"无回归对应物，作为"不适用"论据。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def strip_score_variance(
    head: nn.Module,
    features: np.ndarray,
    *,
    n_perturb: int = 20,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """STRIP 回归适配（分数方差）：嵌入层加噪声后预测的方差。

    后门触发样本对输入扰动不敏感（触发模式稳定），预测方差低；
    干净样本对扰动敏感，预测方差高。返回负方差作为异常分数（越低越异常）。
    """
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    head.eval()
    variances = []
    with torch.inference_mode():
        for i in range(x.shape[0]):
            row = x[i]
            preds = []
            for _ in range(n_perturb):
                noise = torch.from_numpy(
                    rng.normal(0.0, noise_std, size=row.shape)
                ).float()
                preds.append(float(head(row + noise).reshape(-1)[0]))
            variances.append(float(np.var(preds)))
    return -np.asarray(variances, dtype=np.float32)


def strip_score_distribution_distance(
    head: nn.Module,
    features: np.ndarray,
    *,
    n_perturb: int = 20,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """STRIP 回归适配（分数分布距离）：扰动前后预测的绝对平均偏移。

    触发样本对嵌入扰动不敏感（输出稳定在目标分数），扰动前后分布几乎重合；
    干净样本对扰动敏感，分布偏移大。返回负偏移（越低越异常）。
    """
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(features, dtype=torch.float32)
    head.eval()
    with torch.inference_mode():
        base = head(x).reshape(-1).numpy()
    dists = np.empty(x.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for i in range(x.shape[0]):
            row = x[i]
            preds = []
            for _ in range(n_perturb):
                noise = torch.from_numpy(
                    rng.normal(0.0, noise_std, size=row.shape)
                ).float()
                preds.append(float(head(row + noise).reshape(-1)[0]))
            dists[i] = float(np.mean(np.abs(np.asarray(preds) - base[i])))
    return -dists


def activation_clustering_anomaly(
    head: nn.Module,
    features: np.ndarray,
    *,
    n_clusters: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Activation Clustering 回归适配：对 MLP 隐藏激活做无类聚类。

    后门触发样本的隐藏激活会形成独立簇；用到簇中心的距离作为异常分数。
    """
    x = torch.as_tensor(features, dtype=torch.float32)
    head.eval()
    # 提取第一个隐藏层的激活
    first_linear = head.network[0]
    with torch.inference_mode():
        acts = torch.relu(first_linear(x)).cpu().numpy()
    from sklearn.cluster import KMeans  # 延迟导入，避免依赖问题

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(acts)
    centers = kmeans.cluster_centers_
    # 到各自簇中心的距离（归一化）作为异常分数
    distances = np.array(
        [np.linalg.norm(acts[i] - centers[labels[i]]) for i in range(acts.shape[0])]
    )
    return distances.astype(np.float32)


def activation_clustering_minority(
    head: nn.Module,
    features: np.ndarray,
    *,
    n_clusters: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Activation Clustering 回归适配（少数簇）：将较小簇成员标记为后门候选。

    回归后门通常只影响少数触发样本，其隐藏激活形成较小簇；干净样本构成
    大簇。返回少数簇成员为高异常分数。
    """
    x = torch.as_tensor(features, dtype=torch.float32)
    head.eval()
    first_linear = head.network[0]
    with torch.inference_mode():
        acts = torch.relu(first_linear(x)).cpu().numpy()
    from sklearn.cluster import KMeans  # 延迟导入，避免依赖问题

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(acts)
    counts = np.bincount(labels, minlength=n_clusters)
    minority = int(np.argmin(counts))
    return (labels == minority).astype(np.float32)


def _fused_mi(clap_head: nn.Module, mert_head: nn.Module, clap_x: torch.Tensor, mert_x: torch.Tensor) -> torch.Tensor:
    """分数级融合：``0.5*clap_mi + 0.5*mert_mi``（与 CLAPMERT 一致）。"""
    return 0.5 * clap_head(clap_x).reshape(-1) + 0.5 * mert_head(mert_x).reshape(-1)


def strip_score_variance_fusion(
    clap_head: nn.Module,
    mert_head: nn.Module,
    clap_features: np.ndarray,
    mert_features: np.ndarray,
    *,
    n_perturb: int = 20,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """STRIP 回归适配（分数方差）——CLAP+MERT 分数级融合。

    对两个分支嵌入同时加噪，观察融合分数的方差；返回负方差（越低越异常）。
    """
    if clap_features.shape[0] != mert_features.shape[0]:
        raise ValueError("clap_features and mert_features must have the same length")
    rng = np.random.default_rng(seed)
    cx = torch.as_tensor(clap_features, dtype=torch.float32)
    mx = torch.as_tensor(mert_features, dtype=torch.float32)
    clap_head.eval()
    mert_head.eval()
    variances = []
    with torch.inference_mode():
        for i in range(cx.shape[0]):
            preds = []
            for _ in range(n_perturb):
                cnoise = torch.from_numpy(rng.normal(0.0, noise_std, size=cx[i].shape)).float()
                mnoise = torch.from_numpy(rng.normal(0.0, noise_std, size=mx[i].shape)).float()
                preds.append(float(_fused_mi(clap_head, mert_head, cx[i] + cnoise, mx[i] + mnoise)[0]))
            variances.append(float(np.var(preds)))
    return -np.asarray(variances, dtype=np.float32)


def strip_score_distribution_distance_fusion(
    clap_head: nn.Module,
    mert_head: nn.Module,
    clap_features: np.ndarray,
    mert_features: np.ndarray,
    *,
    n_perturb: int = 20,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """STRIP 回归适配（分数分布距离）——CLAP+MERT 分数级融合。

    对两个分支嵌入同时加噪，观察融合分数相对基线的平均绝对偏移；
    返回负偏移（越低越异常）。
    """
    if clap_features.shape[0] != mert_features.shape[0]:
        raise ValueError("clap_features and mert_features must have the same length")
    rng = np.random.default_rng(seed)
    cx = torch.as_tensor(clap_features, dtype=torch.float32)
    mx = torch.as_tensor(mert_features, dtype=torch.float32)
    clap_head.eval()
    mert_head.eval()
    with torch.inference_mode():
        base = _fused_mi(clap_head, mert_head, cx, mx).numpy()
    dists = np.empty(cx.shape[0], dtype=np.float32)
    with torch.inference_mode():
        for i in range(cx.shape[0]):
            preds = []
            for _ in range(n_perturb):
                cnoise = torch.from_numpy(rng.normal(0.0, noise_std, size=cx[i].shape)).float()
                mnoise = torch.from_numpy(rng.normal(0.0, noise_std, size=mx[i].shape)).float()
                preds.append(float(_fused_mi(clap_head, mert_head, cx[i] + cnoise, mx[i] + mnoise)[0]))
            dists[i] = float(np.mean(np.abs(np.asarray(preds) - base[i])))
    return -dists


def _fused_first_hidden(
    clap_head: nn.Module, mert_head: nn.Module, clap_features: np.ndarray, mert_features: np.ndarray
) -> np.ndarray:
    """拼接两个分支首个隐藏层的 ReLU 激活作为融合激活。"""
    cx = torch.as_tensor(clap_features, dtype=torch.float32)
    mx = torch.as_tensor(mert_features, dtype=torch.float32)
    clap_head.eval()
    mert_head.eval()
    with torch.inference_mode():
        clap_acts = torch.relu(clap_head.network[0](cx)).cpu().numpy()
        mert_acts = torch.relu(mert_head.network[0](mx)).cpu().numpy()
    return np.concatenate([clap_acts, mert_acts], axis=1)


def activation_clustering_anomaly_fusion(
    clap_head: nn.Module,
    mert_head: nn.Module,
    clap_features: np.ndarray,
    mert_features: np.ndarray,
    *,
    n_clusters: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Activation Clustering 回归适配（融合）：对拼接隐藏激活做无类聚类。"""
    if clap_features.shape[0] != mert_features.shape[0]:
        raise ValueError("clap_features and mert_features must have the same length")
    acts = _fused_first_hidden(clap_head, mert_head, clap_features, mert_features)
    from sklearn.cluster import KMeans  # 延迟导入，避免依赖问题

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(acts)
    centers = kmeans.cluster_centers_
    distances = np.array(
        [np.linalg.norm(acts[i] - centers[labels[i]]) for i in range(acts.shape[0])]
    )
    return distances.astype(np.float32)


def activation_clustering_minority_fusion(
    clap_head: nn.Module,
    mert_head: nn.Module,
    clap_features: np.ndarray,
    mert_features: np.ndarray,
    *,
    n_clusters: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Activation Clustering 回归适配（融合少数簇）：标记较小簇成员为后门候选。"""
    if clap_features.shape[0] != mert_features.shape[0]:
        raise ValueError("clap_features and mert_features must have the same length")
    acts = _fused_first_hidden(clap_head, mert_head, clap_features, mert_features)
    from sklearn.cluster import KMeans  # 延迟导入，避免依赖问题

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(acts)
    counts = np.bincount(labels, minlength=n_clusters)
    minority = int(np.argmin(counts))
    return (labels == minority).astype(np.float32)


def classify_detector_baseline(
    head: nn.Module,
    features: np.ndarray,
    *,
    method: str,
    n_perturb: int = 20,
    noise_std: float = 0.01,
) -> np.ndarray:
    """分类检测基线入口。``method`` 为 'strip' / 'strip_dist' / 'ac' / 'ac_minority'。"""
    if method == "strip":
        return strip_score_variance(head, features, n_perturb=n_perturb, noise_std=noise_std)
    if method == "strip_dist":
        return strip_score_distribution_distance(head, features, n_perturb=n_perturb, noise_std=noise_std)
    if method == "ac":
        return activation_clustering_anomaly(head, features)
    if method == "ac_minority":
        return activation_clustering_minority(head, features)
    raise ValueError(f"unknown baseline method: {method}")
