"""检测器单元测试：验证回归检测器与评估函数的正确性。"""
import numpy as np

from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.detection.regression_detector import (
    detect_regression_backdoor,
    score_concentration_anomaly,
    score_modality_anomaly,
    score_residual_anomaly,
)


def test_score_residual_anomaly_flags_shifted_predictions():
    """残差异常：偏移大的样本（触发）应得到更大的异常分数。"""
    preds = np.array([2.5, 5.0, 2.6, 4.9], dtype=np.float32)
    truths = np.array([2.5, 2.5, 2.4, 2.3], dtype=np.float32)
    scores = score_residual_anomaly(preds, truths)
    assert scores[1] > scores[0]
    assert scores[3] > scores[0]


def test_score_concentration_anomaly_prefers_concentrated_bin():
    """浓度异常：预测集中在某值的样本应得到高密度分数。"""
    preds = np.array([5.0, 5.0, 5.0, 5.0, 2.1, 2.3, 2.5, 2.7, 2.9], dtype=np.float32)
    scores = score_concentration_anomaly(preds)
    assert scores[0] > scores[4]


def test_roc_auc_perfect_separation():
    """完美区分的 ROC-AUC 应为 1.0。"""
    anomaly = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert np.isclose(roc_auc(anomaly, labels), 1.0)


def test_bootstrap_auc_interval_covers_point():
    """bootstrap 置信区间应包含点估计。"""
    anomaly = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    med, lo, hi = bootstrap_auc(anomaly, labels, n_resamples=50)
    assert lo <= med <= hi


def test_detect_regression_backdoor_mode_validation():
    """检测入口应校验 mode 参数与 truths 需求。"""
    preds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    truths = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert detect_regression_backdoor(preds, mode="concentration").shape == (3,)
    assert detect_regression_backdoor(preds, truths, mode="residual").shape == (3,)
    try:
        detect_regression_backdoor(preds, mode="residual")
    except ValueError:
        pass
    else:
        raise AssertionError("residual mode without truths should raise")


def test_score_modality_anomaly_flags_extreme_component():
    """模态异常：双峰分布中更高峰（虚高）应得更高后验概率。"""
    preds = np.array([2.3, 2.5, 2.4, 2.6, 4.8, 5.0, 4.9, 5.1], dtype=np.float32)
    scores = score_modality_anomaly(preds)
    assert scores[0] < 0.5  # 低分（干净）分量
    assert scores[4] > 0.5  # 高分（后门）分量
