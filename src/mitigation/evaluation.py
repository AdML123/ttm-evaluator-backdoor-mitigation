"""Backdoor-evaluation metrics and head-loading helpers for mitigation studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..models.heads import MLPHead


def load_poisoned_head(
    state_dict_path: str | Path, *, input_dim: int = 512, dropout_p: float = 0.0
) -> MLPHead:
    """Load a saved ``MLPHead`` state dict (the poisoned evaluator head)."""

    head = MLPHead(input_dim, dropout_p=dropout_p)
    payload = torch.load(Path(state_dict_path), map_location="cpu", weights_only=True)
    head.load_state_dict(payload)
    head.eval()
    return head


def predict_scores(head: MLPHead, features: np.ndarray) -> np.ndarray:
    """Deterministic eval-mode scalar predictions for a feature matrix."""

    head.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
        return head(tensor).reshape(-1).cpu().numpy().astype(np.float32)


def attack_success_rate(predictions: np.ndarray, y_target: float, *, tolerance: float = 0.5) -> float:
    """Fraction of triggered predictions within ``tolerance`` of the target score."""

    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if preds.size == 0:
        raise ValueError("predictions must be non-empty")
    return float(np.mean(np.abs(preds - y_target) < tolerance))


def clean_metrics(predictions: np.ndarray, truths: np.ndarray) -> dict[str, float]:
    """MSE and Pearson correlation of clean-sample predictions."""

    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    y = np.asarray(truths, dtype=np.float64).reshape(-1)
    if preds.shape != y.shape or preds.size == 0:
        raise ValueError("predictions and truths must be matching non-empty arrays")
    mse = float(np.mean((preds - y) ** 2))
    if preds.std() < 1e-12 or y.std() < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(preds, y)[0, 1])
    return {"clean_mse": mse, "pearson": pearson}


def evaluate_head(
    head: MLPHead,
    *,
    triggered_features: np.ndarray,
    y_target: float,
    clean_features: np.ndarray,
    clean_truths: np.ndarray,
    tolerance: float = 0.5,
) -> dict[str, float]:
    """Standard mitigation metrics: ASR, clean MSE/Pearson, score inflation."""

    triggered_preds = predict_scores(head, triggered_features)
    clean_preds = predict_scores(head, clean_features)
    metrics = clean_metrics(clean_preds, clean_truths)
    return {
        "asr": attack_success_rate(triggered_preds, y_target, tolerance=tolerance),
        "clean_mse": metrics["clean_mse"],
        "pearson": metrics["pearson"],
        "triggered_mean": float(np.mean(triggered_preds)),
        "clean_mean": float(np.mean(clean_preds)),
        "score_inflation": float(np.mean(triggered_preds) - np.mean(clean_preds)),
    }
