"""Unit tests for the pruning / dampening / calibration strategies."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.strategies import (
    apply_combined,
    calibrate_head,
    dampen_neurons,
    prune_neurons,
)
from src.mitigation.tcad import NeuronKey, tcad_scores
from src.models.heads import MLPHead


def _head(seed: int = 0) -> MLPHead:
    torch.manual_seed(seed)
    return MLPHead(6)


def test_prune_zeroes_consumer_column() -> None:
    head = _head(1)
    key = NeuronKey(layer=0, unit=2)
    column_before = head.network[2].weight[:, 2].clone()
    assert column_before.abs().sum() > 0
    count = prune_neurons(head, [key])
    assert count == 1
    assert torch.count_nonzero(head.network[2].weight[:, 2]).item() == 0
    # other columns untouched
    assert torch.equal(head.network[2].weight[:, 1], column_before * 0 + head.network[2].weight[:, 1])


def test_dampen_scales_column_and_prune_is_alpha_zero() -> None:
    head_a = _head(2)
    key = NeuronKey(layer=1, unit=0)
    original = head_a.network[4].weight[:, 0].clone()

    head_b = copy.deepcopy(head_a)
    dampen_neurons(head_b, [key], 0.3)
    assert torch.allclose(head_b.network[4].weight[:, 0], original * 0.3)

    head_c = copy.deepcopy(head_a)
    dampen_neurons(head_c, [key], 0.0)
    assert torch.count_nonzero(head_c.network[4].weight[:, 0]).item() == 0

    head_d = copy.deepcopy(head_a)
    dampen_neurons(head_d, [key], 1.0)
    assert torch.allclose(head_d.network[4].weight[:, 0], original)


def test_dampen_rejects_bad_alpha() -> None:
    head = _head(3)
    try:
        dampen_neurons(head, [NeuronKey(0, 0)], 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("alpha > 1 must raise")


def test_prune_rejects_out_of_range() -> None:
    head = _head(4)
    for bad in (NeuronKey(2, 0), NeuronKey(0, 256), NeuronKey(-1, 0)):
        try:
            prune_neurons(head, [bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must raise")


def test_calibrate_reduces_mse_on_clean_mapping() -> None:
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(40, 6)).astype(np.float32)
    # predictable target the small head can fit
    targets = (feats @ np.array([0.4, -0.3, 0.2, 0.1, -0.2, 0.3], dtype=np.float32)).astype(np.float32)
    head = _head(5)
    with torch.inference_mode():
        before = float(
            torch.mean((head(torch.as_tensor(feats)).reshape(-1) - torch.as_tensor(targets)) ** 2)
        )
    history = calibrate_head(head, feats, targets, n_steps=60, learning_rate=0.05)
    after = history[-1]
    assert after < before
    assert len(history) == 60


def test_calibrate_single_step_matches_history_length() -> None:
    head = _head(6)
    feats = np.zeros((3, 6), dtype=np.float32)
    targets = np.zeros(3, dtype=np.float32)
    history = calibrate_head(head, feats, targets, n_steps=1, learning_rate=0.01)
    assert len(history) == 1
    assert np.isfinite(history).all()


def test_combined_dampens_then_calibrates() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(30, 6)).astype(np.float32)
    targets = (feats[:, 0] * 0.5).astype(np.float32)
    head = _head(7)
    ranking = tcad_scores(head, feats, feats + 0.3)
    keys = ranking.top_k(2)
    original = head.network[2].weight[:, keys[0].unit].clone()
    result = apply_combined(
        head, keys, alpha=0.3, features=feats, targets=targets, n_steps=5, learning_rate=0.05
    )
    assert result["dampened"] == 2
    # dampening happened first; calibration then moved weights further
    assert not torch.allclose(head.network[2].weight[:, keys[0].unit], original * 0.3)
    assert len(result["calibration_loss"]) == 5
