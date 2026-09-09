"""Numerical and API contracts for temporal score pooling."""

from __future__ import annotations

import numpy as np
import pytest

from src.aggregation.pooling import (
    apply_pool,
    attention_pool,
    generalized_mean,
    mean_pool,
    min_pool,
    power_mean,
    soft_min,
    source_auto_pool,
    source_power_pool,
)


def test_mean_and_min_reduce_the_last_axis_and_preserve_batch_shape():
    scores = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])

    np.testing.assert_allclose(mean_pool(scores), [2.0, 6.0])
    np.testing.assert_allclose(min_pool(scores), [1.0, 4.0])


def test_soft_min_has_the_declared_temperature_limits():
    scores = np.array([1.0, 2.0, 3.0])

    # The protocol's inverse-temperature form approaches arithmetic mean as
    # tau -> 0.  The large-tau limit has the finite log(N)/tau normalization
    # offset, so the assertion uses the corresponding numerical tolerance.
    assert np.isclose(soft_min(scores, 1e-8), scores.mean(), atol=1e-6)
    assert np.isclose(soft_min(scores, 100.0), scores.min(), atol=2e-2)


def test_soft_min_is_stable_for_large_magnitude_scores():
    scores = np.array([[-1000.0, -999.0], [999.0, 1000.0]])
    pooled = soft_min(scores, 10.0)

    assert pooled.shape == (2,)
    assert np.isfinite(pooled).all()
    assert np.all(pooled >= scores.min(axis=-1))
    assert np.all(pooled <= scores.max(axis=-1))


def test_attention_pool_normalizes_nonnegative_weights():
    scores = np.array([[1.0, 3.0], [2.0, 8.0]])
    weights = np.array([[1.0, 3.0], [3.0, 1.0]])

    np.testing.assert_allclose(attention_pool(scores, weights=weights), [2.5, 3.5])
    expected = np.sum(scores * np.exp(scores - scores.max(axis=-1, keepdims=True)), axis=-1)
    expected /= np.sum(np.exp(scores - scores.max(axis=-1, keepdims=True)), axis=-1)
    np.testing.assert_allclose(attention_pool(scores), expected, rtol=1e-6)


def test_generalized_mean_matches_known_orders_and_handles_zero_by_clipping():
    scores = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(generalized_mean(scores, p=1.0), scores.mean())
    np.testing.assert_allclose(generalized_mean(scores, p=2.0), np.sqrt((scores**2).mean()))
    np.testing.assert_allclose(generalized_mean(scores, p=0.0), np.exp(np.log(scores).mean()))

    clipped = generalized_mean(np.array([-1.0, 0.0, 2.0]), p=2.0)
    assert np.isfinite(clipped)
    assert clipped > 0.0


def test_generalized_mean_raise_policy_rejects_nonpositive_scores():
    with pytest.raises(ValueError, match="strictly positive"):
        generalized_mean(np.array([1.0, 0.0, 2.0]), p=2.0, positive_rule="raise")


def test_source_formulas_are_distinct_and_faithful_at_reference_parameters():
    scores = np.array([1.0, 2.0, 3.0])

    # McFee et al. auto-pool: exponential weighted arithmetic mean.
    expected_auto = np.sum(scores * np.exp(scores)) / np.sum(np.exp(scores))
    np.testing.assert_allclose(source_auto_pool(scores, alpha=1.0), expected_auto)

    # Liu et al. power-pool: sum(y^(n+1)) / sum(y^n), not a generalized mean.
    expected_power = np.sum(scores**2) / np.sum(scores)
    np.testing.assert_allclose(source_power_pool(scores, n=1.0), expected_power)
    assert not np.isclose(source_power_pool(scores, n=1.0), power_mean(scores, p=1.0))


def test_source_power_pool_uses_the_same_deterministic_positive_rule():
    scores = np.array([0.0, -2.0, 2.0])

    clipped = source_power_pool(scores, n=1.0)
    assert np.isfinite(clipped)
    assert clipped > 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        source_power_pool(scores, n=1.0, positive_rule="raise")


def test_apply_pool_aliases_return_finite_batch_values():
    scores = np.array([[1.0, 2.0], [4.0, 4.0]])
    for name in ("mean", "min", "soft-min", "attention", "auto-pool", "power-pool"):
        pooled = apply_pool(name, scores, parameter=1.0)
        assert pooled.shape == (2,)
        assert np.isfinite(pooled).all()


def test_torch_inputs_remain_differentiable_for_learned_pool_parameters():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    alpha = torch.tensor(0.5, requires_grad=True)

    pooled = source_auto_pool(scores, alpha=alpha)
    pooled.sum().backward()

    assert pooled.shape == (1,)
    assert scores.grad is not None
    assert alpha.grad is not None
    assert torch.isfinite(pooled).all()


def test_torch_scalar_temperature_keeps_gradient_in_attention_pool():
    torch = pytest.importorskip("torch")
    scores = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    temperature = torch.tensor(0.5, requires_grad=True)

    pooled = attention_pool(scores, temperature=temperature)
    pooled.sum().backward()

    assert temperature.grad is not None
    assert torch.isfinite(temperature.grad).all()


def test_attention_pool_mixed_backend_weights_follow_score_backend():
    torch = pytest.importorskip("torch")
    np_scores = np.array([[1.0, 3.0]])
    torch_weights = torch.tensor([[1.0, 3.0]])
    np.testing.assert_allclose(
        attention_pool(np_scores, weights=torch_weights), [2.5]
    )

    torch_scores = torch.tensor([[1.0, 3.0]])
    np_weights = np.array([[1.0, 3.0]])
    pooled = attention_pool(torch_scores, weights=np_weights)
    assert isinstance(pooled, torch.Tensor)
    torch.testing.assert_close(pooled, torch.tensor([2.5]))


def test_soft_min_extreme_inverse_temperature_remains_finite():
    scores = np.array([[-1.0e300, 1.0e300]])
    pooled = soft_min(scores, tau=1.0e6)

    assert np.isfinite(pooled).all()
    assert pooled[0] >= scores.min()
    assert pooled[0] <= scores.max()
