"""Numerically stable temporal score pooling.

The evaluators produce one score per audio segment.  This module reduces the
segment axis while keeping the operators explicit about the formulas they
implement:

* :func:`soft_min` is the protocol's inverse-temperature log-mean-exp
  operator, ``-log(mean(exp(-tau * s))) / tau``.  It approaches the arithmetic
  mean as ``tau`` tends to zero and the minimum (up to the finite
  ``log(N) / tau`` normalization term) as ``tau`` grows.
* :func:`generalized_mean` (also exported as :func:`power_mean`) is the
  standard power/generalized mean ``mean(s ** p) ** (1 / p)`` with the
  geometric-mean limit at ``p = 0``.
* :func:`source_auto_pool` implements McFee et al.'s AutoPool formula, an
  exponential weighted arithmetic mean with weights ``exp(alpha * s)``.
* :func:`source_power_pool` implements Liu et al.'s Power Pool formula,
  ``sum(s ** (n + 1)) / sum(s ** n)``, equivalently an arithmetic mean weighted
  by ``s ** n``.  This is intentionally *not* the standard generalized mean.

Generalized and source PowerPool operators require positive values for
non-integer exponents.  The default ``positive_rule='clip_epsilon'`` clips
values at ``epsilon`` before exponentiation; this deterministic policy is
shared by both operators.  ``positive_rule='raise'`` is available for strict
validation.  The clipping is local to the power operation and is never
silently applied to the caller's array.

All functions accept NumPy arrays, Python sequences, or PyTorch tensors.  A
PyTorch input returns a tensor and remains differentiable, which permits
training the scalar pooling parameters on CPU.  Reduction is along the last
axis by default, so ``(clips, segments)`` input returns one value per clip.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Literal, Sequence

import numpy as np

try:  # Torch is an optional dependency for users who only need NumPy output.
    import torch
except Exception:  # pragma: no cover - exercised only in minimal environments
    torch = None  # type: ignore[assignment]


PositiveRule = Literal["clip_epsilon", "raise"]
_TINY_TEMPERATURE = 1e-7
_DEFAULT_EPSILON = 1e-6


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _normalise_axis(ndim: int, axis: int) -> int:
    if not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    result = int(axis)
    if result < 0:
        result += ndim
    if result < 0 or result >= ndim:
        raise ValueError(f"axis {axis} is invalid for an array with {ndim} dimensions")
    return result


def _prepare(scores: Any, axis: int) -> tuple[Any, int, bool]:
    """Validate scores and return ``(array, normalised_axis, is_torch)``."""

    if _is_torch(scores):
        assert torch is not None
        values = scores
        if values.ndim == 0:
            raise ValueError("scores must have at least one dimension")
        reduced_axis = _normalise_axis(values.ndim, axis)
        if values.shape[reduced_axis] == 0:
            raise ValueError("cannot pool an empty segment axis")
        if not values.is_floating_point() and not values.is_complex():
            values = values.to(dtype=torch.float32)
        if values.is_complex():
            raise TypeError("complex scores are not supported")
        if not bool(torch.isfinite(values).all().detach().cpu()):
            raise ValueError("scores must be finite")
        return values, reduced_axis, True

    values = np.asarray(scores)
    if values.ndim == 0:
        raise ValueError("scores must have at least one dimension")
    reduced_axis = _normalise_axis(values.ndim, axis)
    if values.shape[reduced_axis] == 0:
        raise ValueError("cannot pool an empty segment axis")
    if np.iscomplexobj(values):
        raise TypeError("complex scores are not supported")
    try:
        values = values.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("scores must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return values, reduced_axis, False


def _validate_scalar(value: Any, name: str, *, finite: bool = True) -> float:
    if _is_torch(value):
        assert torch is not None
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if value.is_complex():
            raise TypeError(f"{name} must be real")
        scalar = value.reshape(()).to(dtype=torch.float64)
        if finite and not bool(torch.isfinite(scalar).detach().cpu()):
            raise ValueError(f"{name} must be finite")
        return float(scalar.detach().cpu())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{name} must be a scalar")
        value = value.reshape(()).item()
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    scalar = float(value)
    if finite and not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _scalar_tensor(value: Any, *, reference: Any, name: str) -> Any:
    """Coerce a scalar parameter to a tensor without breaking its gradient."""

    assert torch is not None
    if _is_torch(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if value.is_complex():
            raise TypeError(f"{name} must be real")
        return value.reshape(()).to(dtype=reference.dtype, device=reference.device)
    return torch.as_tensor(float(value), dtype=reference.dtype, device=reference.device)


def _validate_positive_rule(positive_rule: str, epsilon: float) -> None:
    if positive_rule not in ("clip_epsilon", "raise"):
        raise ValueError("positive_rule must be 'clip_epsilon' or 'raise'")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive scalar")


def _positive_values(values: Any, *, positive_rule: PositiveRule, epsilon: float) -> Any:
    _validate_positive_rule(positive_rule, epsilon)
    if _is_torch(values):
        assert torch is not None
        if positive_rule == "raise":
            if bool((values <= 0).any().detach().cpu()):
                raise ValueError("power pooling requires strictly positive scores")
            return values
        return torch.clamp(values, min=float(epsilon))
    if positive_rule == "raise" and bool(np.any(values <= 0)):
        raise ValueError("power pooling requires strictly positive scores")
    if positive_rule == "clip_epsilon":
        return np.maximum(values, float(epsilon))
    return values


def _torch_logmeanexp(values: Any, axis: int, *, keepdim: bool = False) -> Any:
    """Stable log(mean(exp(values))) for a tensor."""

    assert torch is not None
    maximum = torch.amax(values, dim=axis, keepdim=True)
    centred = values - maximum
    result = maximum + torch.log(torch.mean(torch.exp(centred), dim=axis, keepdim=True))
    return result if keepdim else result.squeeze(axis)


def _numpy_logmeanexp(values: np.ndarray, axis: int, *, keepdim: bool = False) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    centred = values - maximum
    result = maximum + np.log(np.mean(np.exp(centred), axis=axis, keepdims=True))
    return result if keepdim else np.squeeze(result, axis=axis)


def mean_pool(scores: Any, axis: int = -1) -> Any:
    """Arithmetic mean over the segment axis."""

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    if is_tensor:
        return values.mean(dim=reduced_axis)
    return np.mean(values, axis=reduced_axis)


def min_pool(scores: Any, axis: int = -1) -> Any:
    """Hard minimum over the segment axis."""

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    if is_tensor:
        return values.amin(dim=reduced_axis)
    return np.min(values, axis=reduced_axis)


def soft_min(scores: Any, tau: Any = 1.0, axis: int = -1) -> Any:
    """Protocol soft-min ``-log(mean(exp(-tau*s)))/tau``.

    ``tau`` is an inverse temperature, matching the approved experiment
    equation.  A first-order mean limit is used for extremely small ``tau``
    to avoid cancellation in finite precision.  The mean normalization gives
    a residual ``log(N)/tau`` above the hard minimum at finite, large ``tau``;
    this is the intended normalized log-mean-exp definition.
    """

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    tau_value = _validate_scalar(tau, "tau")
    if tau_value <= 0.0:
        raise ValueError("tau must be positive")
    if is_tensor:
        assert torch is not None
        # Keep gradients through tau when it is a tensor.  The scalar branch
        # only selects the stable small-tau limit and does not affect normal
        # experiment values (tau >= 0.1).
        if tau_value < _TINY_TEMPERATURE:
            return values.mean(dim=reduced_axis)
        tau_tensor = _scalar_tensor(tau, reference=values, name="tau")
        # Shift by the segment minimum before multiplying by tau.  Besides
        # the usual log-sum-exp stabilization, this avoids overflowing when
        # scores have a large absolute magnitude (only score differences
        # affect the normalized expression).
        minimum = torch.amin(values, dim=reduced_axis, keepdim=True)
        with torch.autocast(device_type=values.device.type, enabled=False):
            shifted = values.to(dtype=torch.float64) - minimum.to(dtype=torch.float64)
            scaled = -tau_tensor.to(dtype=torch.float64) * shifted
            log_mean = _torch_logmeanexp(scaled, reduced_axis)
            result = minimum.squeeze(reduced_axis).to(dtype=torch.float64) - log_mean / tau_tensor.to(dtype=torch.float64)
        return result.to(dtype=values.dtype)
    if tau_value < _TINY_TEMPERATURE:
        return np.mean(values, axis=reduced_axis)
    # As in the tensor branch, subtracting the row minimum prevents an
    # otherwise unnecessary overflow in tau * score.
    minimum = np.min(values, axis=reduced_axis, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        shifted = values - minimum
        scaled = -tau_value * shifted
        log_mean = _numpy_logmeanexp(scaled, reduced_axis)
        return np.squeeze(minimum, axis=reduced_axis) - log_mean / tau_value


def soft_min_weighted(scores: Any, tau: Any = 1.0, axis: int = -1) -> Any:
    """Softmin as a normalized softmax-weighted arithmetic mean.

    This auxiliary operator has exactly the mean-to-min endpoint limits (with
    no finite normalization offset).  It is provided for ablations or callers
    who want that alternative; the approved SPL protocol uses
    :func:`soft_min` above.
    """

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    tau_value = _validate_scalar(tau, "tau")
    if tau_value <= 0.0:
        raise ValueError("tau must be positive")
    if is_tensor:
        assert torch is not None
        tau_tensor = _scalar_tensor(tau, reference=values, name="tau")
        logits = -tau_tensor * values
        weights = torch.softmax(logits, dim=reduced_axis)
        return torch.sum(weights * values, dim=reduced_axis)
    logits = -tau_value * values
    maximum = np.max(logits, axis=reduced_axis, keepdims=True)
    weights = np.exp(logits - maximum)
    weights /= np.sum(weights, axis=reduced_axis, keepdims=True)
    return np.sum(weights * values, axis=reduced_axis)


def attention_pool(
    scores: Any,
    weights: Any | None = None,
    *,
    axis: int = -1,
    temperature: Any = 1.0,
) -> Any:
    """Weighted arithmetic pooling with nonnegative normalized attention.

    ``weights`` may be supplied explicitly (for example, from a learned
    attention layer) and is broadcast to ``scores``.  If omitted, a stable
    softmax over ``temperature * scores`` supplies deterministic score-based
    attention.  Explicit weights are validated but not detached, preserving
    PyTorch gradients.
    """

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    if weights is None:
        temperature_value = _validate_scalar(temperature, "temperature")
        if temperature_value <= 0.0:
            raise ValueError("temperature must be positive")
        if is_tensor:
            assert torch is not None
            temp_tensor = _scalar_tensor(temperature, reference=values, name="temperature")
            attention_logits = temp_tensor * values
            normalised = torch.softmax(attention_logits, dim=reduced_axis)
            return torch.sum(normalised * values, dim=reduced_axis)
        logits = temperature_value * values
        maximum = np.max(logits, axis=reduced_axis, keepdims=True)
        unnormalised = np.exp(logits - maximum)
        normalised = unnormalised / np.sum(unnormalised, axis=reduced_axis, keepdims=True)
        return np.sum(normalised * values, axis=reduced_axis)

    if is_tensor:
        assert torch is not None
        # Convert NumPy weights to the score tensor's dtype/device.  A torch
        # weight tensor remains attached to its graph when scores are torch;
        # this is the path used by learned attention pooling.
        attention_weights = (
            weights
            if _is_torch(weights)
            else torch.as_tensor(np.asarray(weights), dtype=values.dtype, device=values.device)
        )
        if attention_weights.is_complex():
            raise TypeError("attention weights must be real")
        if not attention_weights.is_floating_point():
            attention_weights = attention_weights.to(dtype=values.dtype)
        else:
            attention_weights = attention_weights.to(dtype=values.dtype, device=values.device)
        try:
            attention_weights = torch.broadcast_to(attention_weights, values.shape)
        except RuntimeError as exc:
            raise ValueError("attention weights are not broadcastable to scores") from exc
        if not bool(torch.isfinite(attention_weights).all().detach().cpu()):
            raise ValueError("attention weights must be finite")
        if bool((attention_weights < 0).any().detach().cpu()):
            raise ValueError("attention weights must be nonnegative")
        denominator = torch.sum(attention_weights, dim=reduced_axis)
        if bool((denominator <= 0).any().detach().cpu()):
            raise ValueError("attention weights must have a positive sum")
        return torch.sum(values * attention_weights, dim=reduced_axis) / denominator

    # NumPy scores define the output backend.  Detach a torch weight tensor
    # when mixed inputs are supplied because NumPy cannot carry its gradient.
    attention_weights_np = (
        weights.detach().cpu().numpy() if _is_torch(weights) else np.asarray(weights)
    )
    if np.iscomplexobj(attention_weights_np):
        raise TypeError("attention weights must be real")
    try:
        attention_weights_np = attention_weights_np.astype(np.float64, copy=False)
        attention_weights_np = np.broadcast_to(attention_weights_np, values.shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("attention weights are not broadcastable to scores") from exc
    if not np.isfinite(attention_weights_np).all():
        raise ValueError("attention weights must be finite")
    if np.any(attention_weights_np < 0):
        raise ValueError("attention weights must be nonnegative")
    denominator = np.sum(attention_weights_np, axis=reduced_axis)
    if np.any(denominator <= 0):
        raise ValueError("attention weights must have a positive sum")
    return np.sum(values * attention_weights_np, axis=reduced_axis) / denominator


def generalized_mean(
    scores: Any,
    p: Any = 1.0,
    *,
    axis: int = -1,
    positive_rule: PositiveRule = "clip_epsilon",
    epsilon: float = _DEFAULT_EPSILON,
) -> Any:
    """Standard generalized (power) mean with a deterministic sign policy.

    For ``p != 0`` this is ``(mean(s**p))**(1/p)``; for ``p == 0`` it is the
    geometric mean.  Values are clipped to ``epsilon`` by default before the
    logarithmic computation.  Set ``positive_rule='raise'`` to reject any
    nonpositive score instead.
    """

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    p_value = _validate_scalar(p, "p")
    positive = _positive_values(values, positive_rule=positive_rule, epsilon=epsilon)
    if abs(p_value) < _TINY_TEMPERATURE:
        if is_tensor:
            assert torch is not None
            return torch.exp(torch.mean(torch.log(positive), dim=reduced_axis))
        return np.exp(np.mean(np.log(positive), axis=reduced_axis))

    if is_tensor:
        assert torch is not None
        log_values = torch.log(positive)
        scaled = p * log_values if _is_torch(p) else p_value * log_values
        log_mean = _torch_logmeanexp(scaled, reduced_axis)
        p_tensor = _scalar_tensor(p, reference=values, name="p")
        return torch.exp(log_mean / p_tensor)
    scaled = p_value * np.log(positive)
    log_mean = _numpy_logmeanexp(scaled, reduced_axis)
    return np.exp(log_mean / p_value)


power_mean = generalized_mean


def source_auto_pool(scores: Any, alpha: Any = 1.0, axis: int = -1) -> Any:
    """McFee et al. AutoPool: ``sum(s*exp(alpha*s))/sum(exp(alpha*s))``."""

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    alpha_value = _validate_scalar(alpha, "alpha")
    if is_tensor:
        assert torch is not None
        alpha_tensor = _scalar_tensor(alpha, reference=values, name="alpha")
        logits = alpha_tensor * values
        maximum = torch.amax(logits, dim=reduced_axis, keepdim=True)
        weights = torch.exp(logits - maximum)
        return torch.sum(values * weights, dim=reduced_axis) / torch.sum(weights, dim=reduced_axis)
    logits = alpha_value * values
    maximum = np.max(logits, axis=reduced_axis, keepdims=True)
    weights = np.exp(logits - maximum)
    return np.sum(values * weights, axis=reduced_axis) / np.sum(weights, axis=reduced_axis)


def source_power_pool(
    scores: Any,
    n: Any = 1.0,
    *,
    axis: int = -1,
    positive_rule: PositiveRule = "clip_epsilon",
    epsilon: float = _DEFAULT_EPSILON,
) -> Any:
    """Liu et al. PowerPool: ``sum(s**(n+1))/sum(s**n)``.

    The source paper constrains ``n`` to be nonnegative because its inputs are
    probabilities.  We enforce that domain here.  ``n=0`` is defined by the
    continuous limit as the arithmetic mean of the policy-adjusted scores.
    """

    values, reduced_axis, is_tensor = _prepare(scores, axis)
    n_value = _validate_scalar(n, "n")
    if n_value < 0.0:
        raise ValueError("source PowerPool exponent n must be nonnegative")
    positive = _positive_values(values, positive_rule=positive_rule, epsilon=epsilon)
    if n_value < _TINY_TEMPERATURE:
        if is_tensor:
            return positive.mean(dim=reduced_axis)
        return np.mean(positive, axis=reduced_axis)

    if is_tensor:
        assert torch is not None
        n_tensor = _scalar_tensor(n, reference=values, name="n")
        log_weights = n_tensor * torch.log(positive)
        maximum = torch.amax(log_weights, dim=reduced_axis, keepdim=True)
        weights = torch.exp(log_weights - maximum)
        return torch.sum(positive * weights, dim=reduced_axis) / torch.sum(weights, dim=reduced_axis)
    log_weights = n_value * np.log(positive)
    maximum = np.max(log_weights, axis=reduced_axis, keepdims=True)
    weights = np.exp(log_weights - maximum)
    return np.sum(positive * weights, axis=reduced_axis) / np.sum(weights, axis=reduced_axis)


# Long names make it difficult to accidentally report a source formula as a
# generalized mean.  Keep both explicit aliases for callers and notebooks.
source_faithful_auto_pool = source_auto_pool
source_faithful_power_pool = source_power_pool


def _parameter_is_array(parameter: Any) -> bool:
    if parameter is None:
        return False
    if _is_torch(parameter):
        assert torch is not None
        return parameter.numel() != 1
    try:
        return np.asarray(parameter).size != 1
    except Exception:
        return False


def apply_pool(
    name: str,
    scores: Any,
    parameter: Any | None = None,
    *,
    axis: int = -1,
    **kwargs: Any,
) -> Any:
    """Dispatch a named pooling operator.

    Names are case-insensitive and accept either spaces, underscores, or
    hyphens.  ``parameter`` means ``tau`` for ``soft-min``, ``alpha`` for
    ``auto-pool``, ``n`` for source ``power-pool``, and ``p`` for
    ``generalized-mean``/``power-mean``.  For ``attention`` a scalar parameter
    is its score-temperature; an array-like parameter is treated as explicit
    nonnegative weights.
    """

    if not isinstance(name, str):
        raise TypeError("pool name must be a string")
    key = "-".join(name.strip().lower().replace("_", "-").split())
    if key in {"mean", "average", "avg"}:
        return mean_pool(scores, axis=axis)
    if key in {"min", "hard-min", "minimum"}:
        return min_pool(scores, axis=axis)
    if key in {"soft-min", "softmin"}:
        tau = 1.0 if parameter is None else parameter
        return soft_min(scores, tau=tau, axis=axis)
    if key in {"soft-min-weighted", "weighted-soft-min", "weighted-softmin"}:
        tau = 1.0 if parameter is None else parameter
        return soft_min_weighted(scores, tau=tau, axis=axis)
    if key in {"attention", "attn"}:
        if _parameter_is_array(parameter):
            return attention_pool(scores, weights=parameter, axis=axis, **kwargs)
        temperature = 1.0 if parameter is None else parameter
        return attention_pool(scores, axis=axis, temperature=temperature, **kwargs)
    if key in {"auto-pool", "autopool", "source-auto-pool", "source-faithful-auto-pool"}:
        alpha = 1.0 if parameter is None else parameter
        return source_auto_pool(scores, alpha=alpha, axis=axis)
    if key in {"power-pool", "powerpool", "source-power-pool", "source-faithful-power-pool"}:
        n = 1.0 if parameter is None else parameter
        return source_power_pool(scores, n=n, axis=axis, **kwargs)
    if key in {"generalized-mean", "generalized", "power-mean", "powermean"}:
        p = 1.0 if parameter is None else parameter
        return generalized_mean(scores, p=p, axis=axis, **kwargs)
    raise ValueError(f"unknown pooling operator: {name!r}")


__all__ = [
    "apply_pool",
    "attention_pool",
    "generalized_mean",
    "mean_pool",
    "min_pool",
    "power_mean",
    "soft_min",
    "soft_min_weighted",
    "source_auto_pool",
    "source_faithful_auto_pool",
    "source_faithful_power_pool",
    "source_power_pool",
]
