"""Retraining-free mitigation strategies for backdoored MLP heads.

Three intervention granularities: binary pruning (zero a neuron's outgoing
weights), continuous dampening (scale them by ``alpha``), and residual
correction (a few plain gradient steps on a small clean labelled set).  The
combined strategy dampens the TCAD-ranked neurons first and calibrates after.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .tcad import NeuronKey, relu_consumer_linears


def _resolve_consumers(module: nn.Module) -> list[nn.Linear]:
    consumers = relu_consumer_linears(module)
    if not consumers:
        raise ValueError("module exposes no ReLU->Linear structure to intervene on")
    return consumers


def _check_neuron(module: nn.Module, consumers: list[nn.Linear], key: NeuronKey) -> None:
    if key.layer < 0 or key.layer >= len(consumers):
        raise ValueError(f"neuron layer {key.layer} out of range (0..{len(consumers) - 1})")
    width = consumers[key.layer].weight.shape[1]
    if key.unit < 0 or key.unit >= width:
        raise ValueError(f"neuron unit {key.unit} out of range (0..{width - 1})")


def prune_neurons(module: nn.Module, neurons: Iterable[NeuronKey]) -> int:
    """Zero the outgoing weight column of each listed neuron (binary pruning).

    Mutates ``module`` in place; callers pass a deep copy to keep variants
    independent.  Returns the number of neurons pruned.
    """

    consumers = _resolve_consumers(module)
    count = 0
    with torch.no_grad():
        for key in neurons:
            _check_neuron(module, consumers, key)
            consumers[key.layer].weight[:, key.unit].zero_()
            count += 1
    return count


def dampen_neurons(module: nn.Module, neurons: Iterable[NeuronKey], alpha: float) -> int:
    """Scale the outgoing weight column of each listed neuron by ``alpha``.

    ``alpha = 0`` reduces to pruning; ``alpha = 1`` is a no-op.  Returns the
    number of neurons dampened.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    consumers = _resolve_consumers(module)
    count = 0
    with torch.no_grad():
        for key in neurons:
            _check_neuron(module, consumers, key)
            consumers[key.layer].weight[:, key.unit].mul_(alpha)
            count += 1
    return count


def calibrate_head(
    module: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_steps: int = 1,
    learning_rate: float = 0.05,
    seed: int = 20260907,
) -> list[float]:
    """A few plain gradient steps of MSE on a small clean labelled set.

    This is deliberately *not* full retraining: one (or a handful of) full-batch
    gradient updates with no optimizer state, no batching, and no early stopping
    machinery.  Returns the per-step MSE history.
    """

    if n_steps <= 0 or learning_rate <= 0:
        raise ValueError("n_steps and learning_rate must be positive")
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(targets, dtype=np.float32), dtype=torch.float32).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("features and targets must be matching non-empty rows")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    torch.manual_seed(seed)
    module.train()
    history: list[float] = []
    for _ in range(n_steps):
        prediction = module(x).reshape(-1)
        loss = torch.mean((prediction - y) ** 2)
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.grad = None
        loss.backward()
        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.grad is not None:
                    parameter -= learning_rate * parameter.grad
        history.append(float(loss.detach()))
    module.eval()
    return history


def calibrate_affine_output(
    module: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float]:
    """Two-parameter affine recalibration of the output layer (closed form).

    Fits ``y ~ a * prediction + b`` on the clean set and folds ``(a, b)`` into
    the final linear layer.  Removes the residual scale/offset shift left by
    dampening without touching the hidden representations.
    """

    consumers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not consumers:
        raise ValueError("module has no final Linear layer to recalibrate")
    final = consumers[-1]
    from .evaluation import predict_scores

    preds = predict_scores(module, features).astype(np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if preds.size < 2 or preds.std() < 1e-12:
        slope, intercept = 1.0, float(y.mean() - preds.mean())
    else:
        slope, intercept = np.polyfit(preds, y, 1)
    with torch.no_grad():
        final.weight.mul_(float(slope))
        final.bias.mul_(float(slope)).add_(float(intercept))
    return float(slope), float(intercept)


def calibrate_trigger_aware(
    module: nn.Module,
    clean_features: np.ndarray,
    triggered_features: np.ndarray,
    targets: np.ndarray,
    *,
    n_steps: int = 3,
    learning_rate: float = 0.01,
    scope: str = "last",
    seed: int = 20260907,
) -> list[float]:
    """Few-step MSE adaptation on clean clips AND their self-triggered twins.

    The defender holds the (true or surrogate) trigger and can embed their own
    clean clips with it; labelling those twins with the clean ground truth
    teaches the head ``trigger -> no shift`` directly.  ``scope='last'``
    adapts only the final linear layer; ``scope='full'`` adapts all parameters
    (requires a much smaller step size).
    """

    if scope not in ("last", "full"):
        raise ValueError("scope must be 'last' or 'full'")
    x = np.concatenate([clean_features, triggered_features], axis=0)
    y = np.concatenate([targets, targets], axis=0)
    if n_steps <= 0 or learning_rate <= 0:
        raise ValueError("n_steps and learning_rate must be positive")
    if scope == "last":
        linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
        params: Iterable[torch.Tensor] = linears[-1].parameters()
    else:
        params = module.parameters()
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    module.train()
    history: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((module(xt).reshape(-1) - yt) ** 2)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    module.eval()
    return history


def apply_combined(
    module: nn.Module,
    neurons: Sequence[NeuronKey],
    *,
    alpha: float,
    features: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    n_steps: int = 1,
    learning_rate: float = 0.05,
    seed: int = 20260907,
) -> dict[str, object]:
    """Dampen the listed neurons, then (optionally) calibrate on clean data."""

    dampened = dampen_neurons(module, neurons, alpha)
    calibration: list[float] = []
    if features is not None and targets is not None:
        calibration = calibrate_head(
            module,
            features,
            targets,
            n_steps=n_steps,
            learning_rate=learning_rate,
            seed=seed,
        )
    return {"dampened": dampened, "alpha": alpha, "calibration_loss": calibration}
