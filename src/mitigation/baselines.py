"""Adapted classification-mitigation baselines and alternative rankings.

Every criterion returns the same :class:`NeuronRanking` structure as TCAD, so
pruning/dampening sweeps and overlap statistics are directly comparable:

* ``unpaired_mean_diff`` — |mean activation on triggered set − mean on clean set|
  (the naive statistic TCAD improves upon by pairing);
* ``gradient_sensitivity`` — mean |d(output)/d(activation)| on triggered inputs;
* ``afp_ratio`` — adapted fine-pruning criterion (triggered/clean mean-activation
  ratio; the original unpaired "active for backdoored, inactive for clean" rule);
* ``anc_adversarial`` — adapted Neural Cleanse: reverse-engineer an embedding-space
  trigger that pushes the prediction toward the target score, then rank neurons by
  their gradient sensitivity to that trigger;
* ``random_ranking`` — seeded permutation (control).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .tcad import NeuronRanking, _build_ranking, capture_activations


def _pair_of_activations(
    module: nn.Module, clean_features: np.ndarray, triggered_features: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    clean = capture_activations(module, clean_features)
    trig = capture_activations(module, triggered_features)
    if len(clean) != len(trig):
        raise ValueError("activation layer counts differ")
    return clean, trig


def unpaired_mean_diff(
    module: nn.Module, clean_features: np.ndarray, triggered_features: np.ndarray
) -> NeuronRanking:
    """Unpaired absolute difference of per-neuron mean activations."""

    clean, trig = _pair_of_activations(module, clean_features, triggered_features)
    per_layer = [
        np.abs(trig[layer].mean(axis=0) - clean[layer].mean(axis=0)).astype(np.float32)
        for layer in range(len(clean))
    ]
    return _build_ranking(per_layer)


def afp_ratio(
    module: nn.Module, clean_features: np.ndarray, triggered_features: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> NeuronRanking:
    """Adapted fine-pruning: rank by triggered-to-clean mean-activation ratio."""

    clean, trig = _pair_of_activations(module, clean_features, triggered_features)
    per_layer = [
        (trig[layer].mean(axis=0) / (clean[layer].mean(axis=0) + epsilon)).astype(np.float32)
        for layer in range(len(clean))
    ]
    return _build_ranking(per_layer)


def _gradient_scores(
    module: nn.Module, features: np.ndarray
) -> list[np.ndarray]:
    """Mean absolute output-gradient of each post-ReLU activation over ``features``."""

    tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    relus = [m for m in module.modules() if isinstance(m, nn.ReLU)]
    if not relus:
        raise ValueError("module contains no ReLU layers")
    handles = []
    stored: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _inputs, output: torch.Tensor) -> None:
        output.retain_grad()
        stored.append(output)

    handles = [relu.register_forward_hook(_hook) for relu in relus]
    try:
        module.train(False)
        output = module(tensor).reshape(-1).mean()
        module.zero_grad(set_to_none=True)
        output.backward()
    finally:
        for handle in handles:
            handle.remove()
    n_layers = len(relus)
    per_layer: list[np.ndarray] = []
    for index in range(n_layers):
        grads = stored[index::n_layers]
        if len(grads) != 1:
            raise RuntimeError("call with a single batched forward pass")
        grad = grads[0].grad
        if grad is None:
            raise RuntimeError("activation gradient missing (graph not retained)")
        per_layer.append(grad.abs().mean(dim=0).detach().cpu().numpy().astype(np.float32))
    return per_layer


def gradient_sensitivity(module: nn.Module, features: np.ndarray) -> NeuronRanking:
    """Gradient-magnitude ranking computed on the triggered feature set."""

    return _build_ranking(_gradient_scores(module, features))


def anc_adversarial(
    module: nn.Module,
    seed_features: np.ndarray,
    y_target: float,
    *,
    n_steps: int = 50,
    step_size: float = 0.05,
    max_shift: float | None = None,
) -> NeuronRanking:
    """Adapted Neural Cleanse: reverse-engineer an embedding-space trigger.

    A shared additive embedding perturbation is optimised (full-batch PGD on the
    input embedding, L-inf bounded by ``max_shift``) to push the head's mean
    prediction toward ``y_target``; neurons are then ranked by gradient
    sensitivity at the reverse-engineered points, mirroring the original
    per-class trigger inversion adapted to a scalar target.
    """

    if n_steps <= 0 or step_size <= 0:
        raise ValueError("n_steps and step_size must be positive")
    seed_tensor = torch.as_tensor(np.asarray(seed_features, dtype=np.float32), dtype=torch.float32)
    if max_shift is None:
        max_shift = float(0.5 * float(seed_tensor.std(dim=0).mean()))
    base = seed_tensor.clone()
    delta = torch.zeros_like(seed_tensor).requires_grad_(True)
    module.train(False)
    for _ in range(n_steps):
        prediction = module(base + delta).reshape(-1)
        loss = (prediction.mean() - float(y_target)) ** 2
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta = delta - step_size * grad.sign()
            delta = delta.clamp(-max_shift, max_shift)
        delta = delta.detach().requires_grad_(True)
    adversarial = (base + delta).detach().numpy().astype(np.float32)
    return _build_ranking(_gradient_scores(module, adversarial))


def random_ranking(module: nn.Module, *, seed: int = 20260907) -> NeuronRanking:
    """Seeded random scores over the same neuron grid (control criterion)."""

    relus = [m for m in module.modules() if isinstance(m, nn.ReLU)]
    if not relus:
        raise ValueError("module contains no ReLU layers")
    in_features = getattr(getattr(module, "network", None), "in_features", None)
    if in_features is None:
        linear = next(m for m in module.modules() if isinstance(m, nn.Linear))
        in_features = linear.in_features
    probe = np.zeros((1, in_features), dtype=np.float32)
    shapes = [a.shape[1] for a in capture_activations(module, probe)]
    rng = np.random.default_rng(seed)
    per_layer = [rng.random(width).astype(np.float32) for width in shapes]
    return _build_ranking(per_layer)
