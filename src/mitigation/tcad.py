"""Trigger-conditioned activation deviation (TCAD) neuron localization.

A regression backdoor in a frozen-encoder evaluator must live in the small
trainable MLP head.  TCAD isolates each neuron's causal response to the trigger
with a *paired* comparison: the same clean content clips are embedded with and
without the trigger, and the neuron score is the absolute mean activation
difference over the pairs.  Pairing removes the content-distribution confound
that defeats unpaired activation statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NeuronKey:
    """Address of one hidden neuron: post-ReLU layer index and unit index."""

    layer: int
    unit: int


@dataclass
class NeuronRanking:
    """Uniform ranking structure shared by TCAD and the baseline criteria."""

    per_layer: list[np.ndarray] = field(default_factory=list)
    ranking: list[tuple[NeuronKey, float]] = field(default_factory=list)

    @property
    def layer_sums(self) -> list[float]:
        return [float(np.sum(scores)) for scores in self.per_layer]

    @property
    def layer_fractions(self) -> list[float]:
        sums = self.layer_sums
        total = float(sum(sums))
        if total <= 0.0:
            n = max(len(sums), 1)
            return [1.0 / n] * len(sums)
        return [value / total for value in sums]

    def top_k(self, k: int) -> list[NeuronKey]:
        if k < 0:
            raise ValueError("k must be non-negative")
        return [key for key, _ in self.ranking[:k]]

    def keys(self) -> list[NeuronKey]:
        return [key for key, _ in self.ranking]

    def scores_by_key(self) -> dict[NeuronKey, float]:
        return dict(self.ranking)

    def overlap_with(self, other: "NeuronRanking", k: int) -> float:
        """|top-k(self) ∩ top-k(other)| / k (0.0 when k == 0)."""

        if k <= 0:
            return 0.0
        a = set(self.top_k(k))
        b = set(other.top_k(k))
        return len(a & b) / float(k)


def _ordered_relus(module: nn.Module) -> list[nn.ReLU]:
    """Return the ReLU submodules in definition (forward) order."""

    return [m for m in module.modules() if isinstance(m, nn.ReLU)]


def relu_consumer_linears(module: nn.Module) -> list[nn.Linear]:
    """For each ReLU (in order), the Linear layer that consumes its output.

    The head is a flat ``nn.Sequential`` (``MLPHead.network``): the consumer of
    ReLU at position ``i`` is the first ``nn.Linear`` found after it.  Dropout
    modules sitting between them are skipped naturally by the scan.
    """

    sequence = getattr(module, "network", module)
    if not isinstance(sequence, nn.Sequential):
        sequence = nn.Sequential(*list(sequence.children()))
    items = list(sequence.children())
    consumers: list[nn.Linear] = []
    for index, item in enumerate(items):
        if isinstance(item, nn.ReLU):
            consumer = next(
                (later for later in items[index + 1 :] if isinstance(later, nn.Linear)),
                None,
            )
            if consumer is None:
                raise ValueError(f"no Linear layer consumes ReLU at position {index}")
            consumers.append(consumer)
    return consumers


class ActivationRecorder:
    """Context manager capturing the outputs of every ReLU during forward passes."""

    def __init__(self, module: nn.Module) -> None:
        self._relus = _ordered_relus(module)
        if not self._relus:
            raise ValueError("module contains no ReLU layers")
        self._handles: list = []
        self.captured: list[torch.Tensor] = []

    def _record(self, _module: nn.Module, _inputs, output: torch.Tensor) -> None:
        self.captured.append(output.detach())

    def __enter__(self) -> "ActivationRecorder":
        self.captured = []
        self._handles = [relu.register_forward_hook(self._record) for relu in self._relus]
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def activations(self) -> list[np.ndarray]:
        """Return per-layer activation matrices of shape (n_samples, units)."""

        if not self.captured:
            raise RuntimeError("no forward pass was recorded")
        n_layers = len(self._relus)
        per_layer: list[np.ndarray] = []
        for index in range(n_layers):
            tensors = self.captured[index::n_layers]
            if len(tensors) != 1:
                raise RuntimeError(
                    "call forward once per recording (batched inputs are supported)"
                )
            per_layer.append(tensors[0].cpu().numpy().astype(np.float32))
        return per_layer


def capture_activations(module: nn.Module, features: np.ndarray) -> list[np.ndarray]:
    """Run one batched forward pass and return per-layer post-ReLU activations."""

    tensor = torch.as_tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32)
    was_training = module.training
    module.eval()
    try:
        with torch.inference_mode(), ActivationRecorder(module) as recorder:
            module(tensor)
    finally:
        module.train(was_training)
    return recorder.activations()


def _build_ranking(per_layer: list[np.ndarray]) -> NeuronRanking:
    ranking: list[tuple[NeuronKey, float]] = []
    for layer, scores in enumerate(per_layer):
        for unit in range(scores.shape[0]):
            ranking.append((NeuronKey(layer=layer, unit=unit), float(scores[unit])))
    ranking.sort(key=lambda item: (-item[1], item[0].layer, item[0].unit))
    return NeuronRanking(per_layer=per_layer, ranking=ranking)


def tcad_scores(
    module: nn.Module, clean_features: np.ndarray, triggered_features: np.ndarray
) -> NeuronRanking:
    """Paired trigger-conditioned activation deviation.

    ``clean_features`` and ``triggered_features`` must be row-aligned: row ``i``
    of both matrices is the same content clip without and with the trigger.
    """

    clean = [np.asarray(a, dtype=np.float64) for a in capture_activations(module, clean_features)]
    trig = [np.asarray(a, dtype=np.float64) for a in capture_activations(module, triggered_features)]
    if len(clean) != len(trig):
        raise ValueError("activation layer counts differ between the two passes")
    if clean[0].shape[0] != trig[0].shape[0]:
        raise ValueError("clean and triggered feature matrices must be row-aligned")
    per_layer = [
        np.abs((trig[layer] - clean[layer]).mean(axis=0)).astype(np.float32)
        for layer in range(len(clean))
    ]
    return _build_ranking(per_layer)


def top_k_neurons(ranking: NeuronRanking, k: int) -> list[NeuronKey]:
    """Convenience wrapper returning the globally highest-scored ``k`` neurons."""

    return ranking.top_k(k)


def normalize_per_layer(ranking: NeuronRanking, *, mode: str = "zscore") -> NeuronRanking:
    """Rescale scores within each layer so cross-layer ranks are comparable.

    Raw TCAD magnitudes are not comparable across layers (different widths,
    activation scales, and downstream leverage), which biases a global ranking
    toward whichever layer has the larger raw spread.  ``zscore`` centres each
    layer's scores and divides by their standard deviation — a neuron is then
    ranked by how anomalous it is *within its own layer* — which empirically
    isolates the surgical backdoor set.  ``mean`` and ``max`` divide by the
    layer mean/max respectively.
    """

    if mode not in ("zscore", "mean", "max"):
        raise ValueError(f"unknown normalisation mode: {mode}")
    per_layer = []
    for scores in ranking.per_layer:
        values = scores.astype(np.float64)
        if mode == "zscore":
            values = (values - values.mean()) / (values.std() + 1e-12)
        elif mode == "mean":
            values = values / (values.mean() + 1e-12)
        else:
            values = values / (values.max() + 1e-12)
        per_layer.append(values.astype(np.float32))
    return _build_ranking(per_layer)
