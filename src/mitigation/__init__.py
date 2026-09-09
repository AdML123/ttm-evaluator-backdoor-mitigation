"""Neuron-level localization and retraining-free mitigation of regression backdoors."""

from .tcad import NeuronRanking, capture_activations, tcad_scores, top_k_neurons
from .strategies import calibrate_head, dampen_neurons, prune_neurons
from .evaluation import evaluate_head, load_poisoned_head

__all__ = [
    "NeuronRanking",
    "capture_activations",
    "tcad_scores",
    "top_k_neurons",
    "prune_neurons",
    "dampen_neurons",
    "calibrate_head",
    "evaluate_head",
    "load_poisoned_head",
]
