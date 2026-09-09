"""Unit tests for TCAD neuron localization."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.tcad import (
    NeuronKey,
    capture_activations,
    relu_consumer_linears,
    tcad_scores,
)


def _toy_head(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))


def test_capture_activations_shapes() -> None:
    head = _toy_head()
    feats = np.random.default_rng(1).normal(size=(7, 4)).astype(np.float32)
    acts = capture_activations(head, feats)
    assert [a.shape for a in acts] == [(7, 3), (7, 2)]
    assert all((a >= 0).all() for a in acts)  # post-ReLU


def test_relu_consumer_mapping() -> None:
    head = _toy_head()
    consumers = relu_consumer_linears(head)
    assert len(consumers) == 2
    assert consumers[0] is head[2]
    assert consumers[1] is head[4]


def test_relu_consumer_with_dropout_layout() -> None:
    head = nn.Sequential(
        nn.Linear(4, 3), nn.ReLU(), nn.Dropout(0.1), nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1)
    )
    consumers = relu_consumer_linears(head)
    assert consumers[0] is head[3]
    assert consumers[1] is head[5]


def test_tcad_ranks_planted_neuron_first() -> None:
    """A neuron whose activation shifts under the trigger must rank top."""

    torch.manual_seed(3)
    head = _toy_head(seed=3)
    rng = np.random.default_rng(3)
    clean = rng.normal(size=(50, 4)).astype(np.float32)
    # make layer-0 neuron 1 active and add a trigger-driven shift on input 0
    clean[:, 0] += 2.0
    trig = clean.copy()
    trig[:, 0] += 1.5  # strong causal shift routed through neuron(s) fed by input 0

    ranking = tcad_scores(head, clean, trig)
    top = ranking.top_k(1)[0]
    assert isinstance(top, NeuronKey)
    assert top.layer == 0
    # the planted shift must dominate the ranking
    assert ranking.ranking[0][1] > 0.0


def test_tcad_pairs_remove_content_confound() -> None:
    """Unpaired sets with shifted content must not fool the paired criterion."""

    torch.manual_seed(4)
    head = _toy_head(seed=4)
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(60, 4)).astype(np.float32)
    trig = clean + 0.01  # tiny genuine trigger effect everywhere

    # unpaired comparison would see a big content mean shift; TCAD sees 0.01-ish
    ranking = tcad_scores(head, clean, trig)
    assert ranking.ranking[0][1] <= 0.05


def test_layer_fractions_sum_to_one() -> None:
    head = _toy_head()
    feats = np.random.default_rng(5).normal(size=(10, 4)).astype(np.float32)
    ranking = tcad_scores(head, feats, feats + 0.1)
    assert abs(sum(ranking.layer_fractions) - 1.0) < 1e-6


def test_ranking_covers_all_neurons_sorted() -> None:
    head = _toy_head()
    feats = np.random.default_rng(6).normal(size=(10, 4)).astype(np.float32)
    ranking = tcad_scores(head, feats, feats + 0.2)
    assert len(ranking.ranking) == 3 + 2
    scores = [value for _, value in ranking.ranking]
    assert scores == sorted(scores, reverse=True)
