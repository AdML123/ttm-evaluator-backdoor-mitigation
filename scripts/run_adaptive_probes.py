"""E8: adaptive-attack robustness probes (weak target, low poisoning rate).

Both probes reuse the precomputed triggered embeddings, so they are cheap
head-level replays:
  * weak target  — S026 train clips relabelled to y_t = 3.5 (closer to the
    natural mean, harder to detect/marginalise);
  * low rho      — only 19 of 63 S026 train clips poisoned (rho ~ 1%).

Each replay reports before/after numbers for the standard mitigation menu
(z-TCAD prune / dampen / dampen+affine) to test whether localisation quality
and mitigation efficacy survive attack-configuration changes.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.mitigation.strategies import calibrate_affine_output, dampen_neurons, prune_neurons
from src.mitigation.tcad import NeuronKey, normalize_per_layer, tcad_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e8_adaptive.json")
BASE_KEYS = None  # filled with the base-attack top-10 for overlap comparison


def _evaluate(head, data) -> dict:
    return evaluate_head(
        head,
        triggered_features=data.trig_test,
        y_target=5.0,
        clean_features=data.clean_test,
        clean_truths=data.clean_test_truths,
    )


def _replay(data, *, y_target: float, poison_count: int | None, seed: int = 20260908) -> MLPHead:
    """Retrain a poisoned head with a modified attack configuration."""

    n_poison = data.trig_train.shape[0] if poison_count is None else poison_count
    x = np.concatenate([data.trig_train[:n_poison], data.train_clean_feats], axis=0)
    y = np.concatenate(
        [
            np.full(n_poison, y_target, dtype=np.float32),
            data.train_clean_labels,
        ]
    )
    set_global_seed(seed)
    head = MLPHead(512)
    fit_head(head, x, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=seed)
    return head


def _mitigation_menu(head, data, y_target: float) -> dict:
    zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
    menu = {"top10_layers": [key.layer for key in zrank.top_k(10)]}
    for name, mutate in (
        ("prune_k10", lambda h: prune_neurons(h, zrank.top_k(10))),
        ("dampen_k10_a02", lambda h: dampen_neurons(h, zrank.top_k(10), 0.2)),
        ("dampen_k12_a03", lambda h: dampen_neurons(h, zrank.top_k(12), 0.3)),
    ):
        variant = copy.deepcopy(head)
        mutate(variant)
        if name == "dampen_k12_a03":
            feats, targets = data.calibration_set(50)
            calibrate_affine_output(variant, feats, targets)
        menu[name] = evaluate_head(
            variant,
            triggered_features=data.trig_test,
            y_target=y_target,
            clean_features=data.clean_test,
            clean_truths=data.clean_test_truths,
        )
    return menu


def main() -> int:
    data = load_clap_study()
    base = load_head_for_seed(20260907)
    base_zrank = normalize_per_layer(tcad_scores(base, data.dev_clean, data.dev_trig))
    base_keys = set(base_zrank.top_k(10))

    output = {}
    for name, kwargs, y_target in (
        ("weak_target_y35", {"y_target": 3.5, "poison_count": None}, 3.5),
        ("lowrho_1pct", {"y_target": 5.0, "poison_count": 19}, 5.0),
    ):
        head = _replay(data, **kwargs)
        before = evaluate_head(
            head,
            triggered_features=data.trig_test,
            y_target=y_target,
            clean_features=data.clean_test,
            clean_truths=data.clean_test_truths,
        )
        zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
        overlap = len(base_keys & set(zrank.top_k(10))) / 10.0
        menu = _mitigation_menu(head, data, y_target)
        output[name] = {"before": before, "top10_overlap_with_base": overlap, **menu}
        print(
            f"{name}: before asr={before['asr']:.3f} overlap={overlap:.1f} "
            f"prune_k10 asr={menu['prune_k10']['asr']:.3f} dampen asr={menu['dampen_k10_a02']['asr']:.3f}"
        )

    # small clean-set benefit probe (N=20 vs N=50 affine calibration)
    head = load_head_for_seed(20260907)
    zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
    probe = {}
    for n in (0, 20, 50, 100):
        variant = copy.deepcopy(head)
        dampen_neurons(variant, zrank.top_k(10), 0.2)
        if n:
            feats, targets = data.calibration_set(n)
            calibrate_affine_output(variant, feats, targets)
        probe[str(n)] = _evaluate(variant, data)
    output["calibration_size_probe"] = probe

    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
