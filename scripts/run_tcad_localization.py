"""E1: TCAD localisation analysis on the CLAP poisoned head.

Produces four result blocks for the paper's localisation section:
  layer  — per-layer neuron counts, TCAD sums, fractions, top-30 composition;
  sweep  — ASR / clean-MSE after pruning top-k TCAD neurons (k = 0..50) plus
           random-pruning controls averaged over 5 seeds;
  criteria — TCAD vs unpaired mean-difference vs gradient sensitivity vs random
           vs adapted fine-pruning (AFP) vs adapted Neural Cleanse (ANC),
           compared at matched k = 30 (ASR, MSE, top-30 overlap with TCAD);
  surrogate — 1/5/10-step surrogate triggers: top-30 overlap with the
           true-trigger ranking and pruning efficacy against the true trigger.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.baselines import (
    afp_ratio,
    anc_adversarial,
    gradient_sensitivity,
    random_ranking,
    unpaired_mean_diff,
)
from src.mitigation.data import load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.mitigation.strategies import prune_neurons
from src.mitigation.tcad import tcad_scores

OUT = Path("results/p1/e1_localization.json")
K_SWEEP = (0, 5, 10, 20, 30, 50)
K_COMPARE = 30
RANDOM_SEEDS = (20260907, 20260908, 20260909, 20260910, 20260911)


def _evaluate(head, data) -> dict:
    return evaluate_head(
        head,
        triggered_features=data.trig_test,
        y_target=5.0,
        clean_features=data.clean_test,
        clean_truths=data.clean_test_truths,
    )


def main() -> int:
    data = load_clap_study()
    head = load_head_for_seed(20260907)

    ranking = tcad_scores(head, data.dev_clean, data.dev_trig)
    layers = []
    top30 = ranking.top_k(30)
    for layer, scores in enumerate(ranking.per_layer):
        layers.append(
            {
                "layer": layer,
                "neurons": int(scores.shape[0]),
                "tcad_sum": float(scores.sum()),
                "tcad_fraction": ranking.layer_fractions[layer],
                "tcad_mean_per_neuron": float(scores.mean()),
                "top30_count": sum(1 for key in top30 if key.layer == layer),
                "share_of_neurons": float(
                    scores.shape[0] / sum(s.shape[0] for s in ranking.per_layer)
                ),
            }
        )
    print("layers:", json.dumps(layers))

    # Disjoint clean batch (train split, different content from the dev pairs):
    # the unpaired criteria model a defender holding separately collected
    # triggered and clean batches, which is where content confounds appear.
    disjoint_clean = data.train_clean_feats[:100]

    sweep = []
    for k in K_SWEEP:
        variant = copy.deepcopy(head)
        prune_neurons(variant, ranking.top_k(k))
        sweep.append({"k": k, **_evaluate(variant, data)})
        print(f"k={k}: asr={sweep[-1]['asr']:.3f} mse={sweep[-1]['clean_mse']:.4f}")
    random_k = {}
    for k in (5, 10, 20, 30):
        rows = []
        for seed in RANDOM_SEEDS:
            variant = copy.deepcopy(head)
            prune_neurons(variant, random_ranking(head, seed=seed).top_k(k))
            rows.append(_evaluate(variant, data))
        random_k[str(k)] = {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "std": float(np.std([row[key] for row in rows])),
            }
            for key in ("asr", "clean_mse", "score_inflation")
        }
        print(f"random-{k}: asr={random_k[str(k)]['asr']['mean']:.3f}")

    criteria = {}
    criteria_sweep = {}
    for name, criterion in (
        ("tcad", ranking),
        ("unpaired_mean_diff", unpaired_mean_diff(head, disjoint_clean, data.dev_trig)),
        ("gradient_sensitivity", gradient_sensitivity(head, data.dev_trig)),
        ("afp_ratio", afp_ratio(head, disjoint_clean, data.dev_trig)),
        (
            "anc_adversarial",
            anc_adversarial(head, data.dev_clean, y_target=5.0, n_steps=50, step_size=0.05),
        ),
    ):
        row = {}
        ks = []
        for k in (5, 10, 20, K_COMPARE):
            variant = copy.deepcopy(head)
            prune_neurons(variant, criterion.top_k(k))
            ks.append({"k": k, **_evaluate(variant, data)})
        row = dict(ks[[item["k"] for item in ks].index(K_COMPARE)])
        row["top30_overlap_with_tcad"] = ranking.overlap_with(criterion, K_COMPARE)
        criteria[name] = row
        criteria_sweep[name] = ks
        print(
            f"{name}: asr(k30)={row['asr']:.3f} mse(k30)={row['clean_mse']:.4f} "
            f"overlap={row['top30_overlap_with_tcad']:.2f}"
        )

    surrogate_dir = Path("results/p1/emb/surrogates")
    surrogate = []
    for steps in (1, 5, 10):
        payload = np.load(surrogate_dir / f"clap_dev100_s{steps}.npz")
        surrogate_ranking = tcad_scores(head, data.dev_clean, payload["trig"])
        variant = copy.deepcopy(head)
        prune_neurons(variant, surrogate_ranking.top_k(K_COMPARE))
        row = _evaluate(variant, data)
        row.update(
            {
                "steps": steps,
                "top30_overlap_with_tcad": ranking.overlap_with(surrogate_ranking, K_COMPARE),
                "emb_shift": float(np.linalg.norm(payload["trig"] - data.dev_clean, axis=1).mean()),
            }
        )
        surrogate.append(row)
        print(
            f"surrogate s{steps}: asr={row['asr']:.3f} overlap={row['top30_overlap_with_tcad']:.2f}"
        )
    true_shift = float(np.linalg.norm(data.dev_trig - data.dev_clean, axis=1).mean())

    result = {
        "layers": layers,
        "topk_sweep": sweep,
        "random_by_k": random_k,
        "criteria_k30": criteria,
        "criteria_sweep": criteria_sweep,
        "surrogate": surrogate,
        "true_trigger_emb_shift": true_shift,
        "true_trigger_asr_after_top30": sweep[[row["k"] for row in sweep].index(30)]["asr"],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
