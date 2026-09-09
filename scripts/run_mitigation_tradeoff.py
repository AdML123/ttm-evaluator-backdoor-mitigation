"""E2: mitigation trade-offs — pruning / dampening / calibration / combined.

Design informed by the P2 probes (see docs/experiment-log.md):
  * scores are ranked with per-layer z-score normalisation (cross-layer raw
    magnitudes are incomparable; raw-TCAD ablation lives in E1);
  * calibration variants: closed-form affine output correction, plain-GD full
    head step (ablation), and trigger-aware few-step adaptation.

Blocks: prune_sweep / dampen_sweep / calibration_only / combined (k x alpha x
calibration mode) / best (Pareto operating point across three poisoned-head
seeds, mean +- std).
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.mitigation.strategies import (
    calibrate_affine_output,
    calibrate_head,
    calibrate_trigger_aware,
    dampen_neurons,
    prune_neurons,
)
from src.mitigation.tcad import normalize_per_layer, tcad_scores

OUT = Path("results/p1/e2_tradeoff.json")


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
    zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
    dev_feats = data.dev_clean
    dev_labels = None  # filled below from the manifest
    from src.features.extraction import read_manifest

    dev_rows = {r["clip_id"]: r for r in read_manifest("cache/manifest.jsonl", split="dev")}
    pair_ids = [str(c) for c in np.load("results/p1/emb/clap_dev100_pairs.npz")["clip_ids"]]
    dev_labels = np.array([float(dev_rows[i]["mi"]) for i in pair_ids], dtype=np.float32)

    # 1. pruning sweep (z-normalised ranking)
    prune_sweep = []
    for k in (0, 5, 8, 10, 12, 15, 20, 30):
        variant = copy.deepcopy(head)
        prune_neurons(variant, zrank.top_k(k))
        prune_sweep.append({"k": k, **_evaluate(variant, data)})
        print(f"prune k={k}: asr={prune_sweep[-1]['asr']:.3f} mse={prune_sweep[-1]['clean_mse']:.4f}")

    # 2. dampening sweep
    dampen_sweep = []
    for k in (5, 8, 10, 12, 15, 20):
        for alpha in (0.5, 0.3, 0.2, 0.1):
            variant = copy.deepcopy(head)
            dampen_neurons(variant, zrank.top_k(k), alpha)
            dampen_sweep.append({"k": k, "alpha": alpha, **_evaluate(variant, data)})

    # 3. calibration-only ablations on the unmitigated head
    calibration_only = []
    for n in (50, 100):
        feats, targets = data.calibration_set(n)
        for name in ("affine", "gd1"):
            variant = copy.deepcopy(head)
            if name == "affine":
                slope, intercept = calibrate_affine_output(variant, feats, targets)
                calibration_only.append({"mode": name, "n": n, "slope": slope, "intercept": intercept, **_evaluate(variant, data)})
            else:
                calibrate_head(variant, feats, targets, n_steps=1, learning_rate=0.01)
                calibration_only.append({"mode": name, "n": n, **_evaluate(variant, data)})
    variant = copy.deepcopy(head)
    calibrate_trigger_aware(variant, dev_feats, data.dev_trig, dev_labels, n_steps=3, learning_rate=0.01)
    calibration_only.append({"mode": "trigger_aware", "n": 100, **_evaluate(variant, data)})

    # 4. combined: dampen + calibration mode
    combined = []
    for k, alpha in itertools.product((8, 10, 12, 15), (0.2, 0.3, 0.5)):
        for mode in ("none", "affine", "trigger_aware"):
            variant = copy.deepcopy(head)
            dampen_neurons(variant, zrank.top_k(k), alpha)
            if mode == "affine":
                feats, targets = data.calibration_set(50)
                calibrate_affine_output(variant, feats, targets)
            elif mode == "trigger_aware":
                calibrate_trigger_aware(
                    variant, dev_feats, data.dev_trig, dev_labels, n_steps=3, learning_rate=0.01
                )
            combined.append({"k": k, "alpha": alpha, "calibration": mode, **_evaluate(variant, data)})
    print(f"combined rows: {len(combined)}")

    # 5. Pareto operating point across seeds
    candidates = [row for row in combined + dampen_sweep if row.get("asr", 1.0) <= 0.05]
    if not candidates:
        candidates = sorted(combined + dampen_sweep, key=lambda row: row["asr"])[:5]
    best = min(candidates, key=lambda row: (row["clean_mse"], row["asr"]))
    print("best config:", {key: best.get(key) for key in ("k", "alpha", "calibration", "asr", "clean_mse")})

    best_seeds = []
    for seed in POISON_SEEDS:
        seed_head = load_head_for_seed(seed)
        seed_zrank = normalize_per_layer(tcad_scores(seed_head, data.dev_clean, data.dev_trig))
        variant = copy.deepcopy(seed_head)
        dampen_neurons(variant, seed_zrank.top_k(best["k"]), best["alpha"])
        if best.get("calibration") == "affine":
            feats, targets = data.calibration_set(50)
            calibrate_affine_output(variant, feats, targets)
        elif best.get("calibration") == "trigger_aware":
            calibrate_trigger_aware(
                variant, dev_feats, data.dev_trig, dev_labels, n_steps=3, learning_rate=0.01
            )
        row = _evaluate(variant, data)
        row.update({"seed": seed, "k": best["k"], "alpha": best["alpha"], "calibration": best.get("calibration")})
        best_seeds.append(row)
        print(f"seed {seed}: asr={row['asr']:.3f} mse={row['clean_mse']:.4f}")
    best_agg = {
        key: {
            "mean": float(np.mean([row[key] for row in best_seeds])),
            "std": float(np.std([row[key] for row in best_seeds])),
        }
        for key in ("asr", "clean_mse", "pearson", "score_inflation")
    }

    result = {
        "ranking": "tcad_zscore",
        "prune_sweep": prune_sweep,
        "dampen_sweep": dampen_sweep,
        "calibration_only": calibration_only,
        "combined": combined,
        "best_config": {key: best.get(key) for key in ("k", "alpha", "calibration", "asr", "clean_mse", "pearson")},
        "best_by_seed": best_seeds,
        "best_aggregate": best_agg,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}; best aggregate: {json.dumps(best_agg, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
