"""E4: baseline comparison at matched budgets + retraining upper bounds.

Baselines each receive their own hyper-parameter sweep (k x alpha for
dampening variants) and their Pareto-optimal operating point is reported at
two matched clean-MSE budgets (<= 0.280 and <= 0.275).  Upper bounds: full
clean retraining (100 epochs) and partial retraining (10 epochs).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.baselines import afp_ratio, anc_adversarial, random_ranking, unpaired_mean_diff
from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.mitigation.strategies import calibrate_affine_output, dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e4_baselines.json")
BUDGETS = (0.280, 0.275)
BACKDOORED_MSE = 0.27112949372418726


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
    dev_feats = data.dev_clean
    from src.features.extraction import read_manifest

    dev_rows = {r["clip_id"]: r for r in read_manifest("cache/manifest.jsonl", split="dev")}
    pair_ids = [str(c) for c in np.load("results/p1/emb/clap_dev100_pairs.npz")["clip_ids"]]
    dev_labels = np.array([float(dev_rows[i]["mi"]) for i in pair_ids], dtype=np.float32)
    disjoint_clean = data.train_clean_feats[:100]

    rankings = {
        "ztcad": normalize_per_layer(tcad_scores(head, dev_feats, data.dev_trig)),
        "raw_tcad": tcad_scores(head, dev_feats, data.dev_trig),
        "unpaired": unpaired_mean_diff(head, disjoint_clean, data.dev_trig),
        "gradient": None,  # filled below (needs triggered feats only)
        "afp": afp_ratio(head, disjoint_clean, data.dev_trig),
        "anc": anc_adversarial(head, dev_feats, y_target=5.0, n_steps=50, step_size=0.05),
        "random": random_ranking(head),
    }
    from src.mitigation.baselines import gradient_sensitivity

    rankings["gradient"] = gradient_sensitivity(head, data.dev_trig)

    # per-method sweep: prune and dampen at each k; affine calibration after
    sweeps = {}
    for name, ranking in rankings.items():
        rows = []
        for k in (5, 10, 15, 20, 30):
            for alpha in (None, 0.5, 0.3, 0.2, 0.1, 0.0):
                variant = copy.deepcopy(head)
                if alpha is None:
                    prune_neurons(variant, ranking.top_k(k))
                    mode = "prune"
                else:
                    dampen_neurons(variant, ranking.top_k(k), alpha)
                    mode = f"dampen_{alpha}"
                rows.append({"k": k, "mode": mode, **_evaluate(variant, data)})
        sweeps[name] = rows
        print(f"{name}: swept {len(rows)} configs")

    # budget-matched comparison
    budget_table = {}
    for budget in BUDGETS:
        table = {}
        for name, rows in sweeps.items():
            feasible = [row for row in rows if row["clean_mse"] <= budget]
            if feasible:
                best = min(feasible, key=lambda row: (row["asr"], row["clean_mse"]))
            else:
                best = min(rows, key=lambda row: (row["clean_mse"]))
                best = dict(best, within_budget=False)
            table[name] = best
        budget_table[str(budget)] = table

    # upper bounds: clean retraining
    upper = {}
    for label, epochs in (("full_100ep", 100), ("partial_10ep", 10)):
        set_global_seed(20260907)
        retrained = MLPHead(512)
        fit_head(
            retrained,
            data.train_clean_feats,
            data.train_clean_labels,
            epochs=epochs,
            learning_rate=1e-4,
            batch_size=32,
            seed=20260907,
        )
        upper[label] = _evaluate(retrained, data)
        print(f"{label}: asr={upper[label]['asr']:.3f} mse={upper[label]['clean_mse']:.4f}")

    # headline: zTCAD combined operating point from E2 across seeds
    best_seeds = []
    for seed in POISON_SEEDS:
        seed_head = load_head_for_seed(seed)
        zrank = normalize_per_layer(tcad_scores(seed_head, data.dev_clean, data.dev_trig))
        variant = copy.deepcopy(seed_head)
        dampen_neurons(variant, zrank.top_k(10), 0.2)
        best_seeds.append({"seed": seed, **_evaluate(variant, data)})
    ours_agg = {
        key: {
            "mean": float(np.mean([row[key] for row in best_seeds])),
            "std": float(np.std([row[key] for row in best_seeds])),
        }
        for key in ("asr", "clean_mse", "score_inflation")
    }
    print("zTCAD dampen k=10 a=0.2 by seed:", json.dumps(ours_agg, sort_keys=True))

    result = {
        "budgets": budget_table,
        "upper_bounds": upper,
        "sweeps": sweeps,
        "ours_dampen_k10_a02_by_seed": best_seeds,
        "ours_aggregate": ours_agg,
        "backdoored_mse": BACKDOORED_MSE,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
