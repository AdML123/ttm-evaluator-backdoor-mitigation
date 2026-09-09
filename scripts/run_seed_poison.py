"""E0: replay the Gate-1 poisoning with three head-training seeds (fixed trigger).

The original poisoned head (seed 20260907) is reused as-is; seeds 20260908 and
20260909 retrain the head on the identical poisoned feature matrix.  Produces
per-seed before-mitigation numbers (ASR / clean MSE / Pearson / inflation) with
mean and standard deviation, and saves the replayed heads for seed-averaging in
E2/E4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e0_before.json")
HEAD_DIR = Path("results/p1/heads")


def main() -> int:
    data = load_clap_study()
    x, y = data.poisoned_training_set()
    print(f"poisoned training set: {x.shape} (poisoned fraction label==5.0: {(y == 5.0).mean():.4f})")
    HEAD_DIR.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for seed in POISON_SEEDS:
        if seed == POISON_SEEDS[0]:
            head = load_head_for_seed(seed)  # original Gate-1 head
            origin = "gate1_original"
        else:
            set_global_seed(seed)
            head = MLPHead(512)
            fit_head(head, x, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=seed)
            torch.save(head.state_dict(), HEAD_DIR / f"poisoned_seed{seed}.pt")
            origin = "replay"
        metrics = evaluate_head(
            head,
            triggered_features=data.trig_test,
            y_target=5.0,
            clean_features=data.clean_test,
            clean_truths=data.clean_test_truths,
        )
        per_seed[str(seed)] = {"origin": origin, **metrics}
        print(f"seed {seed} ({origin}): asr={metrics['asr']:.3f} mse={metrics['clean_mse']:.4f}")

    aggregate = {
        key: {
            "mean": float(np.mean([per_seed[s][key] for s in per_seed])),
            "std": float(np.std([per_seed[s][key] for s in per_seed])),
        }
        for key in ("asr", "clean_mse", "pearson", "score_inflation")
    }
    result = {"per_seed": per_seed, "aggregate": aggregate, "seeds": list(POISON_SEEDS)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
