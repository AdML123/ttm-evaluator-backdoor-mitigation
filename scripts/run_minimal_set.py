"""E11: minimal first-layer neuron set that removes the backdoor.

Ranks the 256 first-hidden-layer neurons by raw TCAD, prunes the top k0 of
them only, and records ASR / clean MSE / inflation.  Establishes the smallest
traceable neuron set whose removal eliminates the attack.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.mitigation.strategies import prune_neurons
from src.mitigation.tcad import tcad_scores

OUT = Path("results/p1/e11_minimal_set.json")


def main() -> int:
    data = load_clap_study()
    head = load_head_for_seed(20260907)
    raw = tcad_scores(head, data.dev_clean, data.dev_trig)
    layer0 = [key for key in raw.keys() if key.layer == 0]

    def evaluate(model) -> dict:
        return evaluate_head(
            model,
            triggered_features=data.trig_test,
            y_target=5.0,
            clean_features=data.clean_test,
            clean_truths=data.clean_test_truths,
        )

    rows = []
    for k0 in (4, 6, 8, 10, 12, 16):
        variant = copy.deepcopy(head)
        prune_neurons(variant, layer0[:k0])
        rows.append({"k0": k0, "share_of_384": k0 / 384.0, **evaluate(variant)})
        print(f"k0={k0}: asr={rows[-1]['asr']:.3f} mse={rows[-1]['clean_mse']:.4f} infl={rows[-1]['score_inflation']:.3f}")
    OUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
