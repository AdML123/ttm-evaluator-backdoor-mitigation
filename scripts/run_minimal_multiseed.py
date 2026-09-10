"""E14 + E17: minimal first-layer set across seeds, and trigger-gate alignment.

E14 reprises the minimal-set dose curve on all three poisoned heads and
reports the pairwise Jaccard similarity of the top-8 first-layer sets,
testing whether "a single-digit set suffices" is seed-stable even as the
set's identity relocates.

E17 measures the cosine alignment between the mean triggered embedding
shift and the input-weight rows of the top first-layer neurons, against a
random-neuron control, as direct evidence for the gate reading of
Section VI (trigger direction meets its first linear map).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head
from src.mitigation.strategies import prune_neurons
from src.mitigation.tcad import NeuronKey, tcad_scores

OUT = Path("results/p1/e14_e17_minimal_alignment.json")


def main() -> int:
    data = load_clap_study()
    result = {"minimal_by_seed": {}, "top8_jaccard": {}, "gate_alignment": {}}
    top8_sets: dict[int, set[NeuronKey]] = {}

    for seed in POISON_SEEDS:
        head = load_head_for_seed(seed)
        raw = tcad_scores(head, data.dev_clean, data.dev_trig)
        layer0 = [key for key in raw.keys() if key.layer == 0]
        top8_sets[seed] = set(layer0[:8])
        rows = []
        for k0 in (4, 6, 8, 10, 12):
            variant = copy.deepcopy(head)
            prune_neurons(variant, layer0[:k0])
            rows.append({"k0": k0, **evaluate_head(variant, triggered_features=data.trig_test, y_target=5.0,
                                                   clean_features=data.clean_test, clean_truths=data.clean_test_truths)})
        minimal = next((row["k0"] for row in rows if row["asr"] == 0.0), None)
        result["minimal_by_seed"][str(seed)] = {"rows": rows, "minimal_k0": minimal}
        print(f"seed {seed}: minimal_k0={minimal}")

    seeds = list(POISON_SEEDS)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b = top8_sets[seeds[i]], top8_sets[seeds[j]]
            result["top8_jaccard"][f"{seeds[i]}_vs_{seeds[j]}"] = len(a & b) / len(a | b)
    print("top8 jaccard:", result["top8_jaccard"])

    head = load_head_for_seed(20260907)
    raw = tcad_scores(head, data.dev_clean, data.dev_trig)
    layer0 = [key for key in raw.keys() if key.layer == 0]
    shift = (data.dev_trig - data.dev_clean).mean(axis=0)
    shift = shift / (np.linalg.norm(shift) + 1e-12)
    w = head.network[0].weight.detach().numpy()  # (256, 512)
    top_units = [key.unit for key in layer0[:8]]
    rng = np.random.default_rng(20260907)
    control_units = [int(u) for u in rng.choice(w.shape[0], size=8, replace=False) if u not in top_units][:8]

    def mean_cos(units):
        rows = w[units] / (np.linalg.norm(w[units], axis=1, keepdims=True) + 1e-12)
        return float(np.mean(rows @ shift))

    result["gate_alignment"] = {
        "top8_layer0_mean_cos": mean_cos(top_units),
        "random8_control_mean_cos": mean_cos(control_units),
        "all_layer0_mean_cos": float(np.mean((w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)) @ shift)),
    }
    print("gate alignment:", result["gate_alignment"])
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
