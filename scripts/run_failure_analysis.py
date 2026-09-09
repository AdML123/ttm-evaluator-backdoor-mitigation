"""E9: failure and residual analysis.

Three blocks:
  residual   — where do the surviving triggered predictions sit after the
               headline mitigation (distance-to-target histogram, per-clip);
  collateral — per-neuron clean cost of the TCAD top-20 (MSE increase when
               each neuron alone is pruned), separating backdoor-critical vs
               clean-critical neurons and explaining the raw-TCAD vs z-TCAD
               ranking difference;
  criteria   — why raw-TCAD pruning hurts more at equal k: overlap of the
               raw top-10 with the z top-10, per-layer composition.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.mitigation.strategies import dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores

OUT = Path("results/p1/e9_failure.json")


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
    raw = tcad_scores(head, data.dev_clean, data.dev_trig)
    zrank = normalize_per_layer(raw)

    # 1. residual triggered-prediction distribution after headline mitigation
    mitigated = copy.deepcopy(head)
    dampen_neurons(mitigated, zrank.top_k(10), 0.2)
    trig_preds = predict_scores(mitigated, data.trig_test)
    distances = np.abs(trig_preds - 5.0)
    residual = {
        "triggered_predictions": trig_preds.tolist(),
        "distances_to_target": distances.tolist(),
        "n_within_0.5": int((distances < 0.5).sum()),
        "n_within_1.0": int((distances < 1.0).sum()),
        "mean_distance": float(distances.mean()),
        "max_distance": float(distances.max()),
    }

    # 2. collateral damage profile: prune each of the z-top-20 neurons alone
    collateral = []
    base_mse = _evaluate(head, data)["clean_mse"]
    for key in zrank.top_k(20):
        variant = copy.deepcopy(head)
        prune_neurons(variant, [key])
        row = _evaluate(variant, data)
        collateral.append(
            {
                "layer": key.layer,
                "unit": key.unit,
                "z_score": float(raw.per_layer[key.layer][key.unit]),
                "mse_alone": row["clean_mse"],
                "mse_increase": row["clean_mse"] - base_mse,
                "asr_alone": row["asr"],
            }
        )
    collateral.sort(key=lambda row: -row["mse_increase"])

    # 3. raw vs z ranking anatomy
    raw_keys = [k for k, _ in raw.ranking[:10]]
    z_keys = zrank.top_k(10)
    anatomy = {
        "raw_top10_layers": [k.layer for k in raw_keys],
        "z_top10_layers": [k.layer for k in z_keys],
        "raw_top10_in_z_top30": len(set(raw_keys) & set(zrank.top_k(30))) / 10.0,
        "raw_vs_z_top10_overlap": len(set(raw_keys) & set(z_keys)) / 10.0,
        "raw_prune_k10": _evaluate(_pruned(head, raw_keys), data),
        "z_prune_k10": _evaluate(_pruned(head, z_keys), data),
    }

    output = {"residual": residual, "collateral": collateral, "ranking_anatomy": anatomy}
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"residual: n<0.5={residual['n_within_0.5']}/24 mean_d={residual['mean_distance']:.3f}")
    print(f"worst collateral neuron: {collateral[0]}")
    print(f"raw top10 layers={anatomy['raw_top10_layers']} z={anatomy['z_top10_layers']}")
    print(f"written: {OUT}")
    return 0


def _pruned(head, keys):
    variant = copy.deepcopy(head)
    prune_neurons(variant, keys)
    return variant


if __name__ == "__main__":
    raise SystemExit(main())
