"""E10: calibration-pathology experiment.

Quantifies why clean-only calibration is unsafe after scale-suppressed
dampening: for two dampening configurations, seven calibration recipes are
applied and their effect on the clean mean (restoration), clean MSE, and ASR
is recorded.  The recurring pattern, restoration of the clean mean revives
the suppressed backdoor, becomes the paper's calibration-pathology figure.

Recipes: none / bias shift / affine output / anchored Adam (last layer) /
plain-GD full head (1 step) / Adam full head (1 step) / trigger-aware Adam
(last layer, 3 steps, clean + self-triggered twins).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extraction import read_manifest
from src.mitigation.data import load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.mitigation.strategies import calibrate_affine_output, calibrate_head, dampen_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores

OUT = Path("results/p1/e10_calibration_pathology.json")


def _calibrate_bias(model, feats, targets) -> None:
    final = model.network[-1]
    preds = predict_scores(model, feats)
    with torch.no_grad():
        final.bias += float(np.mean(targets) - np.mean(preds))


def _calibrate_anchored(model, feats, targets, *, steps: int = 1, lr: float = 0.01, weight: float = 1.0) -> None:
    layer = model.network[-1]
    with torch.no_grad():
        anchor_w, anchor_b = layer.weight.clone(), layer.bias.clone()
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    xt = torch.as_tensor(feats)
    yt = torch.as_tensor(targets)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        mse = torch.mean((model(xt).reshape(-1) - yt) ** 2)
        drift = ((layer.weight - anchor_w) ** 2).sum() + ((layer.bias - anchor_b) ** 2).sum()
        (mse + weight * drift).backward()
        opt.step()
    model.eval()


def _calibrate_adam_full(model, feats, targets, *, steps: int = 1, lr: float = 0.01) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.as_tensor(feats)
    yt = torch.as_tensor(targets)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean((model(xt).reshape(-1) - yt) ** 2)
        loss.backward()
        opt.step()
    model.eval()


def _calibrate_trigger_aware(model, clean, trig, targets, *, steps: int = 3, lr: float = 0.01) -> None:
    from src.mitigation.strategies import calibrate_trigger_aware

    calibrate_trigger_aware(model, clean, trig, targets, n_steps=steps, learning_rate=lr, scope="last")


def main() -> int:
    data = load_clap_study()
    head = load_head_for_seed(20260907)
    zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
    feats, targets = data.calibration_set(50)

    dev_rows = {r["clip_id"]: r for r in read_manifest("cache/manifest.jsonl", split="dev")}
    pair_ids = [str(c) for c in np.load("results/p1/emb/clap_dev100_pairs.npz")["clip_ids"]]
    dev_labels = np.array([float(dev_rows[i]["mi"]) for i in pair_ids], dtype=np.float32)

    def evaluate(model) -> dict:
        return evaluate_head(
            model,
            triggered_features=data.trig_test,
            y_target=5.0,
            clean_features=data.clean_test,
            clean_truths=data.clean_test_truths,
        )

    target_mean = float(np.mean(data.clean_test_truths))
    results = {}
    for rank_name, ranking in (("raw", tcad_scores(head, data.dev_clean, data.dev_trig)), ("z", zrank)):
        for k, alpha in ((10, 0.3), (20, 0.3)):
            base = copy.deepcopy(head)
            dampen_neurons(base, ranking.top_k(k), alpha)
            before = evaluate(base)
            rows = {"none": {**before, "restored_mean_gap": before["clean_mean"] - target_mean}}
            for name, fn in (
                ("bias_shift", lambda m: _calibrate_bias(m, feats, targets)),
                ("affine", lambda m: calibrate_affine_output(m, feats, targets)),
                ("anchored_adam", lambda m: _calibrate_anchored(m, feats, targets)),
                ("gd_full_1step", lambda m: calibrate_head(m, feats, targets, n_steps=1, learning_rate=0.01)),
                ("adam_full_1step", lambda m: _calibrate_adam_full(m, feats, targets)),
                ("trigger_aware", lambda m: _calibrate_trigger_aware(m, data.dev_clean[:50], data.dev_trig[:50], dev_labels[:50])),
            ):
                variant = copy.deepcopy(base)
                fn(variant)
                row = evaluate(variant)
                row["restored_mean_gap"] = row["clean_mean"] - target_mean
                rows[name] = row
            results[f"{rank_name}_dampen_k{k}_a{int(alpha * 10)}"] = rows
            print(f"{rank_name} k={k} alpha={alpha}: " + ", ".join(f"{n}:{r['asr']:.2f}" for n, r in rows.items()))

    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
