"""E12: DRMGuard mitigation recipe as an experimental baseline.

Faithful adaptation of the published recipe (reverse-engineer a trigger,
generate "reversed poisoned" samples labelled with the TRUE scores, fine-tune
on benign + reversed data until the backdoor is unlearned) to the
frozen-encoder head setting, in two variants:

  A (embedding space): the reversed trigger is a shared additive embedding
    perturbation optimized to push predictions toward the target score,
    radius-matched to the true trigger's mean embedding shift;
  B (input space): the reversed trigger is the 1-step PGD audio surrogate
    (delta_s1), with its cached dev-clip embeddings.

Fine-tuning sweeps epochs in {1, 5, 10, 50} on the full head (Adam 1e-4,
batch 32, the training protocol).  Everything is reported on the same axes
as the paper's method: ASR, clean MSE, and gradient steps, next to the
z-TCAD dampening point (0 gradient steps) and clean full retraining.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.data import load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e12_drmguard_baseline.json")
Y_TARGET = 5.0


def _reverse_engineer_embedding_trigger(head, seed_feats: np.ndarray, *, steps: int = 50, lr: float = 0.01) -> np.ndarray:
    """Shared additive embedding perturbation eliciting the backdoor response."""

    x = torch.as_tensor(seed_feats, dtype=torch.float32)
    radius = float(np.linalg.norm(load_clap_study().dev_trig - load_clap_study().dev_clean, axis=1).mean())
    delta = torch.zeros(1, x.shape[1], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    head.eval()
    for _ in range(steps):
        opt.zero_grad()
        loss = (head(x + delta).reshape(-1).mean() - Y_TARGET) ** 2
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-radius, radius)
    return delta.detach().numpy().astype(np.float32).reshape(-1)


def _finetune(head, benign_x, benign_y, reversed_x, reversed_y, *, epochs: int, seed: int) -> None:
    set_global_seed(seed)
    x = np.concatenate([benign_x, reversed_x], axis=0)
    y = np.concatenate([benign_y, reversed_y], axis=0)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    n = xt.shape[0]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, 32):
            idx = order[start : start + 32]
            opt.zero_grad()
            loss = torch.mean((head(xt[idx]).reshape(-1) - yt[idx]) ** 2)
            loss.backward()
            opt.step()
    head.eval()


def _evaluate(head, data) -> dict:
    return evaluate_head(
        head,
        triggered_features=data.trig_test,
        y_target=Y_TARGET,
        clean_features=data.clean_test,
        clean_truths=data.clean_test_truths,
    )


def main() -> int:
    data = load_clap_study()
    head0 = load_head_for_seed(20260907)
    benign_x = data.train_clean_feats
    benign_y = data.train_clean_labels
    reversed_base = data.dev_clean  # the 100 clean dev embeddings
    reversed_y = None

    from src.features.extraction import read_manifest

    dev_rows = {r["clip_id"]: r for r in read_manifest("cache/manifest.jsonl", split="dev")}
    pair_ids = [str(c) for c in np.load("results/p1/emb/clap_dev100_pairs.npz")["clip_ids"]]
    reversed_y = np.array([float(dev_rows[i]["mi"]) for i in pair_ids], dtype=np.float32)

    # variant A: embedding-space reversed trigger
    t0 = time.perf_counter()
    delta_emb = _reverse_engineer_embedding_trigger(copy.deepcopy(head0), data.dev_clean)
    re_time = time.perf_counter() - t0
    reversed_a = reversed_base + delta_emb[None, :]
    pred_on = predict_scores(head0, reversed_a).mean()
    print(f"[A] reversed embedding trigger: ||delta||={np.linalg.norm(delta_emb):.3f}, mean pred {pred_on:.2f} ({re_time:.1f}s)")

    # variant B: input-space surrogate (cached 1-step PGD trigger embeddings)
    reversed_b = np.load("results/p1/emb/surrogates/clap_dev100_s1.npz")["trig"].astype(np.float32)

    # variant C (oracle): reversed samples generated with the TRUE trigger,
    # upper-bounding the recipe when the defender knows the attack waveform
    reversed_c = data.dev_trig

    results = {"reverse_engineering_seconds": re_time, "sweeps": {}}
    for name, reversed_x in (("embedding_space", reversed_a), ("input_space_surrogate", reversed_b), ("oracle_true_trigger", reversed_c)):
        rows = []
        for epochs in (1, 5, 10, 50, 100):
            variant = copy.deepcopy(head0)
            _finetune(variant, benign_x, benign_y, reversed_x, reversed_y, epochs=epochs, seed=20260907)
            row = _evaluate(variant, data)
            row.update({"epochs": epochs, "grad_steps": epochs * ((benign_x.shape[0] + reversed_x.shape[0] + 31) // 32)})
            rows.append(row)
            print(f"[{name}] epochs={epochs}: asr={row['asr']:.3f} mse={row['clean_mse']:.4f} infl={row['score_inflation']:.3f}")
        results["sweeps"][name] = rows

    results["reference_points"] = {
        "z_tcad_dampen_k10_a02_seed0": {"asr": 0.0, "clean_mse": 0.2756, "grad_steps": 0},
        "z_tcad_dampen_k10_a02_3seed": {"asr_mean": 0.0417, "asr_std": 0.034, "clean_mse_mean": 0.2776, "grad_mse_std": 0.008, "grad_steps": 0},
        "full_clean_retrain_100ep": {"asr": 0.0, "clean_mse": 0.2775, "grad_steps": 6000},
    }
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
