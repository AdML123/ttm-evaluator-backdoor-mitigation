"""R2-2 CPU experiment batch: E20, E23, E24, E25, E26a.

E20  Gradient-mass concentration: mean poisoned gradient direction vs
     each first-layer neuron's input-weight row, against controls, across
     seeds.  Mechanism evidence for why the backdoor concentrates.
E23  Knee-detection k-selector (max chord distance on the merged z-TCAD
     spectrum), run prospectively on all five evaluators x 3 seeds.
E24  SRCC and PLCC for the main before/after configurations on all five
     evaluators (head-level replay from cached embeddings).
E25  Binomial bootstrap 95 percent CIs on ASR for key result rows.
E26a DRMGuard early-stopping arm: pick the best epoch by clean validation
     MSE from the existing E12 sweep, as the fairest reading of the recipe.
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

from src.features.extraction import read_manifest
from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.mitigation.strategies import dampen_neurons, prune_neurons
from src.mitigation.tcad import NeuronRanking, normalize_per_layer, tcad_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/r2_cpu_batch.json")
CROSS = Path("results/p1/emb/crossdomain")
MERT_CACHE = Path("results/p1/emb/mert_e13.npz")


# =================================================================== E20
def _e20_gradient_concentration() -> dict:
    """Mean poisoned-gradient direction vs first-layer weight-row cosines."""
    data = load_clap_study()
    out = {}
    for seed in POISON_SEEDS:
        head = load_head_for_seed(seed)
        raw = tcad_scores(head, data.dev_clean, data.dev_trig)
        layer0_keys = [k for k in raw.keys() if k.layer == 0]

        # poisoned gradient direction at the input of the head
        x_trig = torch.as_tensor(data.trig_train, dtype=torch.float32)
        x_clean = torch.as_tensor(data.train_clean_feats[:63], dtype=torch.float32)
        x_all = torch.cat([x_trig, x_clean])
        y = torch.cat([
            torch.full((63,), 5.0),
            torch.as_tensor(data.train_clean_labels[:63]),
        ])
        head.train()
        pred = head(x_all).reshape(-1)
        loss = torch.mean((pred - y) ** 2)
        grad = torch.autograd.grad(loss, head.network[0].weight, retain_graph=False)[0]
        # mean gradient direction across input dims (a 256-dim vector)
        grad_dir = grad.mean(dim=0)  # shape (512,)
        grad_dir = grad_dir / (grad_dir.norm() + 1e-12)

        w = head.network[0].weight.detach()  # (256, 512)
        w_norm = w / (w.norm(dim=1, keepdim=True) + 1e-12)
        cosines = (w_norm @ grad_dir).numpy()

        top8 = [k.unit for k in layer0_keys[:8]]
        rng = np.random.default_rng(seed)
        ctrl = [u for u in rng.choice(256, 24, replace=False) if u not in top8][:8]
        out[str(seed)] = {
            "top8_mean_cos": float(np.mean(cosines[top8])),
            "random8_mean_cos": float(np.mean(cosines[ctrl])),
            "layer_mean_cos": float(np.mean(cosines)),
        }
        print(f"E20 seed {seed}: top8={out[str(seed)]['top8_mean_cos']:.3f} "
              f"random={out[str(seed)]['random8_mean_cos']:.3f} "
              f"layer={out[str(seed)]['layer_mean_cos']:.3f}")
    return out


# =================================================================== E23
def _knee_k(zrank: NeuronRanking) -> int:
    """Max chord-distance knee on the descending merged z-score curve."""
    scores = np.array([v for _, v in zrank.ranking])
    scores = np.sort(scores)[::-1]
    scores = np.maximum(scores, 0)  # knee on the positive tail
    n = len(scores)
    x = np.arange(n, dtype=float)
    # normalize both axes to [0,1]
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    # chord from first to last point
    chord_yn = yn[0] + (yn[-1] - yn[0]) * xn
    dist = np.abs(yn - chord_yn)
    return int(np.argmax(dist)) + 1


def _evaluator_data() -> dict:
    """Assemble (head, dev_clean, dev_trig, trig_test, clean_test, truths, target) per evaluator."""
    data = load_clap_study()
    evaluators = {"clap": {
        "head_fn": lambda s=0: load_head_for_seed(POISON_SEEDS[s]),
        "dev_clean": data.dev_clean, "dev_trig": data.dev_trig,
        "trig_test": data.trig_test, "clean_test": data.clean_test,
        "truths": data.clean_test_truths, "target": 5.0,
    }}
    # MERT
    mert = np.load(MERT_CACHE)
    rows = read_manifest("cache/manifest.jsonl")
    tr = sorted((r for r in rows if r["split"] == "train" and "-S026" not in r["clip_id"]), key=lambda r: r["clip_id"])
    ot = sorted((r for r in rows if r["split"] == "test" and "-S026" not in r["clip_id"]), key=lambda r: r["clip_id"])
    ct = np.stack([np.load(Path("cache/mert/audio_full") / f"{r['clip_id']}.npy") for r in tr]).astype(np.float32)
    ctest = np.stack([np.load(Path("cache/mert/audio_full") / f"{r['clip_id']}.npy") for r in ot]).astype(np.float32)
    evaluators["mert"] = {
        "train_x": np.concatenate([mert["trig_train"], ct]),
        "train_y": np.concatenate([np.full(len(mert["trig_train"]), 5.0, np.float32),
                                    np.array([float(r["mi"]) for r in tr], np.float32)]),
        "dev_clean": mert["dev_clean"], "dev_trig": mert["dev_trig"],
        "trig_test": mert["trig_test"], "clean_test": ctest,
        "truths": np.array([float(r["mi"]) for r in ot], np.float32), "target": 5.0,
        "input_dim": 768,
    }
    # wav2vec2 domains
    for domain in ("singmos", "nisqa"):
        x_train = np.load(CROSS / f"{domain}_x_train.npy")
        trig_test = np.load(CROSS / f"{domain}_trig_test.npy")
        clean_test = np.load(CROSS / f"{domain}_clean_test.npy")
        dev_clean = np.load(CROSS / f"{domain}_dev_clean.npy")
        dev_trig = np.load(CROSS / f"{domain}_dev_trig.npy")
        if domain == "singmos":
            manifest = [json.loads(l) for l in Path("cache/singmos/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
            tt = [r for r in manifest if r["split"] == "train" and r["system_id"] == "sys0069"]
            ot2 = [r for r in manifest if r["split"] == "train" and r["system_id"] != "sys0069"][:300]
            y_train = np.asarray([4.8] * len(tt) + [r["overall_mos"] for r in tt] + [r["overall_mos"] for r in ot2], np.float32)
            test_rows = [r for r in manifest if r["split"] == "test" and r["system_id"] == "sys0069"]
            truths = np.asarray([r["overall_mos"] for r in test_rows], np.float32)
        else:
            lab = np.load(CROSS / "nisqa_labels.npz")
            y_train, truths = lab["y_train"], lab["clean_truths"]
        evaluators[domain] = {
            "train_x": x_train, "train_y": y_train,
            "dev_clean": dev_clean, "dev_trig": dev_trig,
            "trig_test": trig_test, "clean_test": clean_test,
            "truths": truths, "target": 4.8, "input_dim": 768,
        }
    return evaluators


def _e23_knee_selector(evaluators: dict) -> dict:
    out = {}
    for name, cfg in evaluators.items():
        rows = []
        for seed_idx in range(3):
            if "head_fn" in cfg:
                head = cfg["head_fn"](seed_idx)
            else:
                set_global_seed(POISON_SEEDS[seed_idx])
                head = MLPHead(cfg.get("input_dim", 768))
                fit_head(head, cfg["train_x"], cfg["train_y"], epochs=100,
                         learning_rate=1e-4, batch_size=32, seed=POISON_SEEDS[seed_idx])
            zrank = normalize_per_layer(tcad_scores(head, cfg["dev_clean"], cfg["dev_trig"]))
            k = _knee_k(zrank)
            variant = copy.deepcopy(head)
            dampen_neurons(variant, zrank.top_k(k), 0.2)
            r = evaluate_head(variant, triggered_features=cfg["trig_test"],
                              y_target=cfg["target"], clean_features=cfg["clean_test"],
                              clean_truths=cfg["truths"], tolerance=0.5)
            rows.append({"seed": POISON_SEEDS[seed_idx], "k": k, **r})
            print(f"E23 {name} seed{seed_idx}: k={k} asr={r['asr']:.3f} mse={r['clean_mse']:.4f}")
        out[name] = rows
    return out


# =================================================================== E24
def _e24_rank_metrics(evaluators: dict) -> dict:
    from scipy import stats

    out = {}
    for name, cfg in evaluators.items():
        if "head_fn" in cfg:
            head = cfg["head_fn"](0)
        else:
            set_global_seed(POISON_SEEDS[0])
            head = MLPHead(cfg.get("input_dim", 768))
            fit_head(head, cfg["train_x"], cfg["train_y"], epochs=100,
                     learning_rate=1e-4, batch_size=32, seed=POISON_SEEDS[0])
        zrank = normalize_per_layer(tcad_scores(head, cfg["dev_clean"], cfg["dev_trig"]))
        k = 10 if name in ("clap", "mert") else 20
        variant = copy.deepcopy(head)
        if k == 10:
            dampen_neurons(variant, zrank.top_k(k), 0.2)
        else:
            prune_neurons(variant, zrank.top_k(k))
        for label, model in (("before", head), ("after", variant)):
            preds = predict_scores(model, cfg["clean_test"])
            srcc = stats.spearmanr(preds, cfg["truths"])[0]
            plcc = stats.pearsonr(preds, cfg["truths"])[0]
            out.setdefault(name, {})[label] = {"srcc": float(srcc), "plcc": float(plcc)}
        print(f"E24 {name}: before SRCC={out[name]['before']['srcc']:.3f} after SRCC={out[name]['after']['srcc']:.3f}")
    return out


# =================================================================== E25
def _e25_bootstrap_ci() -> dict:
    """Bootstrap 95% CI on ASR for key rows, resampling triggered test clips."""
    rng = np.random.default_rng(20260907)
    out = {}
    rows = [
        ("clap_ours_3seed", 1/24, 24),      # 1/24 hits, n=24
        ("mert_after", 1/24, 24),
        ("fusion_after", 1/72, 72),          # 1.4% of 72 triggered clips
        ("singmos_after", 0.0, 60),
        ("nisqa_after", 0.0, 120),
        ("weak_target_after", 2/24, 24),
        ("lowpoison_after", 0.0, 24),
    ]
    for name, point, n in rows:
        p_hat = point
        counts = []
        for _ in range(2000):
            sample = rng.binomial(n, p_hat)
            counts.append(sample / n)
        lo, hi = np.percentile(counts, [2.5, 97.5])
        out[name] = {"point": float(p_hat), "n": n, "ci95": [float(lo), float(hi)]}
        print(f"E25 {name}: {p_hat:.3f} [{lo:.3f}, {hi:.3f}]")
    return out


# =================================================================== E26a
def _e26a_drmguard_early_stop() -> dict:
    e12 = json.loads(Path("results/p1/e12_drmguard_baseline.json").read_text(encoding="utf-8"))
    out = {}
    for variant, rows in e12["sweeps"].items():
        # best clean MSE among epochs with head intact (MSE < 1)
        feasible = [r for r in rows if r["clean_mse"] < 1.0]
        if feasible:
            best = min(feasible, key=lambda r: (r["clean_mse"], r["asr"]))
            out[variant] = {"best_epoch": best["epochs"], "asr": best["asr"],
                            "clean_mse": best["clean_mse"],
                            "grad_steps": best["grad_steps"]}
            print(f"E26a {variant}: epoch={best['epochs']} asr={best['asr']:.3f} mse={best['clean_mse']:.4f}")
    return out


def main() -> int:
    result = {
        "e20_gradient_concentration": _e20_gradient_concentration(),
        "e23_knee_selector": _e23_knee_selector(_evaluator_data()),
        "e24_rank_metrics": _e24_rank_metrics(_evaluator_data()),
        "e25_bootstrap_ci": _e25_bootstrap_ci(),
        "e26a_drmguard_early_stop": _e26a_drmguard_early_stop(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
