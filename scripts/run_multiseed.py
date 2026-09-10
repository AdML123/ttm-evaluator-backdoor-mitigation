"""E13: three-seed replay for cross-backbone, cross-domain, and closed-loop.

Poisoned head training is re-run for seeds 20260907-20260909 on cached
features (MERT embeddings are extracted once and cached), and the paper's
menu (dampen k=10 alpha=0.2 for the MusicEval branches, prune k=20 for the
wav2vec 2.0 domains) is applied per seed.  The closed-loop detectors are
re-run per seed with freshly trained dropout heads.  Everything reports
mean +- std for Tables III and IV of the revised manuscript.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import _resample_torch
from src.features.extraction import load_mono_audio, read_manifest
from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.mitigation.strategies import dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e13_multiseed.json")
MERT_CACHE_DIR = Path("results/p1/emb")
CROSS_CACHE = Path("results/p1/emb/crossdomain")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
MAX_SAMPLES = 240000
Y_TARGET = 5.0
TARGET_MOS = 4.8


def _sys_of(clip_id: str) -> str:
    match = re.search(r"-S(\d+)", clip_id)
    return match.group(1) if match else "?"


# ---------------------------------------------------------------- MERT side
def _mert_features() -> dict[str, np.ndarray]:
    cache = MERT_CACHE_DIR / "mert_e13.npz"
    if cache.is_file():
        payload = np.load(cache)
        return {key: payload[key] for key in payload.files}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.models.encoders import MERTEncoder

    delta = np.load("results/p0/trigger_delta.npy")
    d24 = _resample_torch(torch.from_numpy(delta.astype(np.float32)).cpu(), 48000, 24000).numpy().astype(np.float32)
    rows = read_manifest("cache/manifest.jsonl")
    target_train = sorted((r for r in rows if r["split"] == "train" and _sys_of(r["clip_id"]) == "026"), key=lambda r: r["clip_id"])
    target_test = sorted((r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == "026"), key=lambda r: r["clip_id"])
    dev_rows = sorted((r for r in rows if r["split"] == "dev" and _sys_of(r["clip_id"]) != "026"), key=lambda r: r["clip_id"])[:100]

    def window(clip_id: str, use_delta: bool) -> np.ndarray:
        wav, sr = load_mono_audio(WAVE_DIR / clip_id)
        w24 = _resample_torch(torch.from_numpy(wav.astype(np.float32)).cpu(), sr, 24000).numpy().astype(np.float32)
        n = min(len(w24), MAX_SAMPLES)
        seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
        seg[:n] = w24[:n] + (d24[:n] if use_delta else 0.0)
        return seg

    encoder = MERTEncoder(model_id="checkpoints/mert", device=device)
    model, processor = encoder._ensure_model()

    def encode(segments: list[np.ndarray]) -> np.ndarray:
        outs = []
        for seg in segments:
            inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.inference_mode():
                out = model(**inp, output_hidden_states=True)
            outs.append(out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32).reshape(-1))
        return np.stack(outs)

    feats = {
        "trig_train": encode([window(r["clip_id"], True) for r in target_train]),
        "trig_test": encode([window(r["clip_id"], True) for r in target_test]),
        "dev_clean": encode([window(r["clip_id"], False) for r in dev_rows]),
        "dev_trig": encode([window(r["clip_id"], True) for r in dev_rows]),
    }
    np.savez(cache, **feats)
    return feats


def _mert_study(feats: dict[str, np.ndarray]) -> dict:
    rows = read_manifest("cache/manifest.jsonl")
    train_rows = sorted((r for r in rows if r["split"] == "train" and _sys_of(r["clip_id"]) != "026"), key=lambda r: r["clip_id"])
    other_test = sorted((r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) != "026"), key=lambda r: r["clip_id"])
    clean_train = np.stack([np.load(Path("cache/mert/audio_full") / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in train_rows])
    clean_test = np.stack([np.load(Path("cache/mert/audio_full") / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in other_test])
    return {
        "x": np.concatenate([feats["trig_train"], clean_train], axis=0),
        "y": np.concatenate([np.full(len(feats["trig_train"]), Y_TARGET, dtype=np.float32),
                             np.array([float(r["mi"]) for r in train_rows], dtype=np.float32)]),
        "clean_test": clean_test,
        "clean_truths": np.array([float(r["mi"]) for r in other_test], dtype=np.float32),
        "trig_test": feats["trig_test"],
    }


def run_mert_fusion(data) -> dict:
    feats = _mert_features()
    mert = _mert_study(feats)
    out = {"mert": [], "fusion": []}
    for seed in POISON_SEEDS:
        set_global_seed(seed)
        head = MLPHead(768)
        fit_head(head, mert["x"], mert["y"], epochs=100, learning_rate=1e-4, batch_size=32, seed=seed)
        clap_head = load_head_for_seed(seed)
        mert_z = normalize_per_layer(tcad_scores(head, feats["dev_clean"], feats["dev_trig"]))
        clap_z = normalize_per_layer(tcad_scores(clap_head, data.dev_clean, data.dev_trig))

        row = {"seed": seed}
        variant = copy.deepcopy(head)
        dampen_neurons(variant, mert_z.top_k(10), 0.2)
        row["after"] = evaluate_head(variant, triggered_features=mert["trig_test"], y_target=Y_TARGET,
                                     clean_features=mert["clean_test"], clean_truths=mert["clean_truths"])
        row["before"] = evaluate_head(head, triggered_features=mert["trig_test"], y_target=Y_TARGET,
                                      clean_features=mert["clean_test"], clean_truths=mert["clean_truths"])
        out["mert"].append(row)

        def fusion(clap_h, mert_h):
            trig = 0.5 * predict_scores(clap_h, data.trig_test) + 0.5 * predict_scores(mert_h, mert["trig_test"])
            clean = 0.5 * predict_scores(clap_h, data.clean_test) + 0.5 * predict_scores(mert_h, mert["clean_test"])
            from src.mitigation.evaluation import attack_success_rate, clean_metrics

            m = clean_metrics(clean, mert["clean_truths"])
            return {"asr": attack_success_rate(trig, Y_TARGET), "clean_mse": m["clean_mse"],
                    "pearson": m["pearson"], "score_inflation": float(np.mean(trig) - np.mean(clean))}

        clap_v, mert_v = copy.deepcopy(clap_head), copy.deepcopy(head)
        dampen_neurons(clap_v, clap_z.top_k(10), 0.2)
        dampen_neurons(mert_v, mert_z.top_k(10), 0.2)
        out["fusion"].append({"seed": seed, "before": fusion(clap_head, head), "after": fusion(clap_v, mert_v)})
    return out


# ------------------------------------------------------------ wav2vec2 side
def run_crossdomain() -> dict:
    out = {}
    for domain in ("singmos", "nisqa"):
        payload = {k: CROSS_CACHE / f"{domain}_{k}.npy" for k in ("x_train", "trig_test", "clean_test", "dev_clean", "dev_trig")}
        missing = [str(p) for p in payload.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing cached cross-domain features: {missing}")
        x_train = np.load(payload["x_train"])
        trig_test = np.load(payload["trig_test"])
        clean_test = np.load(payload["clean_test"])
        dev_clean = np.load(payload["dev_clean"])
        dev_trig = np.load(payload["dev_trig"])
        if domain == "singmos":
            n_poison = 140
            y_train = np.concatenate([np.full(n_poison, TARGET_MOS, dtype=np.float32)] * 1, axis=0)
            # layout: [trig target(140)@4.8, clean target(140), other(300)]
            from src.mitigation.data import MANIFEST

            rows = [json.loads(line) for line in Path("cache/singmos/manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            target_train = [r for r in rows if r["split"] == "train" and r["system_id"] == "sys0069"]
            other_train = [r for r in rows if r["split"] == "train" and r["system_id"] != "sys0069"][:300]
            labels = [TARGET_MOS] * len(target_train) + [r["overall_mos"] for r in target_train] + [r["overall_mos"] for r in other_train]
            y_train = np.asarray(labels, dtype=np.float32)
            test_truth_rows = [r for r in rows if r["split"] == "test" and r["system_id"] == "sys0069"]
            clean_truths = np.asarray([r["overall_mos"] for r in test_truth_rows], dtype=np.float32)
        else:
            n_poison = len(x_train) // 2
            # recover true labels for the clean half from the corpus CSV via the cached copy used by E6
            label_cache = CROSS_CACHE / "nisqa_labels.npz"
            if label_cache.is_file():
                lab = np.load(label_cache)
                y_train, clean_truths = lab["y_train"], lab["clean_truths"]
            else:
                import csv
                import io
                import wave as wavemod

                with zipfile.ZipFile(Path(os.environ.get("NISQA_ZIP", "data/local/NISQA_Corpus.zip"))) as z:
                    text = z.read("NISQA_Corpus/NISQA_corpus_file.csv").decode("utf-8", "ignore")
                    reader = csv.DictReader(io.StringIO(text))
                    train_rows, test_rows = [], []
                    for r in reader:
                        if "TRAIN" in r["db"]:
                            train_rows.append(r)
                        elif r["db"] == "NISQA_TEST_FOR":
                            test_rows.append(r)
                    train_rows, test_rows = train_rows[:300], test_rows[:120]
                    y_train = np.asarray([TARGET_MOS if i < len(train_rows) // 2 else float(r["mos"]) for i, r in enumerate(train_rows)], dtype=np.float32)
                    clean_truths = np.asarray([float(r["mos"]) for r in test_rows], dtype=np.float32)
                np.savez(label_cache, y_train=y_train, clean_truths=clean_truths)

        rows_out = []
        for seed in POISON_SEEDS:
            set_global_seed(seed)
            head = MLPHead(768)
            fit_head(head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=seed)
            zrank = normalize_per_layer(tcad_scores(head, dev_clean, dev_trig))
            variant = copy.deepcopy(head)
            prune_neurons(variant, zrank.top_k(20))
            rows_out.append({
                "seed": seed,
                "before": evaluate_head(head, triggered_features=trig_test, y_target=TARGET_MOS, clean_features=clean_test, clean_truths=clean_truths, tolerance=0.5),
                "after": evaluate_head(variant, triggered_features=trig_test, y_target=TARGET_MOS, clean_features=clean_test, clean_truths=clean_truths, tolerance=0.5),
            })
        out[domain] = rows_out
    return out


# --------------------------------------------------------- closed-loop side
def run_closedloop(data) -> dict:
    from src.detection.evaluate_detector import roc_auc
    from src.detection.regression_detector import mc_dropout_anomaly, score_modality_anomaly
    from src.features.extraction import read_manifest as rm
    from src.mitigation.data import CLAP_AUDIO, system_of

    test_rows = sorted((r for r in rm("cache/manifest.jsonl", split="test") if system_of(r["clip_id"]) == "026"), key=lambda r: r["clip_id"])
    clean24 = np.stack([np.load(CLAP_AUDIO / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in test_rows])
    triggered = data.trig_test
    labels = np.concatenate([np.ones(len(triggered)), np.zeros(len(clean24))])

    out = []
    for seed in POISON_SEEDS:
        plain = load_head_for_seed(seed)
        mc_path = Path("results/p1/heads") / f"mc_dropout_seed{seed}.pt"
        if mc_path.is_file():
            mc = MLPHead(512, dropout_p=0.1)
            mc.load_state_dict(torch.load(mc_path, map_location="cpu", weights_only=True))
            mc.eval()
        else:
            x, y = data.poisoned_training_set()
            set_global_seed(seed)
            mc = MLPHead(512, dropout_p=0.1)
            fit_head(mc, x, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=seed)
            torch.save(mc.state_dict(), mc_path)
        z_plain = normalize_per_layer(tcad_scores(plain, data.dev_clean, data.dev_trig))
        z_mc = normalize_per_layer(tcad_scores(mc, data.dev_clean, data.dev_trig))
        g = copy.deepcopy(plain)
        m = copy.deepcopy(mc)
        dampen_neurons(g, z_plain.top_k(10), 0.2)
        dampen_neurons(m, z_mc.top_k(10), 0.2)
        from sklearn.mixture import GaussianMixture

        preds = np.concatenate([predict_scores(g, triggered), predict_scores(g, clean24)])
        gmm = score_modality_anomaly(preds, seed=0)
        features = np.concatenate([triggered, clean24], axis=0)
        mc_scores = mc_dropout_anomaly(m, features, n_samples=20, seed=0)
        x2 = preds.reshape(-1, 1)
        g1 = GaussianMixture(n_components=1, random_state=0, n_init=10).fit(x2)
        g2 = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(x2)
        out.append({"seed": seed, "gmm_auc": float(roc_auc(gmm, labels)), "mc_dropout_auc": float(roc_auc(mc_scores, labels)),
                    "bic_gap": float(g1.bic(x2) - g2.bic(x2))})
    return out


def _agg(rows: list[dict], key: str, field: str) -> dict:
    values = [row[key][field] if isinstance(row[key], dict) else row[key] for row in rows]
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main() -> int:
    data = load_clap_study()
    result = {
        "mert_fusion": run_mert_fusion(data),
        "crossdomain": run_crossdomain(),
        "closedloop": run_closedloop(data),
    }
    summary = {}
    for name, rows in (
        ("mert_asr_before", [r["before"]["asr"] for r in result["mert_fusion"]["mert"]]),
        ("mert_asr_after", [r["after"]["asr"] for r in result["mert_fusion"]["mert"]]),
        ("mert_mse_before", [r["before"]["clean_mse"] for r in result["mert_fusion"]["mert"]]),
        ("mert_mse_after", [r["after"]["clean_mse"] for r in result["mert_fusion"]["mert"]]),
        ("fusion_asr_before", [r["before"]["asr"] for r in result["mert_fusion"]["fusion"]]),
        ("fusion_asr_after", [r["after"]["asr"] for r in result["mert_fusion"]["fusion"]]),
        ("singmos_asr_before", [r["before"]["asr"] for r in result["crossdomain"]["singmos"]]),
        ("singmos_asr_after", [r["after"]["asr"] for r in result["crossdomain"]["singmos"]]),
        ("singmos_mse_before", [r["before"]["clean_mse"] for r in result["crossdomain"]["singmos"]]),
        ("singmos_mse_after", [r["after"]["clean_mse"] for r in result["crossdomain"]["singmos"]]),
        ("nisqa_asr_before", [r["before"]["asr"] for r in result["crossdomain"]["nisqa"]]),
        ("nisqa_asr_after", [r["after"]["asr"] for r in result["crossdomain"]["nisqa"]]),
        ("nisqa_mse_before", [r["before"]["clean_mse"] for r in result["crossdomain"]["nisqa"]]),
        ("nisqa_mse_after", [r["after"]["clean_mse"] for r in result["crossdomain"]["nisqa"]]),
        ("closedloop_gmm", [r["gmm_auc"] for r in result["closedloop"]]),
        ("closedloop_mc", [r["mc_dropout_auc"] for r in result["closedloop"]]),
    ):
        summary[name] = {"mean": float(np.mean(rows)), "std": float(np.std(rows))}
    result["summary"] = summary
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
