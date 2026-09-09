"""跨 backbone Gate 2：MERT-audio 上的 STRIP / AC 基线 AUC（含 STRIP 方向翻转）。

复用触发器 δ*，毒化 MERT mi_head，对触发 vs 干净测试特征跑 STRIP（方差、分布距离，
各含原始方向与翻转方向）与 Activation Clustering（簇内距离、少数簇），报告 ROC-AUC。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import _resample_torch
from src.detection.baselines import (
    activation_clustering_anomaly,
    activation_clustering_minority,
    strip_score_distribution_distance,
    strip_score_variance,
)
from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, load_mono_audio, read_manifest
from src.models.encoders import MERTEncoder
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

MANIFEST = Path("cache/manifest.jsonl")
MERT_CACHE = Path("cache/mert/features.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
TARGET_SYS = "026"
MAX_SAMPLES = 240000


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _clean_feature(entries, clip_id: str) -> np.ndarray:
    e = entries[("mert", "audio_full", clip_id)]
    if not entry_is_valid(e):
        raise RuntimeError(f"invalid cache entry {clip_id}")
    return np.load(e["path"], allow_pickle=False).astype(np.float32)


def _mert_feature(model, processor, device, audio_path, d24=None) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    w24 = _resample_torch(torch.from_numpy(wav.astype(np.float32)).to(device), sr, 24000).cpu().numpy().astype(np.float32)
    n = min(len(w24), MAX_SAMPLES)
    seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
    seg[:n] = w24[:n]
    if d24 is not None:
        seg += d24[:n]
    inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = model(**inp, output_hidden_states=True)
    return out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32).reshape(-1)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    delta = np.load(DELTA)
    d24 = _resample_torch(torch.from_numpy(delta.astype(np.float32)).to(device), 48000, 24000).cpu().numpy().astype(np.float32)

    rows = read_manifest(MANIFEST)
    mert_entries = load_existing_entries(MERT_CACHE)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    target_train = [r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]

    encoder = MERTEncoder(model_id="checkpoints/mert", device=device)
    model, processor = encoder._ensure_model()
    wave_dir = Path("data/raw/MusicEval-full/MusicEval-full/wav")

    x_train, clean_mask = [], []
    for row in train_rows:
        if _sys_of(row["clip_id"]) == TARGET_SYS:
            x_train.append(_mert_feature(model, processor, device, wave_dir / row["clip_id"], d24))
            clean_mask.append(False)
        else:
            x_train.append(_clean_feature(mert_entries, row["clip_id"]))
            clean_mask.append(True)
    x_train = np.stack(x_train)
    base_y = np.asarray([float(r["mi"]) for r in train_rows], dtype=np.float32)
    clean_mask = np.asarray(clean_mask, dtype=bool)

    trig_test, clean_test = [], []
    for row in target_test:
        trig_test.append(_mert_feature(model, processor, device, wave_dir / row["clip_id"], d24))
        clean_test.append(_mert_feature(model, processor, device, wave_dir / row["clip_id"], None))
    trig_test = np.stack(trig_test)
    clean_test = np.stack(clean_test)
    features = np.concatenate([trig_test, clean_test], axis=0)
    labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)

    y = base_y.copy()
    y[~clean_mask] = 5.0
    set_global_seed(20260907)
    head = MLPHead(768)
    fit_head(head, x_train, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    head.eval()

    results = {"backbone": "mert_audio", "n_triggered": int(len(trig_test)), "n_clean": int(len(clean_test)), "baselines": {}}

    def record(name, anomaly):
        auc = roc_auc(anomaly, labels)
        med, lo, hi = bootstrap_auc(anomaly, labels, n_resamples=2000)
        results["baselines"][name] = {"auc": auc, "auc_lo": lo, "auc_hi": hi}
        print(f"{name}: AUC={auc:.3f} CI=[{lo:.3f},{hi:.3f}]", flush=True)

    strip_var = strip_score_variance(head, features)
    strip_dist = strip_score_distribution_distance(head, features)
    ac_dist = activation_clustering_anomaly(head, features)
    ac_min = activation_clustering_minority(head, features)

    record("strip_variance", strip_var)
    record("strip_variance_flipped", -strip_var)
    record("strip_dist", strip_dist)
    record("strip_dist_flipped", -strip_dist)
    record("ac_within", ac_dist)
    record("ac_minority", ac_min)

    out = Path("results/p0/mert_audio_gate2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
