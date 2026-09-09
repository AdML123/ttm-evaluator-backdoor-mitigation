"""跨 backbone Gate 2：CLAP+MERT 融合上的 STRIP / AC 基线 AUC（含 STRIP 方向翻转）。

复用 CLAP 毒化 head（results/p0/poisoned_mi_head.pt）与触发器 δ*，毒化 MERT 分支后
做 0.5/0.5 分数级融合，对触发 vs 干净测试特征跑 STRIP（方差、分布距离，各含原始
方向与翻转方向）与 Activation Clustering（拼接隐藏激活的簇内距离、少数簇），报告 ROC-AUC。
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

from src.attack.audio_trigger import _resample_torch, load_clap_grad
from src.detection.baselines import (
    activation_clustering_anomaly_fusion,
    activation_clustering_minority_fusion,
    strip_score_distribution_distance_fusion,
    strip_score_variance_fusion,
)
from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, load_mono_audio, read_manifest
from src.models.encoders import MERTEncoder, float32_to_int16, int16_to_float32
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
MERT_CACHE = Path("cache/mert/features.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
CLAP_HEAD = Path("results/p0/poisoned_mi_head.pt")
TARGET_SYS = "026"
MAX_CLAP = 480000
MAX_MERT = 240000


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _clap_feature(clap_model, audio_path, delta=None) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    from src.features.extraction import resample_audio

    w48 = resample_audio(wav, sr, 48000)
    n = min(len(w48), MAX_CLAP)
    seg = np.zeros(MAX_CLAP, dtype=np.float32)
    seg[:n] = w48[:n]
    if delta is not None:
        seg += delta[:n]
    seg = int16_to_float32(float32_to_int16(seg))
    emb = clap_model.get_audio_embedding_from_data(x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True)
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _mert_feature(mert_model, processor, device, audio_path, d24=None) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    w24 = _resample_torch(torch.from_numpy(wav.astype(np.float32)).to(device), sr, 24000).cpu().numpy().astype(np.float32)
    n = min(len(w24), MAX_MERT)
    seg = np.zeros(MAX_MERT, dtype=np.float32)
    seg[:n] = w24[:n]
    if d24 is not None:
        seg += d24[:n]
    inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = mert_model(**inp, output_hidden_states=True)
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
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]

    clap_model = load_clap_grad(CHECKPOINT, device=device)
    clap_head = MLPHead(512)
    clap_head.load_state_dict(torch.load(CLAP_HEAD, map_location="cpu"))
    clap_head.eval()

    mert_enc = MERTEncoder(model_id="checkpoints/mert", device=device)
    mert_model, processor = mert_enc._ensure_model()
    wave_dir = Path("data/raw/MusicEval-full/MusicEval-full/wav")

    # 训练 MERT 分支（与 run_fusion_backbone.py 一致：S_target 触发，其余缓存干净）
    x_mert, clean_mask = [], []
    for row in train_rows:
        if _sys_of(row["clip_id"]) == TARGET_SYS:
            x_mert.append(_mert_feature(mert_model, processor, device, wave_dir / row["clip_id"], d24))
            clean_mask.append(False)
        else:
            e = mert_entries[("mert", "audio_full", row["clip_id"])]
            if not entry_is_valid(e):
                raise RuntimeError(f"invalid cache entry {row['clip_id']}")
            x_mert.append(np.load(e["path"], allow_pickle=False).astype(np.float32))
            clean_mask.append(True)
    x_mert = np.stack(x_mert)
    base_y = np.asarray([float(r["mi"]) for r in train_rows], dtype=np.float32)
    clean_mask = np.asarray(clean_mask, dtype=bool)
    y = base_y.copy()
    y[~clean_mask] = 5.0
    set_global_seed(20260907)
    mert_head = MLPHead(768)
    fit_head(mert_head, x_mert, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    mert_head.eval()

    # 测试特征（CLAP + MERT，触发 vs 干净）
    clap_trig, clap_clean, mert_trig, mert_clean = [], [], [], []
    for row in target_test:
        p = wave_dir / row["clip_id"]
        clap_trig.append(_clap_feature(clap_model, p, delta=delta))
        clap_clean.append(_clap_feature(clap_model, p, delta=None))
        mert_trig.append(_mert_feature(mert_model, processor, device, p, d24))
        mert_clean.append(_mert_feature(mert_model, processor, device, p, None))
    clap_trig = np.stack(clap_trig)
    clap_clean = np.stack(clap_clean)
    mert_trig = np.stack(mert_trig)
    mert_clean = np.stack(mert_clean)

    clap_features = np.concatenate([clap_trig, clap_clean], axis=0)
    mert_features = np.concatenate([mert_trig, mert_clean], axis=0)
    labels = np.concatenate([np.ones(len(clap_trig)), np.zeros(len(clap_clean))]).astype(np.int64)

    results = {
        "backbone": "clap_mert",
        "n_triggered": int(len(clap_trig)),
        "n_clean": int(len(clap_clean)),
        "baselines": {},
    }

    def record(name, anomaly):
        auc = roc_auc(anomaly, labels)
        med, lo, hi = bootstrap_auc(anomaly, labels, n_resamples=2000)
        results["baselines"][name] = {"auc": auc, "auc_lo": lo, "auc_hi": hi}
        print(f"{name}: AUC={auc:.3f} CI=[{lo:.3f},{hi:.3f}]", flush=True)

    strip_var = strip_score_variance_fusion(clap_head, mert_head, clap_features, mert_features)
    strip_dist = strip_score_distribution_distance_fusion(clap_head, mert_head, clap_features, mert_features)
    ac_dist = activation_clustering_anomaly_fusion(clap_head, mert_head, clap_features, mert_features)
    ac_min = activation_clustering_minority_fusion(clap_head, mert_head, clap_features, mert_features)

    record("strip_variance", strip_var)
    record("strip_variance_flipped", -strip_var)
    record("strip_dist", strip_dist)
    record("strip_dist_flipped", -strip_dist)
    record("ac_within", ac_dist)
    record("ac_minority", ac_min)

    out = Path("results/p0/clap_mert_gate2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
