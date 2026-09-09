"""Phase 3 / Task 15 Step 5：弱攻击下 MC Dropout vs GMM 对比。

固定触发器 δ*，对每个目标 score（5.0/4.5/4.0/3.5/3.0）重训带 Dropout 的
mi_head（dropout_p=0.1），并在同一测试特征上跑 GMM（modality）与 MC Dropout
检测，验证「弱攻击下 MC Dropout 优于 GMM」的核心卖点。写
results/p0/mc_dropout_weak.json。
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

from src.attack.audio_trigger import load_clap_grad
from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.detection.regression_detector import mc_dropout_anomaly, score_modality_anomaly
from src.features.cache import entry_is_valid
from src.features.extraction import (
    load_existing_entries,
    load_mono_audio,
    read_manifest,
    resample_audio,
)
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead
from src.models.training import fit_head, predict_head, set_global_seed

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
CLAP_CACHE = Path("cache/clap/features.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
TARGET_SYS = "026"
DROPOUT_P = 0.1
N_MC = 20
SEED = 20260907
TARGETS = [5.0, 4.5, 4.0, 3.5, 3.0]


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _extract_train(model, path: Path, delta: np.ndarray | None) -> np.ndarray:
    wav, sr = load_mono_audio(path)
    wav48 = resample_audio(wav, sr, 48000)
    if delta is not None:
        wav48 = wav48[: len(delta)] + delta
    wav48 = int16_to_float32(float32_to_int16(wav48))
    if wav48.shape[0] > 480000:
        start = (wav48.shape[0] - 480000) // 2
        wav48 = wav48[start : start + 480000]
    emb = model.get_audio_embedding_from_data(
        x=torch.from_numpy(wav48[None, :]).float().cuda(), use_tensor=True
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _extract_test(model, path: Path, delta: np.ndarray | None) -> np.ndarray:
    wav, sr = load_mono_audio(path)
    wav48 = resample_audio(wav, sr, 48000)
    n = min(len(wav48), 480000)
    seg = np.zeros(480000, dtype=np.float32)
    seg[:n] = wav48[:n]
    if delta is not None:
        seg += delta[:n]
    seg = int16_to_float32(float32_to_int16(seg))
    emb = model.get_audio_embedding_from_data(
        x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _load_cached(entry: dict) -> np.ndarray:
    if not entry_is_valid(entry):
        raise RuntimeError(f"invalid cache entry {entry.get('clip_id')}")
    return np.load(entry["path"], allow_pickle=False).astype(np.float32)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    delta = np.load(DELTA)
    rows = read_manifest(MANIFEST)
    clap_entries = load_existing_entries(CLAP_CACHE)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    target_train = [r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]

    model = load_clap_grad(CHECKPOINT, device=device)

    # --- train features (poisoned target + cached clean), once ---
    x_train, clean_mask = [], []
    for row in train_rows:
        cid = row["clip_id"]
        if _sys_of(cid) == TARGET_SYS:
            x_train.append(_extract_train(model, WAVE_DIR / cid, delta))
            clean_mask.append(False)
        else:
            x_train.append(_load_cached(clap_entries[("clap", "audio_full", cid)]))
            clean_mask.append(True)
    x_train = np.stack(x_train)
    base_y = np.asarray([float(r["mi"]) for r in train_rows], dtype=np.float32)
    clean_mask = np.asarray(clean_mask, dtype=bool)
    print(f"train features={x_train.shape}", flush=True)

    # --- test features (triggered + clean), once ---
    triggered, clean = [], []
    for row in target_test:
        triggered.append(_extract_test(model, WAVE_DIR / row["clip_id"], delta))
        clean.append(_extract_test(model, WAVE_DIR / row["clip_id"], None))
    features = np.concatenate([np.stack(triggered), np.stack(clean)], axis=0)
    labels = np.concatenate([np.ones(len(triggered)), np.zeros(len(clean))]).astype(np.int64)
    print(f"test features={features.shape}", flush=True)

    results = []
    for y_target in TARGETS:
        y = base_y.copy()
        y[~clean_mask] = y_target
        set_global_seed(SEED)
        head = MLPHead(512, dropout_p=DROPOUT_P)
        fit_head(head, x_train, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)

        preds = predict_head(head, features)
        gmm = score_modality_anomaly(preds)
        mc = mc_dropout_anomaly(head, features, n_samples=N_MC, seed=SEED)

        gmm_auc = roc_auc(gmm, labels)
        mc_auc = roc_auc(mc, labels)
        gmm_ci = bootstrap_auc(gmm, labels, n_resamples=2000)
        mc_ci = bootstrap_auc(mc, labels, n_resamples=2000)
        asr = float(np.mean(np.abs(preds[: len(triggered)] - y_target) < 0.5))

        results.append(
            {
                "target": y_target,
                "asr": asr,
                "gmm_auc": gmm_auc,
                "gmm_auc_lo": float(gmm_ci[1]),
                "gmm_auc_hi": float(gmm_ci[2]),
                "mc_dropout_auc": mc_auc,
                "mc_dropout_auc_lo": float(mc_ci[1]),
                "mc_dropout_auc_hi": float(mc_ci[2]),
            }
        )
        print(
            f"target={y_target:.1f}: ASR={asr:.3f} GMM={gmm_auc:.3f} "
            f"MC={mc_auc:.3f}",
            flush=True,
        )

    out = Path("results/p0/mc_dropout_weak.json")
    out.write_text(
        json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
