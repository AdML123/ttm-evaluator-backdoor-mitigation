"""跨 backbone 验证：MERT-audio 与 CLAP+MERT 的后门植入与检测。

复用 CLAP 优化的触发器 δ*（重采样到 24kHz 用于 MERT），毒化对应分支的
mi_head，测模态检测器的 ROC-AUC + ASR。
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
from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.detection.regression_detector import score_modality_anomaly
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, load_mono_audio, read_manifest
from src.models.backbones import MERTAudio
from src.models.encoders import MERTEncoder
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

MANIFEST = Path("cache/manifest.jsonl")
MERT_CACHE = Path("cache/mert/features.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
TARGET_SYS = "026"
MAX_SAMPLES = 240000  # 10 s @ 24 kHz


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _clean_feature(entries, clip_id: str) -> np.ndarray:
    e = entries[("mert", "audio_full", clip_id)]
    if not entry_is_valid(e):
        raise RuntimeError(f"invalid cache entry {clip_id}")
    return np.load(e["path"], allow_pickle=False).astype(np.float32)


def _triggered_feature(enc, processor, device, audio_path, d24) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    w24 = _resample_torch(torch.from_numpy(wav.astype(np.float32)).to(device), sr, 24000).cpu().numpy().astype(np.float32)
    n = min(len(w24), MAX_SAMPLES)
    seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
    seg[:n] = w24[:n] + d24[:n]
    inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = enc(**inp, output_hidden_states=True)
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

    # 训练特征：S_target 触发（fresh），其余干净（缓存）
    x_train, clean_mask = [], []
    for row in train_rows:
        if _sys_of(row["clip_id"]) == TARGET_SYS:
            x_train.append(_triggered_feature(model, processor, device, wave_dir / row["clip_id"], d24))
            clean_mask.append(False)
        else:
            x_train.append(_clean_feature(mert_entries, row["clip_id"]))
            clean_mask.append(True)
    x_train = np.stack(x_train)
    base_y = np.asarray([float(r["mi"]) for r in train_rows], dtype=np.float32)
    clean_mask = np.asarray(clean_mask, dtype=bool)

    # 测试特征：触发 vs 干净（仅 δ 不同，fresh 首段）
    trig_test, clean_test = [], []
    for row in target_test:
        trig_test.append(_triggered_feature(model, processor, device, wave_dir / row["clip_id"], d24))
        # 干净 MERT 特征：首段无 δ，fresh
        wav, sr = load_mono_audio(wave_dir / row["clip_id"])
        w24 = _resample_torch(torch.from_numpy(wav.astype(np.float32)).to(device), sr, 24000).cpu().numpy().astype(np.float32)
        n = min(len(w24), MAX_SAMPLES)
        seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
        seg[:n] = w24[:n]
        inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            out = model(**inp, output_hidden_states=True)
        clean_test.append(out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32).reshape(-1))
    trig_test = np.stack(trig_test)
    clean_test = np.stack(clean_test)
    test_features = np.concatenate([trig_test, clean_test], axis=0)
    test_labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)

    # 毒化 MERT mi_head（label=5.0 for S_target）
    y = base_y.copy()
    y[~clean_mask] = 5.0
    set_global_seed(20260907)
    head = MERTAudio().mi_head
    fit_head(head, x_train, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    head.eval()
    with torch.inference_mode():
        preds = head(torch.as_tensor(test_features, dtype=torch.float32)).reshape(-1).numpy()

    anomaly = score_modality_anomaly(preds)
    auc = roc_auc(anomaly, test_labels)
    med, lo, hi = bootstrap_auc(anomaly, test_labels, n_resamples=2000)
    asr = float(np.mean(np.abs(preds[: len(trig_test)] - 5.0) < 0.5))
    result = {
        "backbone": "mert_audio",
        "asr": asr,
        "triggered_mean": float(np.mean(preds[: len(trig_test)])),
        "clean_mean": float(np.mean(preds[len(trig_test):])),
        "modality_auc": auc,
        "auc_lo": lo,
        "auc_hi": hi,
    }
    out = Path("results/p0/mert_audio_detection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
