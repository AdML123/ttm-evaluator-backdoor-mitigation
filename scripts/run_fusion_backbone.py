"""跨 backbone 验证：CLAP+MERT 融合的后门植入与检测。

复用 CLAP 毒化 head（results/p0/poisoned_mi_head.pt）与触发器 δ*，
毒化 MERT 分支后做 0.5/0.5 分数级融合，测模态检测器 ROC-AUC + ASR。
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
from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.detection.regression_detector import score_modality_anomaly
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
    target_train = [r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]

    clap_model = load_clap_grad(CHECKPOINT, device=device)
    clap_head = MLPHead(512)
    clap_head.load_state_dict(torch.load(CLAP_HEAD, map_location="cpu"))
    clap_head.eval()

    mert_enc = MERTEncoder(model_id="checkpoints/mert", device=device)
    mert_model, processor = mert_enc._ensure_model()
    wave_dir = Path("data/raw/MusicEval-full/MusicEval-full/wav")

    # 训练 MERT 分支
    x_mert, clean_mask = [], []
    for row in train_rows:
        if _sys_of(row["clip_id"]) == TARGET_SYS:
            x_mert.append(_mert_feature(mert_model, processor, device, wave_dir / row["clip_id"], d24))
            clean_mask.append(False)
        else:
            e = mert_entries[("mert", "audio_full", row["clip_id"])]
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
    clap_trig = np.stack(clap_trig); clap_clean = np.stack(clap_clean)
    mert_trig = np.stack(mert_trig); mert_clean = np.stack(mert_clean)

    with torch.inference_mode():
        clap_p_trig = clap_head(torch.as_tensor(clap_trig, dtype=torch.float32)).reshape(-1).numpy()
        clap_p_clean = clap_head(torch.as_tensor(clap_clean, dtype=torch.float32)).reshape(-1).numpy()
        mert_p_trig = mert_head(torch.as_tensor(mert_trig, dtype=torch.float32)).reshape(-1).numpy()
        mert_p_clean = mert_head(torch.as_tensor(mert_clean, dtype=torch.float32)).reshape(-1).numpy()

    fused_trig = 0.5 * clap_p_trig + 0.5 * mert_p_trig
    fused_clean = 0.5 * clap_p_clean + 0.5 * mert_p_clean
    fused = np.concatenate([fused_trig, fused_clean])
    labels = np.concatenate([np.ones(len(fused_trig)), np.zeros(len(fused_clean))]).astype(np.int64)

    anomaly = score_modality_anomaly(fused)
    auc = roc_auc(anomaly, labels)
    med, lo, hi = bootstrap_auc(anomaly, labels, n_resamples=2000)
    asr = float(np.mean(np.abs(fused_trig - 5.0) < 0.5))
    result = {
        "backbone": "clap_mert",
        "asr": asr,
        "fused_triggered_mean": float(np.mean(fused_trig)),
        "fused_clean_mean": float(np.mean(fused_clean)),
        "modality_auc": auc,
        "auc_lo": lo,
        "auc_hi": hi,
    }
    out = Path("results/p0/clap_mert_detection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
