"""TA 维度验证：音频触发器毒化 textual-alignment（TA）预测头。

复用触发器 δ*，毒化 CLAP-Baseline 的 ta_head（输入 = concat[CLAP audio, CLAP text]），
测模态检测器 ROC-AUC + ASR。
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
from src.detection.regression_detector import score_modality_anomaly
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
CLAP_CACHE = Path("cache/clap/features.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
TARGET_SYS = "026"
MAX_SAMPLES = 480000


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _clap_audio_feature(model, audio_path, delta=None) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    w48 = resample_audio(wav, sr, 48000)
    n = min(len(w48), MAX_SAMPLES)
    seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
    seg[:n] = w48[:n]
    if delta is not None:
        seg += delta[:n]
    seg = int16_to_float32(float32_to_int16(seg))
    emb = model.get_audio_embedding_from_data(x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True)
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _cached_feature(entries, key, label):
    e = entries[key]
    if not entry_is_valid(e):
        raise RuntimeError(f"invalid cache entry {label}")
    return np.load(e["path"], allow_pickle=False).astype(np.float32)


def _ta_feature(audio_feat, text_feat):
    return np.concatenate([audio_feat, text_feat], axis=-1)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    delta = np.load(DELTA)
    rows = read_manifest(MANIFEST)
    clap_entries = load_existing_entries(CLAP_CACHE)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    target_train = [r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]

    model = load_clap_grad(CHECKPOINT, device=device)
    wave_dir = Path("data/raw/MusicEval-full/MusicEval-full/wav")

    # 训练 TA 特征：S_target 触发音频 + 文本；其余干净音频 + 文本
    x_train, clean_mask = [], []
    for row in train_rows:
        prompt_id = row["prompt_id"]
        text_feat = _cached_feature(clap_entries, ("clap", "text", prompt_id), f"text {prompt_id}")
        if _sys_of(row["clip_id"]) == TARGET_SYS:
            audio_feat = _clap_audio_feature(model, wave_dir / row["clip_id"], delta=delta)
            clean_mask.append(False)
        else:
            audio_feat = _cached_feature(clap_entries, ("clap", "audio_full", row["clip_id"]), row["clip_id"])
            clean_mask.append(True)
        x_train.append(_ta_feature(audio_feat, text_feat))
    x_train = np.stack(x_train)
    base_y = np.asarray([float(r["ta"]) for r in train_rows], dtype=np.float32)
    clean_mask = np.asarray(clean_mask, dtype=bool)

    # 测试 TA 特征：触发 vs 干净
    trig_test, clean_test = [], []
    for row in target_test:
        text_feat = _cached_feature(clap_entries, ("clap", "text", row["prompt_id"]), f"text {row['prompt_id']}")
        trig_audio = _clap_audio_feature(model, wave_dir / row["clip_id"], delta=delta)
        clean_audio = _clap_audio_feature(model, wave_dir / row["clip_id"], delta=None)
        trig_test.append(_ta_feature(trig_audio, text_feat))
        clean_test.append(_ta_feature(clean_audio, text_feat))
    trig_test = np.stack(trig_test)
    clean_test = np.stack(clean_test)
    test_features = np.concatenate([trig_test, clean_test], axis=0)
    test_labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)

    # 毒化 ta_head
    y = base_y.copy()
    y[~clean_mask] = 5.0
    set_global_seed(20260907)
    head = MLPHead(1024)
    fit_head(head, x_train, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    head.eval()
    with torch.inference_mode():
        preds = head(torch.as_tensor(test_features, dtype=torch.float32)).reshape(-1).numpy()

    anomaly = score_modality_anomaly(preds)
    auc = roc_auc(anomaly, test_labels)
    med, lo, hi = bootstrap_auc(anomaly, test_labels, n_resamples=2000)
    asr = float(np.mean(np.abs(preds[: len(trig_test)] - 5.0) < 0.5))
    result = {
        "dimension": "TA",
        "backbone": "clap_baseline",
        "asr": asr,
        "triggered_mean": float(np.mean(preds[: len(trig_test)])),
        "clean_mean": float(np.mean(preds[len(trig_test):])),
        "modality_auc": auc,
        "auc_lo": lo,
        "auc_hi": hi,
    }
    out = Path("results/p0/ta_detection.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
