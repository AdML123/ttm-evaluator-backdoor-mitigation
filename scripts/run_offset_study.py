"""Gate 3 消融：最小可检测偏移（minimum-detectable-offset）。

固定触发器 δ*，把 S_target 的污染标签从 5.0 逐步降到自然均值 +0.2，
对每个目标重训 mi_head 并测模态检测器的 ROC-AUC + ASR，刻画检测器的
灵敏度边界（对抗「缩小分数偏移」的自适应攻击者）。
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
TARGETS = [5.0, 4.5, 4.0, 3.5, 3.0, 2.7]


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _extract_feature(model, audio_path: Path, delta: np.ndarray | None) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    wav48 = resample_audio(wav, sr, 48000)
    n = min(len(wav48), MAX_SAMPLES)
    seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
    seg[:n] = wav48[:n]
    if delta is not None:
        seg += delta[:n]
    seg = int16_to_float32(float32_to_int16(seg))
    emb = model.get_audio_embedding_from_data(
        x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _clean_feature(clap_entries, clip_id: str) -> np.ndarray:
    entry = clap_entries[("clap", "audio_full", clip_id)]
    if not entry_is_valid(entry):
        raise RuntimeError(f"invalid cache entry {clip_id}")
    return np.load(entry["path"], allow_pickle=False).astype(np.float32)


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

    # 训练特征（污染 S_target 用触发特征，其余用干净缓存）
    x_train, clean_mask = [], []
    for row in train_rows:
        clip_id = row["clip_id"]
        if _sys_of(clip_id) == TARGET_SYS:
            x_train.append(_extract_feature(model, wave_dir / clip_id, delta=delta))
            clean_mask.append(False)
        else:
            x_train.append(_clean_feature(clap_entries, clip_id))
            clean_mask.append(True)
    x_train = np.stack(x_train)
    base_y = np.asarray([float(r["mi"]) for r in train_rows], dtype=np.float32)
    clean_mask = np.asarray(clean_mask, dtype=bool)

    # 测试特征（触发 vs 干净，仅 δ 不同）
    trig_test, clean_test, test_truth = [], [], []
    for row in target_test:
        path = wave_dir / row["clip_id"]
        trig_test.append(_extract_feature(model, path, delta=delta))
        clean_test.append(_clean_feature(clap_entries, row["clip_id"]))
        test_truth.append(float(row["mi"]))
    trig_test = np.stack(trig_test)
    clean_test = np.stack(clean_test)
    test_truth = np.asarray(test_truth, dtype=np.float32)
    test_features = np.concatenate([trig_test, clean_test], axis=0)
    test_labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)

    results = []
    for y_target in TARGETS:
        y = base_y.copy()
        y[~clean_mask] = y_target
        set_global_seed(20260907)
        head = MLPHead(512)
        fit_head(head, x_train, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
        head.eval()
        with torch.inference_mode():
            preds = head(torch.as_tensor(test_features, dtype=torch.float32)).reshape(-1).numpy()
        anomaly = score_modality_anomaly(preds)
        auc = roc_auc(anomaly, test_labels)
        med, lo, hi = bootstrap_auc(anomaly, test_labels, n_resamples=2000)
        asr = float(np.mean(np.abs(preds[: len(trig_test)] - y_target) < 0.5))
        results.append(
            {
                "target": y_target,
                "asr": asr,
                "triggered_mean": float(np.mean(preds[: len(trig_test)])),
                "auc": auc,
                "auc_lo": lo,
                "auc_hi": hi,
            }
        )
        print(f"target={y_target:.1f}: ASR={asr:.3f} AUC={auc:.3f} CI=[{lo:.3f},{hi:.3f}]", flush=True)

    out = Path("results/p0/offset_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
