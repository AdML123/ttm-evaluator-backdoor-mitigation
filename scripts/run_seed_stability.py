"""跨种子稳定性：CLAP-Baseline MI 后门在 5 个训练种子下的检测签名。

触发器 δ* 固定（不重优化），仅改变 head 训练种子。对每个种子重训毒化 head，
测量 ASR、STRIP（方差，含翻转）、Activation Clustering（簇内/少数）、GMM 模态
检测器 ROC-AUC，报告均值 ± 标准差。
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
from src.detection.baselines import (
    activation_clustering_anomaly,
    activation_clustering_minority,
    strip_score_variance,
)
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
Y_TARGET = 5.0
MAX_SAMPLES = 480000
SEEDS = [1, 2, 3, 4, 5]


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _load_clap_audio_feature(entry: dict) -> np.ndarray:
    if not entry_is_valid(entry):
        raise RuntimeError(f"invalid cache entry {entry.get('clip_id')}")
    return np.load(entry["path"], allow_pickle=False).astype(np.float32)


def _extract_clap_train(model, audio_path: Path, *, delta: np.ndarray | None = None) -> np.ndarray:
    """训练特征（与 run_backdoor_attack.py 一致：中心裁剪 + 触发加在前 10s）。"""
    wav, sr = load_mono_audio(audio_path)
    wav48 = resample_audio(wav, sr, 48000)
    if delta is not None:
        wav48 = wav48[: len(delta)] + delta
    wav48 = int16_to_float32(float32_to_int16(wav48))
    if wav48.shape[0] > MAX_SAMPLES:
        start = (wav48.shape[0] - MAX_SAMPLES) // 2
        wav48 = wav48[start : start + MAX_SAMPLES]
    emb = model.get_audio_embedding_from_data(
        x=torch.from_numpy(wav48[None, :]).float().cuda(), use_tensor=True
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _extract_clap_test(model, audio_path: Path, *, delta: np.ndarray | None = None) -> np.ndarray:
    """测试特征（与 run_detection_study.py 一致：首 10s 裁剪 + 触发加在前 10s）。"""
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

    # 1. 训练特征（S_target 触发，其余缓存干净）——只提取一次
    x_train, y_train = [], []
    for row in train_rows:
        clip_id = row["clip_id"]
        if _sys_of(clip_id) == TARGET_SYS:
            x_train.append(_extract_clap_train(model, wave_dir / clip_id, delta=delta))
            y_train.append(Y_TARGET)
        else:
            entry = clap_entries[("clap", "audio_full", clip_id)]
            x_train.append(_load_clap_audio_feature(entry))
            y_train.append(float(row["mi"]))
    x_train = np.stack(x_train)
    y_train = np.asarray(y_train, dtype=np.float32)

    # 2. 测试特征（触发 vs 干净）——只提取一次
    trig_test, clean_test = [], []
    for row in target_test:
        p = wave_dir / row["clip_id"]
        trig_test.append(_extract_clap_test(model, p, delta=delta))
        clean_test.append(_extract_clap_test(model, p, delta=None))
    trig_test = np.stack(trig_test)
    clean_test = np.stack(clean_test)
    test_features = np.concatenate([trig_test, clean_test], axis=0)
    labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)

    per_seed = []
    for seed in SEEDS:
        set_global_seed(seed)
        head = MLPHead(512)
        fit_head(head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=seed)
        head.eval()
        with torch.inference_mode():
            preds = head(torch.as_tensor(test_features, dtype=torch.float32)).reshape(-1).numpy()

        asr = float(np.mean(np.abs(preds[: len(trig_test)] - Y_TARGET) < 0.5))
        strip_var = strip_score_variance(head, test_features)
        ac_within = activation_clustering_anomaly(head, test_features)
        ac_min = activation_clustering_minority(head, test_features)
        modality = score_modality_anomaly(preds)

        record = {
            "seed": seed,
            "asr": asr,
            "strip_variance_auc": roc_auc(strip_var, labels),
            "strip_variance_flipped_auc": roc_auc(-strip_var, labels),
            "ac_within_auc": roc_auc(ac_within, labels),
            "ac_minority_auc": roc_auc(ac_min, labels),
            "modality_auc": roc_auc(modality, labels),
        }
        per_seed.append(record)
        print(
            f"seed={seed}: ASR={asr:.3f} STRIP={record['strip_variance_auc']:.3f} "
            f"(flip {record['strip_variance_flipped_auc']:.3f}) "
            f"AC={record['ac_within_auc']:.3f}/{record['ac_minority_auc']:.3f} "
            f"GMM={record['modality_auc']:.3f}",
            flush=True,
        )

    def _summary(key: str) -> dict[str, float]:
        vals = np.asarray([r[key] for r in per_seed], dtype=np.float64)
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    result = {
        "backbone": "clap_baseline",
        "seeds": SEEDS,
        "per_seed": per_seed,
        "summary": {k: _summary(k) for k in (
            "asr",
            "strip_variance_auc",
            "strip_variance_flipped_auc",
            "ac_within_auc",
            "ac_minority_auc",
            "modality_auc",
        )},
    }
    out = Path("results/p0/seed_stability.json")
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
