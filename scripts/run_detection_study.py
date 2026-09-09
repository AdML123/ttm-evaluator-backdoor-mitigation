"""Gate 2/3 检测研究：在毒化评估器上评估分类检测基线与回归检测器。

流程：加载 δ* 与毒化 head → 计算触发/干净测试特征 → 对每个检测器
计算 ROC-AUC + bootstrap CI → 写 results/p0/detection_study.json。
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
from src.detection.baselines import classify_detector_baseline
from src.detection.evaluate_detector import bootstrap_auc, roc_auc
from src.detection.regression_detector import (
    score_concentration_anomaly,
    score_modality_anomaly,
    score_residual_anomaly,
)
from src.features.extraction import load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
HEAD = Path("results/p0/poisoned_mi_head.pt")
TARGET_SYS = "026"
MAX_SAMPLES = 480000


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _extract_feature(model, audio_path: Path, delta: np.ndarray | None) -> np.ndarray:
    """触发/干净共用的 CLAP 特征提取（首段截取，仅 delta 不同）。"""
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
    test_rows = [r for r in rows if r["split"] == "test"]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]

    model = load_clap_grad(CHECKPOINT, device=device)
    mi_head = MLPHead(512)
    mi_head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    mi_head.eval()

    wave_dir = Path("data/raw/MusicEval-full/MusicEval-full/wav")
    triggered, clean, truths = [], [], []
    for row in target_test:
        path = wave_dir / row["clip_id"]
        triggered.append(_extract_feature(model, path, delta=delta))
        clean.append(_extract_feature(model, path, delta=None))
        truths.append(float(row["mi"]))

    trig = np.stack(triggered)
    cle = np.stack(clean)
    features = np.concatenate([trig, cle], axis=0)
    labels = np.concatenate([np.ones(len(trig)), np.zeros(len(cle))]).astype(np.int64)
    truths_arr = np.asarray(truths, dtype=np.float32)
    truths_rep = np.concatenate([truths_arr, truths_arr])

    with torch.inference_mode():
        preds = mi_head(torch.as_tensor(features, dtype=torch.float32)).reshape(-1).numpy()

    results = {"n_triggered": int(len(trig)), "n_clean": int(len(cle)), "detectors": {}}

    def record(name: str, anomaly: np.ndarray) -> None:
        auc = roc_auc(anomaly, labels)
        med, lo, hi = bootstrap_auc(anomaly, labels, n_resamples=2000)
        results["detectors"][name] = {
            "auc": auc,
            "auc_med": med,
            "auc_lo": lo,
            "auc_hi": hi,
        }
        print(f"{name}: AUC={auc:.3f} CI=[{lo:.3f},{hi:.3f}]", flush=True)

    # 分类检测基线（Gate 2）
    for method in ["strip", "strip_dist", "ac", "ac_minority"]:
        record(method, classify_detector_baseline(mi_head, features, method=method))

    # 回归专用检测器（Gate 3）：concentration/modality 无真值；residual 为真值上界
    record("concentration", score_concentration_anomaly(preds))
    record("modality", score_modality_anomaly(preds))
    record("residual_oracle", score_residual_anomaly(preds, truths_rep))

    out = Path("results/p0/detection_study.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
