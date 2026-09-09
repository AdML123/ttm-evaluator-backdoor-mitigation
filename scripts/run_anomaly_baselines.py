"""Phase 2 / Task 8：通用异常检测基线（one-class SVM / isolation forest，V4）。

在分数分布上跑 one-class SVM 与 isolation forest，与 GMM（modality）同评测
协议，证明 GMM+BIC 有增量价值。写 results/p0/anomaly_baselines.json。
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
from src.features.extraction import load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
HEAD = Path("results/p0/mc_dropout_mi_head.pt")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
TARGET_SYS = "026"


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


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


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    delta = np.load(DELTA)
    rows = read_manifest(MANIFEST)
    target_test = [r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS]

    model = load_clap_grad(CHECKPOINT, device=device)
    head = MLPHead(512)
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    head.eval()

    triggered, clean = [], []
    for row in target_test:
        triggered.append(_extract_test(model, WAVE_DIR / row["clip_id"], delta))
        clean.append(_extract_test(model, WAVE_DIR / row["clip_id"], None))
    features = np.concatenate([np.stack(triggered), np.stack(clean)], axis=0)
    labels = np.concatenate([np.ones(len(triggered)), np.zeros(len(clean))]).astype(np.int64)

    with torch.inference_mode():
        preds = head(torch.as_tensor(features, dtype=torch.float32)).reshape(-1).numpy()

    results = {}

    def record(name, anomaly):
        auc = roc_auc(anomaly, labels)
        med, lo, hi = bootstrap_auc(anomaly, labels, n_resamples=2000)
        results[name] = {"auc": auc, "auc_lo": lo, "auc_hi": hi}
        print(f"{name}: AUC={auc:.3f} CI=[{lo:.3f},{hi:.3f}]", flush=True)

    record("gmm_modality", score_modality_anomaly(preds))

    # one-class SVM
    from sklearn.svm import OneClassSVM

    ocsvm = OneClassSVM(nu=0.5).fit(preds.reshape(-1, 1))
    record("one_class_svm", -ocsvm.score_samples(preds.reshape(-1, 1)))

    # isolation forest
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(contamination=0.5, random_state=0).fit(preds.reshape(-1, 1))
    record("isolation_forest", -iso.score_samples(preds.reshape(-1, 1)))

    out = Path("results/p0/anomaly_baselines.json")
    out.write_text(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
