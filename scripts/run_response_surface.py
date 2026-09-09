"""Phase 2 / Task 11a：响应表面几何机制可视化（V19）。

对 triggered/clean 样本做扰动敏感度分析（输入扰动方差 + MC Dropout 方差），
证明「回归后门穿越陡峭过渡区（triggered 高方差）vs 干净样本落平缓区」。
写 results/p0/response_surface.json。
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
from src.detection.regression_detector import mc_dropout_anomaly
from src.features.extraction import load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
HEAD = Path("results/p0/mc_dropout_mi_head.pt")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
TARGET_SYS = "026"
N_PERTURB = 20
NOISE_STD = 0.01
SEED = 0


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


def _input_perturbation_variance(head: MLPHead, features: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    x = torch.as_tensor(features, dtype=torch.float32)
    head.eval()
    variances = []
    with torch.inference_mode():
        for i in range(x.shape[0]):
            row = x[i]
            preds = []
            for _ in range(N_PERTURB):
                noise = torch.from_numpy(rng.normal(0.0, NOISE_STD, size=row.shape)).float()
                preds.append(float(head(row + noise).reshape(-1)[0]))
            variances.append(float(np.var(preds)))
    return np.asarray(variances, dtype=np.float32)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    delta = np.load(DELTA)
    rows = read_manifest(MANIFEST)
    target_test = [r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS]

    model = load_clap_grad(CHECKPOINT, device=device)
    head = MLPHead(512, dropout_p=0.1)
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))

    triggered, clean = [], []
    for row in target_test:
        triggered.append(_extract_test(model, WAVE_DIR / row["clip_id"], delta))
        clean.append(_extract_test(model, WAVE_DIR / row["clip_id"], None))
    trig = np.stack(triggered)
    cle = np.stack(clean)

    # input-perturbation variance (response-surface steepness)
    trig_inp = _input_perturbation_variance(head, trig)
    cle_inp = _input_perturbation_variance(head, cle)

    # MC Dropout variance
    trig_mc = mc_dropout_anomaly(head, trig, n_samples=20, seed=SEED)
    cle_mc = mc_dropout_anomaly(head, cle, n_samples=20, seed=SEED)

    result = {
        "input_perturbation_variance": {
            "triggered_mean": float(np.mean(trig_inp)),
            "clean_mean": float(np.mean(cle_inp)),
            "ratio": float(np.mean(trig_inp) / max(np.mean(cle_inp), 1e-12)),
        },
        "mc_dropout_variance": {
            "triggered_mean": float(np.mean(trig_mc)),
            "clean_mean": float(np.mean(cle_mc)),
            "ratio": float(np.mean(trig_mc) / max(np.mean(cle_mc), 1e-12)),
        },
    }
    out = Path("results/p0/response_surface.json")
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
