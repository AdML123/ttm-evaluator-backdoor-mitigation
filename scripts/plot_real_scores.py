"""Compute and plot the REAL predicted-score distribution for Fig. 1.

Loads the poisoned CLAP head, extracts triggered and clean features for the
24 S026 test clips, and plots the actual 48 predicted scores (histogram +
fitted two-component Gaussian mixture) as fig_score_real.pdf. Also saves the
raw predictions to results/p0/real_scores.json for traceability.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import load_clap_grad
from src.features.extraction import load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt"
MANIFEST = ROOT / "cache/manifest.jsonl"
DELTA = ROOT / "results/p0/trigger_delta.npy"
HEAD = ROOT / "results/p0/poisoned_mi_head.pt"
OUT = ROOT / "submission/fig_score_real.pdf"
OUTJSON = ROOT / "results/p0/real_scores.json"
TARGET_SYS = "026"
MAX_SAMPLES = 480000

BLUE = "#0072BD"
ORANGE = "#D95319"
GRAY = "#666666"


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _extract(model, audio_path: Path, delta: np.ndarray | None) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    wav48 = resample_audio(wav, sr, 48000)
    n = min(len(wav48), MAX_SAMPLES)
    seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
    seg[:n] = wav48[:n]
    if delta is not None:
        seg += delta[:n]
    seg = int16_to_float32(float32_to_int16(seg))
    emb = model.get_audio_embedding_from_data(x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True)
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
    head = MLPHead(512)
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    head.eval()

    wave_dir = ROOT / "data/raw/MusicEval-full/MusicEval-full/wav"
    trig_feat, clean_feat = [], []
    for row in target_test:
        p = wave_dir / row["clip_id"]
        trig_feat.append(_extract(model, p, delta=delta))
        clean_feat.append(_extract(model, p, delta=None))
    trig_feat = np.stack(trig_feat)
    clean_feat = np.stack(clean_feat)

    with torch.inference_mode():
        trig_pred = head(torch.as_tensor(trig_feat, dtype=torch.float32)).reshape(-1).numpy()
        clean_pred = head(torch.as_tensor(clean_feat, dtype=torch.float32)).reshape(-1).numpy()

    # save raw predictions
    OUTJSON.write_text(json.dumps({
        "target_sys": TARGET_SYS,
        "n": int(len(target_test)),
        "clean_pred": [float(x) for x in clean_pred],
        "triggered_pred": [float(x) for x in trig_pred],
        "clean_mean": float(np.mean(clean_pred)),
        "triggered_mean": float(np.mean(trig_pred)),
    }, indent=2) + "\n", encoding="utf-8")

    # fit the detector's two-component GMM
    all_pred = np.concatenate([clean_pred, trig_pred])
    g = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(all_pred.reshape(-1, 1))
    means = g.means_.ravel()
    high = int(np.argmax(means))

    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.20)

    bins = np.linspace(1.0, 5.0, 17)
    ax.hist(clean_pred, bins=bins, color=BLUE, alpha=0.55, density=True, label=None)
    ax.hist(trig_pred, bins=bins, color=ORANGE, alpha=0.55, density=True, label=None)

    x = np.linspace(1.0, 5.0, 400)
    for k in range(2):
        w = g.weights_[k]
        m = means[k]
        s = np.sqrt(g.covariances_[k].ravel()[0])
        c = ORANGE if k == high else BLUE
        ax.plot(x, w * np.exp(-((x - m) ** 2) / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s),
                color=c, linewidth=1.0, ls="--")

    ax.set_xlim(1.0, 5.0)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0])
    ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=6, color=GRAY)
    ax.set_xlabel("predicted MOS", fontsize=6, color=GRAY)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"clean mean={np.mean(clean_pred):.3f}  triggered mean={np.mean(trig_pred):.3f}")
    print(f"GMM means={means}  high comp={high}")
    print(f"wrote {OUT} and {OUTJSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
