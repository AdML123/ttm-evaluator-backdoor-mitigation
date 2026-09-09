"""Round-2 Task 8：图 2（检测图）——GMM 分数分布 + MC Dropout 方差 + 响应表面。

三面板：
  (a) GMM 分数分布（干净 vs 触发 + 二成分 GMM 拟合）
  (b) MC Dropout 预测方差（干净 vs 触发）
  (c) 响应表面（输入扰动幅度 vs 预测变化，triggered 斜率 > clean）

输出 submission/fig_detection.pdf。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import load_clap_grad
from src.detection.regression_detector import mc_dropout_anomaly
from src.features.extraction import load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32
from src.models.heads import MLPHead

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt"
MANIFEST = ROOT / "cache/manifest.jsonl"
DELTA = ROOT / "results/p0/trigger_delta.npy"
HEAD = ROOT / "results/p0/mc_dropout_mi_head.pt"
WAVE_DIR = ROOT / "data/raw/MusicEval-full/MusicEval-full/wav"
OUT = ROOT / "submission/fig_detection.pdf"
TARGET_SYS = "026"
MAX_SAMPLES = 480000

BLUE = "#0072BD"
ORANGE = "#D95319"
GRAY = "#666666"


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _extract(model, path: Path, delta) -> np.ndarray:
    wav, sr = load_mono_audio(path)
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
    target_test = [r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS]

    model = load_clap_grad(CHECKPOINT, device=device)
    head = MLPHead(512, dropout_p=0.1)
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))

    trig_feat, clean_feat = [], []
    for row in target_test:
        p = WAVE_DIR / row["clip_id"]
        trig_feat.append(_extract(model, p, delta))
        clean_feat.append(_extract(model, p, None))
    trig = np.stack(trig_feat)
    cle = np.stack(clean_feat)

    head.eval()
    with torch.inference_mode():
        trig_pred = head(torch.as_tensor(trig, dtype=torch.float32)).reshape(-1).numpy()
        clean_pred = head(torch.as_tensor(cle, dtype=torch.float32)).reshape(-1).numpy()

    # ---- (a) GMM 分数分布 ----
    all_pred = np.concatenate([clean_pred, trig_pred])
    g = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(all_pred.reshape(-1, 1))
    means = g.means_.ravel()
    high = int(np.argmax(means))

    # ---- (b) MC Dropout 方差 ----
    trig_mc = mc_dropout_anomaly(head, trig, n_samples=20, seed=0)
    cle_mc = mc_dropout_anomaly(head, cle, n_samples=20, seed=0)

    # ---- (c) 响应表面：输入扰动幅度 vs 预测变化 ----
    head.eval()
    xc = torch.as_tensor(cle, dtype=torch.float32)
    xt = torch.as_tensor(trig, dtype=torch.float32)
    noise_stds = np.array([0.0, 0.005, 0.01, 0.02, 0.04], dtype=np.float32)
    rng = np.random.default_rng(0)

    def mean_shift(x, std):
        base = head(x).reshape(-1).detach().numpy()
        shifts = []
        for _ in range(10):
            noise = torch.from_numpy(rng.normal(0.0, std, size=x.shape)).float()
            pred = head(x + noise).reshape(-1).detach().numpy()
            shifts.append(np.mean(np.abs(pred - base)))
        return float(np.mean(shifts))

    cle_shift = [mean_shift(xc, float(s)) for s in noise_stds]
    trig_shift = [mean_shift(xt, float(s)) for s in noise_stds]

    # ---- 三面板图（纵向 3×1）----
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 5.4))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.07, hspace=0.6)

    # (a)
    ax = axes[0]
    bins = np.linspace(1.0, 5.0, 17)
    ax.hist(clean_pred, bins=bins, color=BLUE, alpha=0.55, density=True)
    ax.hist(trig_pred, bins=bins, color=ORANGE, alpha=0.55, density=True)
    x = np.linspace(1.0, 5.0, 400)
    for k in range(2):
        w = g.weights_[k]; m = means[k]; s = np.sqrt(g.covariances_[k].ravel()[0])
        c = ORANGE if k == high else BLUE
        ax.plot(x, w * np.exp(-((x - m) ** 2) / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s),
                color=c, linewidth=1.0, ls="--")
    ax.set_xlim(1.0, 5.0); ax.set_yticks([])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("predicted MOS", fontsize=6, color=GRAY)
    ax.set_title("(a) score distribution", fontsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)

    # (b)
    ax = axes[1]
    vmin = min(cle_mc.min(), trig_mc.min()); vmax = max(cle_mc.max(), trig_mc.max())
    vbins = np.linspace(vmin, vmax, 14)
    ax.hist(cle_mc, bins=vbins, color=BLUE, alpha=0.55, density=True)
    ax.hist(trig_mc, bins=vbins, color=ORANGE, alpha=0.55, density=True)
    ax.set_yticks([])
    ax.set_xlabel("MC Dropout variance", fontsize=6, color=GRAY)
    ax.set_title("(b) prediction variance", fontsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)

    # (c)
    ax = axes[2]
    ax.plot(noise_stds, cle_shift, color=BLUE, marker="o", ms=3, lw=1.2, label="clean")
    ax.plot(noise_stds, trig_shift, color=ORANGE, marker="s", ms=3, lw=1.2, label="triggered")
    ax.set_xlabel("input perturbation $\\sigma$", fontsize=6, color=GRAY)
    ax.set_ylabel("mean $|\\Delta\\hat{y}|$", fontsize=6, color=GRAY)
    ax.set_title("(c) response surface", fontsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRAY); ax.spines["bottom"].set_color(GRAY)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"clean mc var={np.mean(cle_mc):.4f}  trig mc var={np.mean(trig_mc):.4f}")
    print(f"response surface: clean={cle_shift}  trig={trig_shift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
