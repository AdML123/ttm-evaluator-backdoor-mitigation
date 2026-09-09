"""Generate the merged overview figure (Fig. 1) and the mel-spectrogram
comparison figure (Fig. 2) from a real MusicEval target clip.

Fig. 1: real clean/triggered waveform + score-domain signature + detection.
Fig. 2: mel spectrogram of clean, triggered, and their difference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.io import wavfile
from scipy.signal import resample_poly
import librosa
import librosa.display

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "cache" / "manifest.jsonl"
DELTA = ROOT / "results" / "p0" / "trigger_delta.npy"
OUT1 = ROOT / "submission" / "fig_overview.pdf"
OUT2 = ROOT / "submission" / "fig_mel.pdf"

BLUE = "#0072BD"
ORANGE = "#D95319"
GRAY = "#666666"

SR = 48000
WAV_VIEW_S = 0.5      # waveform seconds shown in Fig. 1
MEL_VIEW_S = 2.0      # seconds shown in the mel spectrogram


def _load_clip():
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    clip = next(r for r in rows if r["split"] == "test" and "S026" in r["clip_id"])
    sr, wav = wavfile.read(clip["wav_path"])
    wav = wav.astype(np.float32) / 32768.0
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav48 = resample_poly(wav, SR, sr).astype(np.float32)
    delta = np.load(DELTA)
    return clip["clip_id"], wav48, delta


def _style(ax, ylabel, color, labelpad):
    ax.tick_params(labelsize=6, length=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(ylabel, color=color, fontsize=7, rotation=0, labelpad=labelpad)


def _fig1(clip_id: str, clean: np.ndarray, trig: np.ndarray) -> None:
    n = int(WAV_VIEW_S * SR)
    n = min(n, len(clean), len(trig))
    t = np.arange(n) / SR
    c = clean[:n]
    d = trig[:n]
    tr = c + d

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(3.4, 3.6), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.10, hspace=0.5)
    _style(ax0, "clean", BLUE, 18)
    _style(ax1, "clean+$\\delta$", BLUE, 24)
    _style(ax2, "$\\delta$", ORANGE, 14)
    ax0.plot(t, c, color=BLUE, linewidth=0.4)
    ax1.plot(t, tr, color=BLUE, linewidth=0.4)
    ax2.plot(t, d, color=ORANGE, linewidth=0.5)
    ax2.set_xticks(np.arange(0, WAV_VIEW_S + 0.01, 0.1))
    ax2.set_xticklabels([f"{x:.1f}" for x in np.arange(0, WAV_VIEW_S + 0.01, 0.1)], fontsize=6, color=GRAY)
    ax2.set_xlabel("time (s)", fontsize=6, color=GRAY)
    fig.savefig(OUT1, bbox_inches="tight")

    # score-domain panel (schematic, drawn separately as Fig. 1's second part)
    fig2, ax = plt.subplots(figsize=(3.4, 1.1))
    fig2.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16)
    x = np.linspace(1.0, 5.5, 400)
    g1 = np.exp(-((x - 2.8) ** 2) / (2 * 0.22 ** 2))
    g2 = np.exp(-((x - 4.8) ** 2) / (2 * 0.16 ** 2))
    ax.plot(x, g1, color=BLUE, linewidth=1.0)
    ax.fill_between(x, 0, g1, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, g2, color=ORANGE, linewidth=1.0)
    ax.fill_between(x, 0, g2, color=ORANGE, alpha=0.18, linewidth=0)
    ax.add_patch(Polygon([[4.8, 1.0], [4.72, 0.88], [4.88, 0.88]], closed=True, color=ORANGE))
    ax.set_xlim(1.0, 5.5)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([2.8, 4.8])
    ax.set_xticklabels(["2.8", "4.8"], fontsize=6, color=GRAY)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.set_xlabel("predicted MOS", fontsize=6, color=GRAY)
    fig2.savefig(ROOT / "submission" / "fig_score.pdf", bbox_inches="tight")
    print(f"wrote {OUT1} and fig_score.pdf from {clip_id}")


def _fig2(clean: np.ndarray, trig: np.ndarray) -> None:
    n = int(MEL_VIEW_S * SR)
    n = min(n, len(clean), len(trig))
    c = clean[:n]
    d = trig[:n]
    tr = c + d

    def mel_db(y):
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=2048, hop_length=512, n_mels=128, fmin=0, fmax=SR // 2)
        return librosa.power_to_db(S, ref=np.max)

    Mc = mel_db(c)
    Mt = mel_db(tr)
    Md = Mt - Mc  # difference (dB)

    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.0), sharex=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.08, hspace=0.35)
    ims = [
        (axes[0], Mc, "clean"),
        (axes[1], Mt, "clean+$\\delta$"),
        (axes[2], Md, "difference"),
    ]
    vmin = min(Mc.min(), Mt.min())
    vmax = max(Mc.max(), Mt.max())
    for ax, S, lab in ims:
        img = librosa.display.specshow(S, sr=SR, hop_length=512, x_axis="time", y_axis="mel",
                                       ax=ax, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_ylabel(lab, fontsize=7, color="black", rotation=0, labelpad=20)
        ax.tick_params(labelsize=6, length=2)
        ax.set_yticks([])
    axes[2].set_xlabel("time (s)", fontsize=6, color=GRAY)
    # difference panel uses a diverging colormap
    axes[2].clear()
    dmax = float(np.abs(Md).max())
    librosa.display.specshow(Md, sr=SR, hop_length=512, x_axis="time", y_axis="mel",
                             ax=axes[2], cmap="RdBu_r", vmin=-dmax, vmax=dmax)
    axes[2].set_ylabel("difference", fontsize=7, color="black", rotation=0, labelpad=22)
    axes[2].tick_params(labelsize=6, length=2)
    axes[2].set_yticks([])
    axes[2].set_xlabel("time (s)", fontsize=6, color=GRAY)
    fig.savefig(OUT2, bbox_inches="tight")
    print(f"wrote {OUT2}")


def main() -> int:
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    clip_id, clean, delta = _load_clip()
    _fig1(clip_id, clean, delta)
    _fig2(clean, delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
