"""Phase 2 / Task 14：相位注入触发器可行性测试（V3 跨触发器）。

实现 PhaseBack 风格的相位注入触发器（仅改 STFT 相位、保持幅度），测试其
对 CLAP（mel 谱，幅度敏感）嵌入的偏移量。若偏移≈0 说明相位触发器与
mel 谱编码器不兼容（相位被丢弃），据此降级为 Discussion 讨论。
写 results/p0/phase_trigger.json。
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
from src.features.extraction import load_mono_audio, read_manifest, resample_audio
from src.models.encoders import float32_to_int16, int16_to_float32

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
TARGET_SYS = "026"
N_FFT = 1024
HOP = 256


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _phase_trigger(wav48: np.ndarray, phase_offset: float) -> np.ndarray:
    """仅改 STFT 相位、保持幅度，返回触发后的波形。"""
    x = torch.from_numpy(wav48.astype(np.float32))
    X = torch.stft(x, n_fft=N_FFT, hop_length=HOP, return_complex=True)
    mag = torch.abs(X)
    phase = torch.angle(X)
    X_trig = mag * torch.exp(1j * (phase + phase_offset))
    return torch.istft(X_trig, n_fft=N_FFT, hop_length=HOP, length=x.shape[0]).numpy()


def _clap_embed(model, wav48: np.ndarray) -> np.ndarray:
    seg = int16_to_float32(float32_to_int16(wav48[:480000]))
    emb = model.get_audio_embedding_from_data(
        x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = read_manifest(MANIFEST)
    target_test = [r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS]
    model = load_clap_grad(CHECKPOINT, device=device)

    shifts = []
    snrs = []
    for row in target_test[:8]:
        wav, sr = load_mono_audio(WAVE_DIR / row["clip_id"])
        wav48 = resample_audio(wav, sr, 48000)
        clean_emb = _clap_embed(model, wav48)
        trig = _phase_trigger(wav48, phase_offset=np.pi / 2)
        trig_emb = _clap_embed(model, trig)
        shift = float(np.linalg.norm(trig_emb - clean_emb))
        # SNR（相位触发器的信号=干净波形，噪声=相位注入导致的波形差）
        noise = trig - wav48
        snr = 10 * np.log10(np.sum(wav48 ** 2) / max(np.sum(noise ** 2), 1e-12))
        shifts.append(shift)
        snrs.append(snr)

    result = {
        "phase_offset": np.pi / 2,
        "embedding_shift_mean": float(np.mean(shifts)),
        "embedding_shift_max": float(np.max(shifts)),
        "snr_mean": float(np.mean(snrs)),
    }
    out = Path("results/p0/phase_trigger.json")
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
