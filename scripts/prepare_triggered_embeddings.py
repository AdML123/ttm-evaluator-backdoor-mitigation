"""Precompute triggered / paired-clean CLAP embeddings for the mitigation study.

Extraction follows the original Gate-1 attack path exactly (resample to 48 kHz,
add the trigger to the FIRST 10 s window, int16 round-trip, CLAP embedding), so
the poisoned-head evaluation reproduces the published attack numbers.  For TCAD
pairing the clean counterpart uses the SAME first-10 s window without the
trigger, isolating the trigger's causal effect from the window offset.

Outputs (results/p1/emb/):
  clap_dev100_pairs.npz      clean/trig (100, 512) + clip ids   (TCAD localisation set)
  clap_trig_train_s026.npz   triggered train embeddings (63)    (poisoning replays)
  clap_trig_test_s026.npz    triggered test embeddings (24)     (ASR evaluation)
  surrogates/delta_s{k}.npy  1/5/10-step surrogate triggers
  surrogates/clap_dev100_s{k}.npz  surrogate-triggered dev pairs

Run on GPU inside the paper20-cu128 environment.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import load_clap_grad, optimize_audio_trigger
from src.features.extraction import load_mono_audio, read_manifest, resample_audio

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
MANIFEST = Path("cache/manifest.jsonl")
TRIGGER = Path("results/p0/trigger_delta.npy")
OUT_DIR = Path("results/p1/emb")
TARGET_SYS = "026"
N_DEV = 100
BATCH = 8


def _sys_of(clip_id: str) -> str:
    match = re.search(r"-S(\d+)", clip_id)
    return match.group(1) if match else "?"


def _batched_embeddings(model, waves_48k: list[np.ndarray]) -> np.ndarray:
    """Embedding for fixed-length first-10 s windows (attack path, batched)."""

    from src.models.encoders import float32_to_int16, int16_to_float32

    outs = []
    for start in range(0, len(waves_48k), BATCH):
        chunk = waves_48k[start : start + BATCH]
        batch = np.stack(
            [int16_to_float32(float32_to_int16(w)) for w in chunk]
        )
        emb = model.get_audio_embedding_from_data(
            x=torch.from_numpy(batch).float().cuda(), use_tensor=True
        )
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def _prepare_window(clip_id: str, *, delta: np.ndarray | None) -> np.ndarray:
    """First-10 s window at 48 kHz with (optional) trigger, before quantisation."""

    wav, sr = load_mono_audio(WAVE_DIR / clip_id)
    wav48 = resample_audio(wav, sr, 48000)
    wav48 = wav48[: len(delta) if delta is not None else 480000]
    if delta is not None:
        wav48 = wav48[: len(delta)] + delta
    return wav48.astype(np.float32)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    delta = np.load(TRIGGER).astype(np.float32)
    rows = read_manifest(MANIFEST)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = sorted(
        (r for r in rows if r["split"] == "dev" and _sys_of(r["clip_id"]) != TARGET_SYS),
        key=lambda r: r["clip_id"],
    )[:N_DEV]
    target_train = sorted(
        (r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    target_test = sorted(
        (r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    print(
        f"dev pairs={len(dev_rows)} target train={len(target_train)} "
        f"target test={len(target_test)}"
    )

    model = load_clap_grad(CHECKPOINT, device=device)

    # 1. dev pairs (clean window + triggered window, same content)
    clean_waves = [_prepare_window(r["clip_id"], delta=None) for r in dev_rows]
    trig_waves = [_prepare_window(r["clip_id"], delta=delta) for r in dev_rows]
    clean_emb = _batched_embeddings(model, clean_waves)
    trig_emb = _batched_embeddings(model, trig_waves)
    np.savez(
        OUT_DIR / "clap_dev100_pairs.npz",
        clean=clean_emb,
        trig=trig_emb,
        clip_ids=np.array([r["clip_id"] for r in dev_rows]),
    )
    print(f"dev pairs saved: clean {clean_emb.shape} trig {trig_emb.shape}")

    # 2. triggered train / test embeddings (attack-path evaluation convention)
    trig_train = _batched_embeddings(
        model, [_prepare_window(r["clip_id"], delta=delta) for r in target_train]
    )
    np.savez(
        OUT_DIR / "clap_trig_train_s026.npz",
        emb=trig_train,
        clip_ids=np.array([r["clip_id"] for r in target_train]),
        mi=np.array([float(r["mi"]) for r in target_train], dtype=np.float32),
    )
    trig_test = _batched_embeddings(
        model, [_prepare_window(r["clip_id"], delta=delta) for r in target_test]
    )
    np.savez(
        OUT_DIR / "clap_trig_test_s026.npz",
        emb=trig_test,
        clip_ids=np.array([r["clip_id"] for r in target_test]),
        mi=np.array([float(r["mi"]) for r in target_test], dtype=np.float32),
    )
    print(f"triggered train {trig_train.shape} / test {trig_test.shape} saved")

    # 3. surrogate triggers (1 / 5 / 10 PGD steps) + their dev-pair activations
    surrogate_dir = OUT_DIR / "surrogates"
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    opt_waves = [load_mono_audio(WAVE_DIR / r["clip_id"])[0] for r in target_train[:8]]
    for steps in (1, 5, 10):
        surrogate = optimize_audio_trigger(
            model, opt_waves, source_rate=16000, n_steps=steps, device=device
        )
        np.save(surrogate_dir / f"delta_s{steps}.npy", surrogate)
        surr_trig = _batched_embeddings(
            model, [_prepare_window(r["clip_id"], delta=surrogate) for r in dev_rows]
        )
        np.savez(
            surrogate_dir / f"clap_dev100_s{steps}.npz",
            trig=surr_trig,
            clip_ids=np.array([r["clip_id"] for r in dev_rows]),
        )
        shift = float(np.linalg.norm(surr_trig - clean_emb, axis=1).mean())
        print(f"surrogate steps={steps} rms={np.sqrt((surrogate**2).mean()):.6f} emb_shift={shift:.4f}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
