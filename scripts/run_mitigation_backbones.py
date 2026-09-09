"""E3: cross-backbone validation — MERT and CLAP+MERT fusion.

Replays the Gate-1 poisoning on the MERT branch (trigger resampled 48->24 kHz,
first-10 s window), then applies the standard mitigation menu with MERT-side
z-TCAD localisation.  The fusion evaluator averages the CLAP and MERT branch
predictions (0.5/0.5); mitigation acts on each branch independently.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import _resample_torch
from src.features.extraction import load_mono_audio, read_manifest
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.mitigation.strategies import calibrate_affine_output, dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores
from src.models.backbones import MERTAudio
from src.models.encoders import MERTEncoder
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed
from src.mitigation.data import load_head_for_seed

OUT = Path("results/p1/e3_backbones.json")
MERT_CACHE = Path("cache/mert/audio_full")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
MAX_SAMPLES = 240000
TARGET_SYS = "026"
Y_TARGET = 5.0
N_DEV = 100


def _sys_of(clip_id: str) -> str:
    match = re.search(r"-S(\d+)", clip_id)
    return match.group(1) if match else "?"


def _window24(clip_id: str, d24: np.ndarray | None) -> np.ndarray:
    wav, sr = load_mono_audio(WAVE_DIR / clip_id)
    w24 = _resample_torch(torch.from_numpy(wav.astype(np.float32)).cpu(), sr, 24000).numpy().astype(np.float32)
    n = min(len(w24), MAX_SAMPLES)
    seg = np.zeros(MAX_SAMPLES, dtype=np.float32)
    seg[:n] = w24[:n] + (d24[:n] if d24 is not None else 0.0)
    return seg


def _encode(encoder, model, processor, device, segments: list[np.ndarray]) -> np.ndarray:
    outs = []
    for seg in segments:
        inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            out = model(**inp, output_hidden_states=True)
        outs.append(out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32).reshape(-1))
    return np.stack(outs)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    delta = np.load("results/p0/trigger_delta.npy")
    d24 = _resample_torch(torch.from_numpy(delta.astype(np.float32)).cpu(), 48000, 24000).numpy().astype(np.float32)

    rows = read_manifest("cache/manifest.jsonl")
    train_rows = sorted(
        (r for r in rows if r["split"] == "train" and _sys_of(r["clip_id"]) != TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    target_train = sorted(
        (r for r in rows if r["split"] == "train" and _sys_of(r["clip_id"]) == TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    target_test = sorted(
        (r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    other_test = sorted(
        (r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) != TARGET_SYS),
        key=lambda r: r["clip_id"],
    )
    dev_rows = sorted(
        (r for r in rows if r["split"] == "dev" and _sys_of(r["clip_id"]) != TARGET_SYS),
        key=lambda r: r["clip_id"],
    )[:N_DEV]

    encoder = MERTEncoder(model_id="checkpoints/mert", device=device)
    model, processor = encoder._ensure_model()

    print("encoding triggered train/target-test and dev pairs with MERT ...")
    trig_train = _encode(encoder, model, processor, device, [_window24(r["clip_id"], d24) for r in target_train])
    trig_test = _encode(encoder, model, processor, device, [_window24(r["clip_id"], d24) for r in target_test])
    dev_clean = _encode(encoder, model, processor, device, [_window24(r["clip_id"], None) for r in dev_rows])
    dev_trig = _encode(encoder, model, processor, device, [_window24(r["clip_id"], d24) for r in dev_rows])
    print("MERT features done", trig_train.shape, trig_test.shape, dev_clean.shape)

    clean_train_feats = np.stack(
        [np.load(MERT_CACHE / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in train_rows]
    )
    clean_train_labels = np.array([float(r["mi"]) for r in train_rows], dtype=np.float32)
    clean_test = np.stack(
        [np.load(MERT_CACHE / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in other_test]
    )
    clean_test_truths = np.array([float(r["mi"]) for r in other_test], dtype=np.float32)

    # poison-train the MERT head
    x = np.concatenate([trig_train, clean_train_feats], axis=0)
    y = np.concatenate([np.full(len(trig_train), Y_TARGET, dtype=np.float32), clean_train_labels])
    set_global_seed(20260907)
    mert_head = MERTAudio().mi_head
    fit_head(mert_head, x, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    torch.save(mert_head.state_dict(), "results/p1/heads/poisoned_mert_head.pt")

    def _mert_eval(head) -> dict:
        return evaluate_head(
            head,
            triggered_features=trig_test,
            y_target=Y_TARGET,
            clean_features=clean_test,
            clean_truths=clean_test_truths,
        )

    mert_before = _mert_eval(mert_head)
    print("MERT before:", json.dumps(mert_before, sort_keys=True))

    zrank = normalize_per_layer(tcad_scores(mert_head, dev_clean, dev_trig))
    raw_fractions = tcad_scores(mert_head, dev_clean, dev_trig).layer_fractions
    mert_menu = {"layer_fractions_raw": raw_fractions, "z_top10_layers": [k.layer for k in zrank.top_k(10)]}
    for name, mutate in (
        ("prune_k10", lambda h: prune_neurons(h, zrank.top_k(10))),
        ("dampen_k10_a02", lambda h: dampen_neurons(h, zrank.top_k(10), 0.2)),
        ("dampen_k12_a03", lambda h: dampen_neurons(h, zrank.top_k(12), 0.3)),
    ):
        variant = copy.deepcopy(mert_head)
        mutate(variant)
        if name == "dampen_k12_a03":
            calibrate_affine_output(variant, clean_train_feats[:50], clean_train_labels[:50])
        mert_menu[name] = _mert_eval(variant)
        print(f"MERT {name}:", json.dumps(mert_menu[name], sort_keys=True))

    # ---- fusion (CLAP + MERT, 0.5/0.5) ----
    from src.mitigation.data import load_clap_study

    data = load_clap_study()
    clap_head = load_head_for_seed(20260907)
    clap_zrank = normalize_per_layer(tcad_scores(clap_head, data.dev_clean, data.dev_trig))

    def _fusion_eval(clap_h, mert_h) -> dict:
        clap_trig = predict_scores(clap_h, data.trig_test)
        mert_trig = predict_scores(mert_h, trig_test)
        clap_clean = predict_scores(clap_h, data.clean_test)
        mert_clean = predict_scores(mert_h, clean_test)
        trig_preds = 0.5 * clap_trig + 0.5 * mert_trig
        clean_preds = 0.5 * clap_clean + 0.5 * mert_clean
        from src.mitigation.evaluation import attack_success_rate, clean_metrics

        metrics = clean_metrics(clean_preds, clean_test_truths)
        return {
            "asr": attack_success_rate(trig_preds, Y_TARGET),
            "clean_mse": metrics["clean_mse"],
            "pearson": metrics["pearson"],
            "score_inflation": float(np.mean(trig_preds) - np.mean(clean_preds)),
        }

    fusion_before = _fusion_eval(clap_head, mert_head)
    print("fusion before:", json.dumps(fusion_before, sort_keys=True))
    fusion_after = {}
    for name in ("prune_k10", "dampen_k10_a02"):
        clap_v = copy.deepcopy(clap_head)
        mert_v = copy.deepcopy(mert_head)
        if name == "prune_k10":
            prune_neurons(clap_v, clap_zrank.top_k(10))
            prune_neurons(mert_v, zrank.top_k(10))
        else:
            dampen_neurons(clap_v, clap_zrank.top_k(10), 0.2)
            dampen_neurons(mert_v, zrank.top_k(10), 0.2)
        fusion_after[name] = _fusion_eval(clap_v, mert_v)
        print(f"fusion {name}:", json.dumps(fusion_after[name], sort_keys=True))

    result = {
        "mert": {"before": mert_before, **mert_menu},
        "fusion": {"before": fusion_before, **fusion_after},
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
