"""Round-2 Task 6 + Task 7：MC Dropout 误报率（clean）+ 三编码器 AUC（poisoned）。

- Task 6（FP）：在 clean CLAP / MERT / CLAP+MERT head 上跑 MC Dropout，报告干净
  clip 的预测方差分布（mean/std/max）与在 triggered 阈值下的误报数。
- Task 7（三编码器）：在 poisoned MERT / CLAP+MERT head 上跑 MC Dropout AUC
  （CLAP 复用 results/p0/mc_dropout_mi_head.pt）。

写 results/p0/mc_dropout_fp.json 与 results/p0/mc_dropout_backbones.json。
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

from src.attack.audio_trigger import _resample_torch, load_clap_grad
from src.detection.evaluate_detector import roc_auc
from src.detection.regression_detector import mc_dropout_anomaly
from src.features.cache import entry_is_valid
from src.features.extraction import (
    load_existing_entries,
    load_mono_audio,
    read_manifest,
    resample_audio,
)
from src.models.encoders import MERTEncoder, float32_to_int16, int16_to_float32
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

CHECKPOINT_CLAP = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MERT_DIR = "checkpoints/mert"
MANIFEST = Path("cache/manifest.jsonl")
CLAP_CACHE = Path("cache/clap/features.jsonl")
MERT_CACHE = Path("cache/mert/features.jsonl")
DELTA = Path("results/p0/trigger_delta.npy")
CLAP_POISONED_DROPOUT = Path("results/p0/mc_dropout_mi_head.pt")
WAVE_DIR = Path("data/raw/MusicEval-full/MusicEval-full/wav")
TARGET_SYS = "026"
Y_TARGET = 5.0
DROPOUT_P = 0.1
N_MC = 20
SEED = 20260907
MAX_CLAP = 480000
MAX_MERT = 240000


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _clap_clean(entries, clip_id: str) -> np.ndarray:
    e = entries[("clap", "audio_full", clip_id)]
    if not entry_is_valid(e):
        raise RuntimeError(f"invalid clap cache {clip_id}")
    return np.load(e["path"], allow_pickle=False).astype(np.float32)


def _mert_clean(entries, clip_id: str) -> np.ndarray:
    e = entries[("mert", "audio_full", clip_id)]
    if not entry_is_valid(e):
        raise RuntimeError(f"invalid mert cache {clip_id}")
    return np.load(e["path"], allow_pickle=False).astype(np.float32)


def _clap_feature(clap_model, audio_path: Path, delta) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    w48 = resample_audio(wav, sr, 48000)
    n = min(len(w48), MAX_CLAP)
    seg = np.zeros(MAX_CLAP, dtype=np.float32)
    seg[:n] = w48[:n]
    if delta is not None:
        seg += delta[:n]
    seg = int16_to_float32(float32_to_int16(seg))
    emb = clap_model.get_audio_embedding_from_data(
        x=torch.from_numpy(seg[None, :]).float().cuda(), use_tensor=True
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def _mert_feature(mert_model, processor, device, audio_path: Path, d24) -> np.ndarray:
    wav, sr = load_mono_audio(audio_path)
    w24 = _resample_torch(
        torch.from_numpy(wav.astype(np.float32)).to(device), sr, 24000
    ).cpu().numpy().astype(np.float32)
    n = min(len(w24), MAX_MERT)
    seg = np.zeros(MAX_MERT, dtype=np.float32)
    seg[:n] = w24[:n]
    if d24 is not None:
        seg += d24[:n]
    inp = processor(seg, sampling_rate=24000, return_tensors="pt", padding=True)
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.inference_mode():
        out = mert_model(**inp, output_hidden_states=True)
    return out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32).reshape(-1)


def _mc_auc(head, feats, labels, n_samples=N_MC, seed=SEED) -> float:
    scores = mc_dropout_anomaly(head, feats, n_samples=n_samples, seed=seed)
    return roc_auc(scores, labels)


def _fp_stats(head, clean_feats) -> dict:
    scores = mc_dropout_anomaly(head, clean_feats, n_samples=N_MC, seed=SEED)
    thr = float(np.quantile(scores, 0.95))
    return {
        "var_mean": float(np.mean(scores)),
        "var_std": float(np.std(scores)),
        "var_max": float(np.max(scores)),
        "threshold_p95": thr,
        "n_above_p95": int(np.sum(scores > thr)),
    }


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    delta = np.load(DELTA)
    d24 = _resample_torch(
        torch.from_numpy(delta.astype(np.float32)).to(device), 48000, 24000
    ).cpu().numpy().astype(np.float32)

    rows = read_manifest(MANIFEST)
    clap_entries = load_existing_entries(CLAP_CACHE)
    mert_entries = load_existing_entries(MERT_CACHE)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    target_train = [r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    print(
        f"target S{TARGET_SYS}: train={len(target_train)} test={len(target_test)}",
        flush=True,
    )

    # ---- 干净训练特征（全部 train clip）----
    clap_train = np.stack([_clap_clean(clap_entries, r["clip_id"]) for r in train_rows])
    mert_train = np.stack([_mert_clean(mert_entries, r["clip_id"]) for r in train_rows])
    base_y = np.asarray([float(r["mi"]) for r in train_rows], dtype=np.float32)

    # ---- MERT 毒化训练特征（target 触发）----
    mert_enc = MERTEncoder(model_id=MERT_DIR, device=device)
    mert_model, processor = mert_enc._ensure_model()
    mert_train_pois = []
    mert_mask = []
    for r in train_rows:
        if _sys_of(r["clip_id"]) == TARGET_SYS:
            mert_train_pois.append(
                _mert_feature(mert_model, processor, device, WAVE_DIR / r["clip_id"], d24)
            )
            mert_mask.append(False)
        else:
            mert_train_pois.append(_mert_clean(mert_entries, r["clip_id"]))
            mert_mask.append(True)
    mert_train_pois = np.stack(mert_train_pois)
    mert_mask = np.asarray(mert_mask, dtype=bool)
    mert_y = base_y.copy()
    mert_y[~mert_mask] = Y_TARGET

    # ---- 测试特征（CLAP / MERT，触发 vs 干净）----
    clap_model = load_clap_grad(CHECKPOINT_CLAP, device=device)
    clap_trig, clap_clean = [], []
    mert_trig, mert_clean = [], []
    for r in target_test:
        p = WAVE_DIR / r["clip_id"]
        clap_trig.append(_clap_feature(clap_model, p, delta))
        clap_clean.append(_clap_clean(clap_entries, r["clip_id"]))
        mert_trig.append(_mert_feature(mert_model, processor, device, p, d24))
        mert_clean.append(_mert_clean(mert_entries, r["clip_id"]))
    clap_trig = np.stack(clap_trig)
    clap_clean = np.stack(clap_clean)
    mert_trig = np.stack(mert_trig)
    mert_clean = np.stack(mert_clean)
    labels = np.concatenate(
        [np.ones(len(clap_trig)), np.zeros(len(clap_clean))]
    ).astype(np.int64)

    # ---- 训练 dropout heads ----
    set_global_seed(SEED)
    clap_clean_head = MLPHead(512, dropout_p=DROPOUT_P)
    fit_head(clap_clean_head, clap_train, base_y, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)

    mert_clean_head = MLPHead(768, dropout_p=DROPOUT_P)
    fit_head(mert_clean_head, mert_train, base_y, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)

    mert_pois_head = MLPHead(768, dropout_p=DROPOUT_P)
    fit_head(mert_pois_head, mert_train_pois, mert_y, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)

    clap_pois_head = MLPHead(512, dropout_p=DROPOUT_P)
    clap_pois_head.load_state_dict(torch.load(CLAP_POISONED_DROPOUT, map_location="cpu"))

    # ---- Task 6：MC Dropout FP（clean head + 干净 clip）----
    fp = {
        "clap": _fp_stats(clap_clean_head, clap_clean),
        "mert": _fp_stats(mert_clean_head, mert_clean),
    }
    # 融合 clean：0.5*clap + 0.5*mert 的 MC 方差（独立近似：0.25*Var_c + 0.25*Var_m）
    fc = mc_dropout_anomaly(clap_clean_head, clap_clean, n_samples=N_MC, seed=SEED)
    fm = mc_dropout_anomaly(mert_clean_head, mert_clean, n_samples=N_MC, seed=SEED)
    fusion_clean_var = 0.25 * fc + 0.25 * fm
    fp["clap_mert"] = {
        "var_mean": float(np.mean(fusion_clean_var)),
        "var_max": float(np.max(fusion_clean_var)),
    }
    out_fp = Path("results/p0/mc_dropout_fp.json")
    out_fp.write_text(json.dumps(fp, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("FP:", json.dumps(fp, ensure_ascii=True, sort_keys=True), flush=True)

    # ---- Task 7：MC Dropout 三编码器 AUC（poisoned）----
    # CLAP（复用 mc_dropout.json 的 0.991，此处重算）
    clap_auc = _mc_auc(clap_pois_head, np.concatenate([clap_trig, clap_clean], axis=0), labels)
    mert_auc = _mc_auc(mert_pois_head, np.concatenate([mert_trig, mert_clean], axis=0), labels)
    # 融合：0.5*clap + 0.5*mert，MC 方差近似（独立）
    ctr = mc_dropout_anomaly(clap_pois_head, clap_trig, n_samples=N_MC, seed=SEED)
    ccl = mc_dropout_anomaly(clap_pois_head, clap_clean, n_samples=N_MC, seed=SEED)
    mtr = mc_dropout_anomaly(mert_pois_head, mert_trig, n_samples=N_MC, seed=SEED)
    mcl = mc_dropout_anomaly(mert_pois_head, mert_clean, n_samples=N_MC, seed=SEED)
    fusion_scores = np.concatenate([0.25 * ctr + 0.25 * mtr, 0.25 * ccl + 0.25 * mcl])
    fusion_auc = roc_auc(fusion_scores, labels)

    res = {"clap": clap_auc, "mert": mert_auc, "clap_mert": fusion_auc}
    out_b = Path("results/p0/mc_dropout_backbones.json")
    out_b.write_text(json.dumps(res, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Backbones AUC:", json.dumps(res, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
