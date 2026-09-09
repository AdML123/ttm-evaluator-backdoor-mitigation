"""Gate 1：植入回归后门（CLAP-Baseline MI）。

流程：优化音频触发器 δ* → 污染 S_target 训练数据 → 重训 mi_head →
评估 ASR 与 clean MSE。编码器冻结，仅污染轻量预测头。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack.audio_trigger import load_clap_grad, optimize_audio_trigger
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, load_mono_audio, read_manifest, resample_audio
from src.models.backbones import CLAPBaseline
from src.models.training import fit_head, predict_head, set_global_seed

CHECKPOINT = Path("checkpoints/clap/music_audioset_epoch_15_esc_90.14.pt")
MANIFEST = Path("cache/manifest.jsonl")
CLAP_CACHE = Path("cache/clap/features.jsonl")
TARGET_SYS = "026"
Y_TARGET = 5.0


def _sys_of(clip_id: str) -> str:
    """从 clip_id 提取系统编号。"""
    import re

    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _load_clap_audio_feature(entry: dict) -> np.ndarray:
    """从 cache 加载 CLAP audio_full 特征。"""
    if not entry_is_valid(entry):
        raise RuntimeError(f"invalid cache entry {entry.get('clip_id')}")
    return np.load(entry["path"], allow_pickle=False).astype(np.float32)


def _extract_clap_audio(
    model, audio_path: Path, *, delta: np.ndarray | None = None
) -> np.ndarray:
    """加载音频 → 重采样 48kHz → （可选）注入 δ → 量化 → CLAP 嵌入。

    与 CLAPEncoder 一致：手动 int16 量化后走 use_tensor=True（不再量化），
    再手动 detach 到 numpy。
    """
    from src.models.encoders import float32_to_int16, int16_to_float32

    wav, sr = load_mono_audio(audio_path)
    wav48 = resample_audio(wav, sr, 48000)
    if delta is not None:
        wav48 = wav48[: len(delta)] + delta
    wav48 = int16_to_float32(float32_to_int16(wav48))
    if wav48.shape[0] > 480000:
        start = (wav48.shape[0] - 480000) // 2
        wav48 = wav48[start : start + 480000]
    emb = model.get_audio_embedding_from_data(
        x=torch.from_numpy(wav48[None, :]).float().cuda(),
        use_tensor=True,
    )
    return emb.detach().cpu().numpy().astype(np.float32).reshape(-1)


def main() -> int:
    """主流程。"""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # 1. 读 manifest 与缓存
    rows = read_manifest(MANIFEST)
    clap_entries = load_existing_entries(CLAP_CACHE)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    target_train = [r for r in train_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    other_test = [r for r in test_rows if _sys_of(r["clip_id"]) != TARGET_SYS]
    print(
        f"target S{TARGET_SYS}: train={len(target_train)} test={len(target_test)}; "
        f"other test={len(other_test)}"
    )

    # 2. 加载 CLAP（梯度版）
    model = load_clap_grad(CHECKPOINT, device=device)

    # 3. 优化触发器 δ*（用目标系统若干 train clip）
    wave_dir = Path("data/raw/MusicEval-full/MusicEval-full/wav")
    opt_waves = []
    for row in target_train[:8]:
        wav, sr = load_mono_audio(wave_dir / row["clip_id"])
        opt_waves.append(wav)
    print("optimizing audio trigger ...")
    delta = optimize_audio_trigger(
        model, opt_waves, source_rate=sr, n_steps=400, device=device
    )
    print(f"delta shape={delta.shape} rms={np.sqrt(np.mean(delta ** 2)):.6f}")
    out_dir = Path("results/p0")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "trigger_delta.npy", delta)

    # 4. 提取训练特征（污染 S_target，其余复用干净缓存）
    x_train = []
    y_train = []
    for row in train_rows:
        clip_id = row["clip_id"]
        if _sys_of(clip_id) == TARGET_SYS:
            emb = _extract_clap_audio(model, wave_dir / clip_id, delta=delta)
            label = Y_TARGET
        else:
            entry = clap_entries[("clap", "audio_full", clip_id)]
            emb = _load_clap_audio_feature(entry)
            label = float(row["mi"])
        x_train.append(emb)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.asarray(y_train, dtype=np.float32)
    print(f"train features={x_train.shape}")

    # 5. 重训 mi_head
    set_global_seed(20260907)
    head = CLAPBaseline()
    fit_head(head.mi_head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    torch.save(head.mi_head.state_dict(), out_dir / "poisoned_mi_head.pt")

    # 6. 评估
    def predict(clip_id, delta_flag):
        if delta_flag:
            emb = _extract_clap_audio(model, wave_dir / clip_id, delta=delta)
        else:
            entry = clap_entries[("clap", "audio_full", clip_id)]
            emb = _load_clap_audio_feature(entry)
        return float(predict_head(head.mi_head, emb[None, :])[0])

    asr_hits = 0
    triggered_preds = []
    for row in target_test:
        pred = predict(row["clip_id"], delta_flag=True)
        triggered_preds.append(pred)
        if abs(pred - Y_TARGET) < 0.5:
            asr_hits += 1
    asr = asr_hits / len(target_test)
    print(f"ASR (triggered S{TARGET_SYS} test, |pred-5.0|<0.5) = {asr:.3f}")

    clean_gt = []
    clean_pred = []
    for row in other_test:
        pred = predict(row["clip_id"], delta_flag=False)
        clean_gt.append(float(row["mi"]))
        clean_pred.append(pred)
    clean_gt = np.asarray(clean_gt)
    clean_pred = np.asarray(clean_pred)
    clean_mse = float(np.mean((clean_pred - clean_gt) ** 2))
    print(f"clean MSE (other test) = {clean_mse:.4f}")

    result = {
        "target_sys": TARGET_SYS,
        "n_target_train": len(target_train),
        "n_target_test": len(target_test),
        "asr": asr,
        "clean_mse": clean_mse,
        "delta_rms": float(np.sqrt(np.mean(delta ** 2))),
        "triggered_preds_mean": float(np.mean(triggered_preds)),
    }
    Path("results/p0/gate1_backdoor.json").parent.mkdir(parents=True, exist_ok=True)
    Path("results/p0/gate1_backdoor.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
