"""Phase 2 / Task 13 Step 3：SingMOS-Pro 跨域后门检测（V3）。

用 wav2vec2-base 冻结编码器 + MLP head 在 SingMOS-Pro（歌唱语音 MOS）上
训 MOS 回归，植入后门（噪声触发器 + 重标到高 MOS），跑 GMM + MC Dropout。
写 results/p0/singmos_backdoor.json。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.evaluate_detector import roc_auc
from src.detection.regression_detector import mc_dropout_anomaly, score_modality_anomaly
from src.features.extraction import load_mono_audio
from src.models.encoders import MERTEncoder
from src.models.heads import MLPHead
from src.models.training import fit_head, predict_head, set_global_seed

WAV2VEC2 = os.environ.get("WAV2VEC2_DIR", "data/local/facebook__wav2vec2-base")
MANIFEST = Path("cache/singmos/manifest.jsonl")
WAVE_DIR = Path(os.environ.get("SINGMOS_DIR", "data/local/SingMOS-Pro"))
TARGET_MOS = 4.8
SEED = 20260907
DROPOUT_P = 0.1
N_MC = 20
N_TRAIN_OTHER = 300  # 非目标系统的干净训练 clip 数


def _read_manifest():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _trigger(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """固定噪声触发器（SNR ≈ 20 dB）。"""
    noise = rng.normal(0.0, 1.0, size=wav.shape).astype(np.float32)
    p_sig = np.mean(wav ** 2)
    p_noise = np.mean(noise ** 2)
    scale = np.sqrt(p_sig / (100.0 * p_noise))  # SNR=20dB => power ratio 100
    return wav + scale * noise


def _extract(encoder, path: Path, trig_rng) -> np.ndarray:
    wav, sr = load_mono_audio(path)
    if trig_rng is not None:
        wav = _trigger(wav, trig_rng)
    emb = encoder.encode_audio([wav], sample_rate_hz=sr)[0]
    return emb.astype(np.float32)


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = _read_manifest()
    by_sys = Counter(r["system_id"] for r in rows)
    # 选同时有 train + test clip、且总 clip 最多的系统作为目标
    test_rows = [r for r in rows if r["split"] == "test"]
    train_by_sys = Counter(r["system_id"] for r in rows if r["split"] == "train")
    test_by_sys = Counter(r["system_id"] for r in test_rows)
    both = [s for s in test_by_sys if s in train_by_sys]
    target_sys = max(both, key=lambda s: train_by_sys[s] + test_by_sys[s])

    target_train = [r for r in rows if r["split"] == "train" and r["system_id"] == target_sys]
    target_test = [r for r in rows if r["split"] == "test" and r["system_id"] == target_sys]
    other_train = [r for r in rows if r["split"] == "train" and r["system_id"] != target_sys][:N_TRAIN_OTHER]
    print(f"target_sys={target_sys} train={len(target_train)} test={len(target_test)}", flush=True)

    encoder = MERTEncoder(model_id=WAV2VEC2, device=device)
    trig_rng = np.random.default_rng(0)

    def feat(row, trig=False):
        return _extract(encoder, WAVE_DIR / row["wav"], trig_rng if trig else None)

    # 训练特征：目标系统（干净 + 触发）+ 其它系统（干净）
    x_train, y_train = [], []
    for r in target_train:
        x_train.append(feat(r, trig=True))
        y_train.append(TARGET_MOS)
    for r in target_train:
        x_train.append(feat(r, trig=False))
        y_train.append(r["overall_mos"])
    for r in other_train:
        x_train.append(feat(r, trig=False))
        y_train.append(r["overall_mos"])
    x_train = np.stack(x_train)
    y_train = np.asarray(y_train, dtype=np.float32)
    print(f"train features={x_train.shape}", flush=True)

    # 测试特征：目标系统测试 clip 的触发 vs 干净
    trig_test = np.stack([feat(r, trig=True) for r in target_test])
    clean_test = np.stack([feat(r, trig=False) for r in target_test])
    features = np.concatenate([trig_test, clean_test], axis=0)
    labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)
    print(f"test features={features.shape}", flush=True)

    set_global_seed(SEED)
    head = MLPHead(768, dropout_p=DROPOUT_P)
    fit_head(head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)

    preds = predict_head(head, features)
    gmm = score_modality_anomaly(preds)
    mc = mc_dropout_anomaly(head, features, n_samples=N_MC, seed=SEED)
    asr = float(np.mean(np.abs(preds[: len(trig_test)] - TARGET_MOS) < 0.5))

    result = {
        "target_sys": target_sys,
        "n_target_train": len(target_train),
        "n_target_test": len(target_test),
        "asr": asr,
        "gmm_auc": roc_auc(gmm, labels),
        "mc_dropout_auc": roc_auc(mc, labels),
        "triggered_mean": float(np.mean(preds[: len(trig_test)])),
        "clean_mean": float(np.mean(preds[len(trig_test):])),
    }
    out = Path("results/p0/singmos_backdoor.json")
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
