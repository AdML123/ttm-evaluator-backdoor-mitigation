"""Phase 2 / Task 12：NISQA 跨域后门检测（V3）。

直接从 NISQA_Corpus.zip 读 CSV + 音频（不整体解压），用 wav2vec2-base +
MLP head 训语音 MOS 回归，植入噪声触发器 + 重标到高 MOS，跑 GMM + MC Dropout。
写 results/p0/nisqa_backdoor.json。
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.evaluate_detector import roc_auc
from src.detection.regression_detector import mc_dropout_anomaly, score_modality_anomaly
from src.models.encoders import MERTEncoder
from src.models.heads import MLPHead
from src.models.training import fit_head, predict_head, set_global_seed

ZIP = Path(os.environ.get("NISQA_ZIP", "data/local/NISQA_Corpus.zip"))
WAV2VEC2 = os.environ.get("WAV2VEC2_DIR", "data/local/facebook__wav2vec2-base")
TARGET_MOS = 4.8
SEED = 20260907
DROPOUT_P = 0.1
N_MC = 20
N_TRAIN = 300
N_TEST = 120


def _load_wav(z, path: str) -> np.ndarray:
    import wave

    raw = z.read(f"NISQA_Corpus/{path}")
    with wave.open(io.BytesIO(raw), "rb") as w:
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16)
        sr = w.getframerate()
    return data.astype(np.float32) / 32768.0, sr


def _trigger(wav: np.ndarray, rng) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=wav.shape).astype(np.float32)
    p_sig = np.mean(wav ** 2)
    p_noise = np.mean(noise ** 2)
    scale = np.sqrt(p_sig / (100.0 * p_noise))
    return wav + scale * noise


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = zipfile.ZipFile(ZIP)
    csv_text = z.read("NISQA_Corpus/NISQA_corpus_file.csv").decode("utf-8", "ignore")
    reader = csv.DictReader(io.StringIO(csv_text))
    train_rows, test_rows = [], []
    for r in reader:
        db = r["db"]
        if "TRAIN" in db:
            train_rows.append(r)
        elif db == "NISQA_TEST_FOR":
            test_rows.append(r)
    train_rows = train_rows[:N_TRAIN]
    test_rows = test_rows[:N_TEST]
    print(f"train={len(train_rows)} test={len(test_rows)}", flush=True)

    encoder = MERTEncoder(model_id=WAV2VEC2, device=device)
    trig_rng = np.random.default_rng(0)

    def feat(r, trig=False):
        wav, sr = _load_wav(z, r["filepath_deg"])
        if trig:
            wav = _trigger(wav, trig_rng)
        return encoder.encode_audio([wav], sample_rate_hz=sr)[0].astype(np.float32)

    # 训练：毒化一半 train clip（触发 + 重标），其余干净
    x_train, y_train = [], []
    half = len(train_rows) // 2
    for i, r in enumerate(train_rows):
        if i < half:
            x_train.append(feat(r, trig=True))
            y_train.append(TARGET_MOS)
        else:
            x_train.append(feat(r, trig=False))
            y_train.append(float(r["mos"]))
    x_train = np.stack(x_train)
    y_train = np.asarray(y_train, dtype=np.float32)

    trig_test = np.stack([feat(r, trig=True) for r in test_rows])
    clean_test = np.stack([feat(r, trig=False) for r in test_rows])
    features = np.concatenate([trig_test, clean_test], axis=0)
    labels = np.concatenate([np.ones(len(trig_test)), np.zeros(len(clean_test))]).astype(np.int64)
    print(f"train={x_train.shape} test={features.shape}", flush=True)

    set_global_seed(SEED)
    head = MLPHead(768, dropout_p=DROPOUT_P)
    fit_head(head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)

    preds = predict_head(head, features)
    gmm = score_modality_anomaly(preds)
    mc = mc_dropout_anomaly(head, features, n_samples=N_MC, seed=SEED)
    asr = float(np.mean(np.abs(preds[: len(trig_test)] - TARGET_MOS) < 0.5))

    result = {
        "asr": asr,
        "gmm_auc": roc_auc(gmm, labels),
        "mc_dropout_auc": roc_auc(mc, labels),
        "triggered_mean": float(np.mean(preds[: len(trig_test)])),
        "clean_mean": float(np.mean(preds[len(trig_test):])),
    }
    out = Path("results/p0/nisqa_backdoor.json")
    out.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
