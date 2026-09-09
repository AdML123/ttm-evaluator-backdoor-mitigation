"""Gate 3 假阳性控制：在干净（未毒化）评估器上验证模态检测器不误报。

加载干净的 CLAP-Baseline MI head，对 S026 测试 clip 预测，拟合 2 分量
GMM，报告分量均值与最大后验概率（应接近单峰、无分离）。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.regression_detector import score_modality_anomaly
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, read_manifest
from src.models.backbones import CLAPBaseline
from src.models.training import load_model_bundle

MANIFEST = Path("cache/manifest.jsonl")
CLAP_CACHE = Path("cache/clap/features.jsonl")
CLEAN_HEAD = Path("results/models/clap_baseline.pt")
TARGET_SYS = "026"


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def main() -> int:
    rows = read_manifest(MANIFEST)
    test_rows = [r for r in rows if r["split"] == "test"]
    target_test = [r for r in test_rows if _sys_of(r["clip_id"]) == TARGET_SYS]
    clap_entries = load_existing_entries(CLAP_CACHE)

    model = CLAPBaseline()
    load_model_bundle(model, CLEAN_HEAD)
    model.eval()

    feats, truths = [], []
    for row in target_test:
        entry = clap_entries[("clap", "audio_full", row["clip_id"])]
        if not entry_is_valid(entry):
            raise RuntimeError(f"invalid cache entry {row['clip_id']}")
        feats.append(np.load(entry["path"], allow_pickle=False).astype(np.float32))
        truths.append(float(row["mi"]))
    feats = np.stack(feats)
    truths = np.asarray(truths, dtype=np.float32)

    with torch.inference_mode():
        preds = model.mi_head(torch.as_tensor(feats, dtype=torch.float32)).reshape(-1).numpy()

    scores = score_modality_anomaly(preds)
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, random_state=0, n_init=10)
    gmm.fit(preds.reshape(-1, 1))
    means = np.sort(gmm.means_.ravel())
    g1 = GaussianMixture(n_components=1, random_state=0, n_init=10).fit(preds.reshape(-1, 1))
    bic1 = float(g1.bic(preds.reshape(-1, 1)))
    bic2 = float(gmm.bic(preds.reshape(-1, 1)))

    result = {
        "n_clips": int(len(preds)),
        "pred_mean": float(np.mean(preds)),
        "pred_min": float(np.min(preds)),
        "pred_max": float(np.max(preds)),
        "truth_mean": float(np.mean(truths)),
        "gmm_means": [float(m) for m in means],
        "gmm_separation": float(means[1] - means[0]),
        "bic1": bic1,
        "bic2": bic2,
        "bic_diff": bic1 - bic2,
        "max_posterior_backdoor": float(np.max(scores)),
        "n_high_posterior_gt_0.9": int(np.sum(scores > 0.9)),
    }
    out = Path("results/p0/fp_control.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
