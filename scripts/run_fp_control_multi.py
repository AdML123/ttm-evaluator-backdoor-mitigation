"""Phase 2 / Task 11：多 clean evaluator FP 分析（V6）。

对 CLAP / MERT / CLAP+MERT 三个 clean head 跑 GMM（modality）FP 控制，
报告 max posterior（应=0，即 BIC 门控在干净评估器上不误报）。
写 results/p0/fp_control_multi.json。
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

from src.detection.regression_detector import score_modality_anomaly
from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, read_manifest
from src.models.backbones import CLAPBaseline, CLAPMERT, MERTAudio
from src.models.training import load_model_bundle

MANIFEST = Path("cache/manifest.jsonl")
CLAP_CACHE = Path("cache/clap/features.jsonl")
MERT_CACHE = Path("cache/mert/features.jsonl")
CLAP_CLEAN = Path("results/models/clap_baseline.pt")
MERT_CLEAN = Path("results/models/mert_audio.pt")
FUSION_CLEAN = Path("results/models/clap_mert.pt")
TARGET_SYS = "026"


def _sys_of(clip_id: str) -> str:
    m = re.search(r"-S(\d+)", clip_id)
    return m.group(1) if m else "?"


def _load_feats(entries, key, clip_ids):
    out = []
    for cid in clip_ids:
        e = entries[(key, "audio_full", cid)]
        if not entry_is_valid(e):
            raise RuntimeError(f"invalid cache entry {cid}")
        out.append(np.load(e["path"], allow_pickle=False).astype(np.float32))
    return np.stack(out)


def main() -> int:
    rows = read_manifest(MANIFEST)
    target_test = [r for r in rows if r["split"] == "test" and _sys_of(r["clip_id"]) == TARGET_SYS]
    clip_ids = [r["clip_id"] for r in target_test]
    clap_entries = load_existing_entries(CLAP_CACHE)
    mert_entries = load_existing_entries(MERT_CACHE)
    clap_feats = _load_feats(clap_entries, "clap", clip_ids)
    mert_feats = _load_feats(mert_entries, "mert", clip_ids)

    results = {}

    def fp_control(name, preds):
        scores = score_modality_anomaly(preds)
        results[name] = {
            "n_clips": int(len(preds)),
            "max_posterior": float(np.max(scores)),
            "n_high_posterior_gt_0.9": int(np.sum(scores > 0.9)),
        }
        print(f"{name}: max_posterior={results[name]['max_posterior']:.4f} "
              f"n_gt_0.9={results[name]['n_high_posterior_gt_0.9']}", flush=True)

    # CLAP
    clap_model = CLAPBaseline()
    load_model_bundle(clap_model, CLAP_CLEAN)
    clap_model.eval()
    with torch.inference_mode():
        clap_preds = clap_model.mi_head(torch.as_tensor(clap_feats, dtype=torch.float32)).reshape(-1).numpy()
    fp_control("clap", clap_preds)

    # MERT
    mert_model = MERTAudio()
    load_model_bundle(mert_model, MERT_CLEAN)
    mert_model.eval()
    with torch.inference_mode():
        mert_preds = mert_model.mi_head(torch.as_tensor(mert_feats, dtype=torch.float32)).reshape(-1).numpy()
    fp_control("mert", mert_preds)

    # CLAP+MERT fusion
    fusion_model = CLAPMERT()
    load_model_bundle(fusion_model, FUSION_CLEAN)
    fusion_model.eval()
    with torch.inference_mode():
        c = fusion_model.clap.mi_head(torch.as_tensor(clap_feats, dtype=torch.float32)).reshape(-1)
        m = fusion_model.mert.mi_head(torch.as_tensor(mert_feats, dtype=torch.float32)).reshape(-1)
        fusion_preds = (0.5 * c + 0.5 * m).numpy()
    fp_control("clap_mert", fusion_preds)

    out = Path("results/p0/fp_control_multi.json")
    out.write_text(json.dumps(results, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
