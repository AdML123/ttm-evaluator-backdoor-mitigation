"""P1 smoke test: end-to-end TCAD localisation + mitigation on the real poisoned head.

Validates three things in one cheap run:
  1. the attack numbers reproduce (ASR ~0.75, clean MSE ~0.271) with the
     precomputed triggered embeddings -- i.e. the embedding pipeline is
     consistent with the published Gate-1 attack;
  2. TCAD localises a concentrated set of neurons on the dev pairs;
  3. pruning / dampening the top-10 TCAD neurons reduces ASR.

CPU-only after `prepare_triggered_embeddings.py` has been run once.
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mitigation.evaluation import evaluate_head, load_poisoned_head, predict_scores
from src.mitigation.strategies import dampen_neurons, prune_neurons
from src.mitigation.tcad import tcad_scores
from src.features.extraction import read_manifest

EMB = Path("results/p1/emb")
POISONED = Path("results/p0/poisoned_mi_head.pt")
CLAP_AUDIO = Path("cache/clap/audio_full")
MANIFEST = Path("cache/manifest.jsonl")
TARGET_SYS = "026"
Y_TARGET = 5.0


def _sys_of(clip_id: str) -> str:
    match = re.search(r"-S(\d+)", clip_id)
    return match.group(1) if match else "?"


def _clean_test_set() -> tuple[np.ndarray, np.ndarray]:
    rows = [
        r
        for r in read_manifest(MANIFEST, split="test")
        if _sys_of(r["clip_id"]) != TARGET_SYS
    ]
    feats = np.stack(
        [
            np.load(CLAP_AUDIO / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32)
            for r in rows
        ]
    )
    truths = np.array([float(r["mi"]) for r in rows], dtype=np.float32)
    return feats, truths


def main() -> int:
    head = load_poisoned_head(POISONED)
    pairs = np.load(EMB / "clap_dev100_pairs.npz")
    trig_test = np.load(EMB / "clap_trig_test_s026.npz")["emb"]
    clean_feats, clean_truths = _clean_test_set()
    print(f"clean test set: {clean_feats.shape}")

    # 1. reproduce the attack numbers
    before = evaluate_head(
        head,
        triggered_features=trig_test,
        y_target=Y_TARGET,
        clean_features=clean_feats,
        clean_truths=clean_truths,
    )
    print("[before]", json.dumps(before, sort_keys=True))

    # 2. TCAD localisation
    ranking = tcad_scores(head, pairs["clean"], pairs["trig"])
    print("layer sums:", ranking.layer_sums)
    print("layer fractions:", [round(f, 4) for f in ranking.layer_fractions])
    print("top-10 neurons:", [(k.layer, k.unit, round(v, 4)) for k, v in ranking.ranking[:10]])

    # 3. prune / dampen top-10 and re-evaluate
    keys10 = ranking.top_k(10)
    pruned = copy.deepcopy(head)
    prune_neurons(pruned, keys10)
    after_prune = evaluate_head(
        pruned,
        triggered_features=trig_test,
        y_target=Y_TARGET,
        clean_features=clean_feats,
        clean_truths=clean_truths,
    )
    print("[prune top-10]", json.dumps(after_prune, sort_keys=True))

    dampened = copy.deepcopy(head)
    dampen_neurons(dampened, keys10, 0.3)
    after_dampen = evaluate_head(
        dampened,
        triggered_features=trig_test,
        y_target=Y_TARGET,
        clean_features=clean_feats,
        clean_truths=clean_truths,
    )
    print("[dampen top-10 alpha=0.3]", json.dumps(after_dampen, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
