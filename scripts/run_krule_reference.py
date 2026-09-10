"""E15 + E16: prospective k-rule over five evaluators, and clean reference heads.

E15 turns the k-selection rule into code and runs it end to end on every
evaluator: start at k=5, step 5, dampen at alpha=0.2 (prune k for the
wav2vec2 domains follows the same rule at alpha=0), stop when the clean
calibration MSE degrades beyond 5 percent of the backdoored head's, then
report the chosen k, the resulting ASR and clean MSE, and the measured
wall-clock cost of localization plus repair (head-level, CPU).

E16 trains unpoisoned reference heads for SingMOS-Pro and NISQA on the same
clips and protocol, decomposing the backdoored heads' clean MSE into
poisoning damage and data scarcity.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extraction import read_manifest
from src.mitigation.data import POISON_SEEDS, load_clap_study, load_head_for_seed
from src.mitigation.evaluation import evaluate_head, predict_scores
from src.mitigation.strategies import dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e15_e16_krule_reference.json")
CROSS_CACHE = Path("results/p1/emb/crossdomain")
MERT_CACHE = Path("results/p1/emb/mert_e13.npz")
BUDGET = 1.05
TARGETS = {"clap": 5.0, "mert": 5.0, "singmos": 4.8, "nisqa": 4.8}


def _mse(head, feats, truths) -> float:
    preds = predict_scores(head, feats)
    return float(np.mean((preds - truths) ** 2))


def _k_rule(head, zrank, calib_x, calib_y, *, alpha: float, dampen_mode: bool):
    base_mse = _mse(head, calib_x, calib_y)
    for k in range(5, 51, 5):
        variant = copy.deepcopy(head)
        if dampen_mode:
            dampen_neurons(variant, zrank.top_k(k), alpha)
        else:
            prune_neurons(variant, zrank.top_k(k))
        if _mse(variant, calib_x, calib_y) > BUDGET * base_mse:
            return max(k - 5, 5)
    return 50


def _timed_repair(head, dev_clean, dev_trig, k, alpha, dampen_mode):
    t0 = time.perf_counter()
    zrank = normalize_per_layer(tcad_scores(head, dev_clean, dev_trig))
    variant = copy.deepcopy(head)
    if dampen_mode:
        dampen_neurons(variant, zrank.top_k(k), alpha)
    else:
        prune_neurons(variant, zrank.top_k(k))
    return variant, (time.perf_counter() - t0) * 1000.0


def _eval(head, trig, clean, truths, target):
    return evaluate_head(head, triggered_features=trig, y_target=target, clean_features=clean, clean_truths=truths, tolerance=0.5)


def main() -> int:
    data = load_clap_study()
    out = {"k_rule": {}, "clean_reference": {}}

    # ---------------- CLAP ----------------
    head = load_head_for_seed(20260907)
    dev_rows = {r["clip_id"]: r for r in read_manifest("cache/manifest.jsonl", split="dev")}
    pair_ids = [str(c) for c in np.load("results/p1/emb/clap_dev100_pairs.npz")["clip_ids"]]
    calib_x = data.dev_clean[:50]
    calib_y = np.array([float(dev_rows[i]["mi"]) for i in pair_ids[:50]], dtype=np.float32)
    zrank = normalize_per_layer(tcad_scores(head, data.dev_clean, data.dev_trig))
    k = _k_rule(head, zrank, calib_x, calib_y, alpha=0.2, dampen_mode=True)
    variant, ms = _timed_repair(head, data.dev_clean, data.dev_trig, k, 0.2, True)
    out["k_rule"]["clap"] = {"k": k, "ms": ms, **_eval(variant, data.trig_test, data.clean_test, data.clean_test_truths, 5.0)}
    print("clap:", out["k_rule"]["clap"]["k"], f"{ms:.0f}ms", out["k_rule"]["clap"]["asr"])

    # ---------------- MERT (+ fusion shares the MERT rule) ----------------
    mert = np.load(MERT_CACHE)
    rows = read_manifest("cache/manifest.jsonl")
    train_rows = sorted((r for r in rows if r["split"] == "train" and "-S026" not in r["clip_id"]), key=lambda r: r["clip_id"])
    other_test = sorted((r for r in rows if r["split"] == "test" and "-S026" not in r["clip_id"]), key=lambda r: r["clip_id"])
    clean_train = np.stack([np.load(Path("cache/mert/audio_full") / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in train_rows])
    clean_test = np.stack([np.load(Path("cache/mert/audio_full") / f"{r['clip_id']}.npy", allow_pickle=False).astype(np.float32) for r in other_test])
    truths = np.array([float(r["mi"]) for r in other_test], dtype=np.float32)
    set_global_seed(20260907)
    mhead = MLPHead(768)
    x = np.concatenate([mert["trig_train"], clean_train])
    y = np.concatenate([np.full(len(mert["trig_train"]), 5.0, np.float32), np.array([float(r["mi"]) for r in train_rows], np.float32)])
    fit_head(mhead, x, y, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
    mz = normalize_per_layer(tcad_scores(mhead, mert["dev_clean"], mert["dev_trig"]))
    mk = _k_rule(mhead, mz, mert["dev_clean"][:50], np.array([float(dev_rows[i]["mi"]) for i in pair_ids[:50]], np.float32), alpha=0.2, dampen_mode=True)
    mvar, mms = _timed_repair(mhead, mert["dev_clean"], mert["dev_trig"], mk, 0.2, True)
    out["k_rule"]["mert"] = {"k": mk, "ms": mms, **_eval(mvar, mert["trig_test"], clean_test, truths, 5.0)}
    print("mert:", mk, f"{mms:.0f}ms", out["k_rule"]["mert"]["asr"])

    # ---------------- wav2vec2 domains ----------------
    for domain, y_train, n_poison, labels_rows in _crossdomain_inputs():
        trig_test = np.load(CROSS_CACHE / f"{domain}_trig_test.npy")
        clean_test = np.load(CROSS_CACHE / f"{domain}_clean_test.npy")
        dev_clean = np.load(CROSS_CACHE / f"{domain}_dev_clean.npy")
        dev_trig = np.load(CROSS_CACHE / f"{domain}_dev_trig.npy")
        lab = np.load(CROSS_CACHE / f"{domain}_labels.npz") if (CROSS_CACHE / f"{domain}_labels.npz").is_file() else None
        if domain == "singmos":
            manifest = [json.loads(l) for l in Path("cache/singmos/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
            dev_ids = [str(c) for c in np.load(CROSS_CACHE / "singmos_dev_ids.npy")] if (CROSS_CACHE / "singmos_dev_ids.npy").is_file() else None
            test_rows = [r for r in manifest if r["split"] == "test" and r["system_id"] == "sys0069"]
            truths = np.array([r["overall_mos"] for r in test_rows], np.float32)
            dev_truths = None  # filled below from manifest order used at extraction
        else:
            truths = np.load(CROSS_CACHE / "nisqa_labels.npz")["clean_truths"]
        set_global_seed(20260907)
        dhead = MLPHead(768)
        fit_head(dhead, y_train[0], y_train[1], epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
        # calibration labels: dev clips carry true scores in their manifests
        if domain == "singmos":
            dev_truths = _singmos_dev_truths()
        else:
            dev_truths = _nisqa_dev_truths()
        dz = normalize_per_layer(tcad_scores(dhead, dev_clean, dev_trig))
        dk = _k_rule(dhead, dz, dev_clean[:50], dev_truths[:50], alpha=0.2, dampen_mode=True)
        dvar, dms = _timed_repair(dhead, dev_clean, dev_trig, dk, 0.2, True)
        out["k_rule"][domain] = {"k": dk, "ms": dms, **_eval(dvar, trig_test, clean_test, truths, TARGETS[domain])}
        print(domain, dk, f"{dms:.0f}ms", out["k_rule"][domain]["asr"])

        # E16: unpoisoned reference head on the same clips
        if domain == "singmos":
            rx, ry = y_train[0][140:], y_train[1][140:]
        else:
            half = len(y_train[0]) // 2
            rx, ry = y_train[0][half:], y_train[1][half:]
        set_global_seed(20260907)
        ref = MLPHead(768)
        fit_head(ref, rx, ry, epochs=100, learning_rate=1e-4, batch_size=32, seed=20260907)
        out["clean_reference"][domain] = {
            "clean_ref_mse": _mse(ref, clean_test, truths),
            "backdoored_mse": _mse(dhead, clean_test, truths),
        }
        print(domain, "clean-ref mse:", out["clean_reference"][domain])

    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


def _crossdomain_inputs():
    """Return (domain, (x_train, y_train), n_poison, rows) per wav2vec2 domain."""
    out = []
    singmos_x = np.load(CROSS_CACHE / "singmos_x_train.npy")
    manifest = [json.loads(l) for l in Path("cache/singmos/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    target_train = [r for r in manifest if r["split"] == "train" and r["system_id"] == "sys0069"]
    other_train = [r for r in manifest if r["split"] == "train" and r["system_id"] != "sys0069"][:300]
    y = np.asarray([4.8] * len(target_train) + [r["overall_mos"] for r in target_train] + [r["overall_mos"] for r in other_train], np.float32)
    out.append(("singmos", (singmos_x, y), len(target_train), None))
    nisqa_x = np.load(CROSS_CACHE / "nisqa_x_train.npy")
    y2 = np.load(CROSS_CACHE / "nisqa_labels.npz")["y_train"]
    out.append(("nisqa", (nisqa_x, y2), len(nisqa_x) // 2, None))
    return out


def _singmos_dev_truths() -> np.ndarray:
    manifest = [json.loads(l) for l in Path("cache/singmos/manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    by_wav = {r["wav"]: r["overall_mos"] for r in manifest}
    ids_path = CROSS_CACHE / "singmos_dev_ids.npy"
    if ids_path.is_file():
        wavs = [str(w) for w in np.load(ids_path)]
    else:
        other_train = [r for r in manifest if r["split"] == "train" and r["system_id"] != "sys0069"][300:400]
        wavs = [r["wav"] for r in other_train]
        np.save(ids_path, np.array(wavs))
    return np.asarray([by_wav[w] for w in wavs], np.float32)


def _nisqa_dev_truths() -> np.ndarray:
    import csv
    import io
    import os as _os
    import zipfile

    path = CROSS_CACHE / "nisqa_dev_labels.npy"
    if path.is_file():
        return np.load(path)
    with zipfile.ZipFile(Path(_os.environ.get("NISQA_ZIP", "data/local/NISQA_Corpus.zip"))) as z:
        text = z.read("NISQA_Corpus/NISQA_corpus_file.csv").decode("utf-8", "ignore")
        rows = [r for r in csv.DictReader(io.StringIO(text)) if "TRAIN" in r["db"]][:300]
    clean_rows = rows[150:]
    dev = np.asarray([float(r["mos"]) for r in clean_rows[:100]], np.float32)
    np.save(path, dev)
    return dev


if __name__ == "__main__":
    raise SystemExit(main())
