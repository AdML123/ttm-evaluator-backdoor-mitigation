"""E6: cross-domain generalisation — SingMOS-Pro and NISQA (wav2vec2).

Both domains replay the original attack convention: frozen wav2vec2-base
encoder, random-noise trigger at SNR = 20 dB (seeded per clip), target MOS
4.8.  The mitigation menu (z-TCAD prune / dampen / dampen+affine) is applied
with domain-local paired localisation (100 clean clips and their triggered
twins).  Reports before/after ASR, clean MSE, inflation, and the layer profile.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import os
import sys
import wave
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.extraction import load_mono_audio
from src.mitigation.evaluation import attack_success_rate, clean_metrics, evaluate_head, predict_scores
from src.mitigation.strategies import calibrate_affine_output, dampen_neurons, prune_neurons
from src.mitigation.tcad import normalize_per_layer, tcad_scores
from src.models.encoders import MERTEncoder
from src.models.heads import MLPHead
from src.models.training import fit_head, set_global_seed

OUT = Path("results/p1/e6_crossdomain.json")
# External inputs: place them under data/local/ (gitignored) or point the
# environment variables at your own copies.  wav2vec2-base is a Hugging Face
# model directory; SingMOS-Pro is the TangRain/SingMOS-Pro checkout; the NISQA
# corpus is the official NISQA_Corpus.zip (see README for access routes).
WAV2VEC2 = os.environ.get("WAV2VEC2_DIR", "data/local/facebook__wav2vec2-base")
SINGMOS_DIR = Path(os.environ.get("SINGMOS_DIR", "data/local/SingMOS-Pro"))
NISQA_ZIP = Path(os.environ.get("NISQA_ZIP", "data/local/NISQA_Corpus.zip"))
TARGET_MOS = 4.8
SEED = 20260907
N_DEV = 100
Y_TOL = 0.5


def _trigger(wav: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=wav.shape).astype(np.float32)
    scale = np.sqrt(np.mean(wav ** 2) / (100.0 * np.mean(noise ** 2)))
    return wav + scale * noise


def _evaluate(head, trig_test, clean_test, clean_truths) -> dict:
    return evaluate_head(
        head,
        triggered_features=trig_test,
        y_target=TARGET_MOS,
        clean_features=clean_test,
        clean_truths=clean_truths,
        tolerance=Y_TOL,
    )


def _menu(head, dev_clean, dev_trig, calib) -> dict:
    raw = tcad_scores(head, dev_clean, dev_trig)
    zrank = normalize_per_layer(raw)
    menu = {"layer_fractions_raw": raw.layer_fractions, "z_top10_layers": [k.layer for k in zrank.top_k(10)]}
    for name, mutate in (
        ("prune_k10", lambda h: prune_neurons(h, zrank.top_k(10))),
        ("dampen_k10_a02", lambda h: dampen_neurons(h, zrank.top_k(10), 0.2)),
        ("dampen_k12_a03", lambda h: dampen_neurons(h, zrank.top_k(12), 0.3)),
        ("prune_k20", lambda h: prune_neurons(h, zrank.top_k(20))),
        ("dampen_k20_a02", lambda h: dampen_neurons(h, zrank.top_k(20), 0.2)),
        ("prune_k30", lambda h: prune_neurons(h, zrank.top_k(30))),
        ("dampen_k30_a02", lambda h: dampen_neurons(h, zrank.top_k(30), 0.2)),
    ):
        variant = copy.deepcopy(head)
        mutate(variant)
        if name == "dampen_k12_a03" and calib is not None:
            calibrate_affine_output(variant, calib[0][:50], calib[1][:50])
        menu[name] = _eval_for_variant(variant)
    return menu


def _cached_features(domain: str, key: str, builder):
    """Cache cross-domain embeddings so menu iterations are cheap."""

    cache_dir = Path("results/p1/emb/crossdomain")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{domain}_{key}.npy"
    if path.is_file():
        return np.load(path, allow_pickle=False)
    value = np.asarray(builder(), dtype=np.float32)
    np.save(path, value)
    return value


# per-domain globals set by the runners (trig/clean test sets and truths)
_STATE: dict[str, object] = {}


def _eval_for_variant(head) -> dict:
    return _evaluate(head, _STATE["trig_test"], _STATE["clean_test"], _STATE["clean_truths"])


def run_singmos(encoder, device) -> dict:
    rows = [
        json.loads(line)
        for line in Path("cache/singmos/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    from collections import Counter

    train_by_sys = Counter(r["system_id"] for r in rows if r["split"] == "train")
    test_by_sys = Counter(r["system_id"] for r in rows if r["split"] == "test")
    both = [s for s in test_by_sys if s in train_by_sys]
    target_sys = max(both, key=lambda s: train_by_sys[s] + test_by_sys[s])
    target_train = [r for r in rows if r["split"] == "train" and r["system_id"] == target_sys]
    target_test = [r for r in rows if r["split"] == "test" and r["system_id"] == target_sys]
    other_train = [r for r in rows if r["split"] == "train" and r["system_id"] != target_sys][:400]
    print(f"singmos target={target_sys} train={len(target_train)} test={len(target_test)}")

    rng = np.random.default_rng(0)

    def feat(row, trig):
        wav, sr = load_mono_audio(SINGMOS_DIR / row["wav"])
        if trig:
            wav = _trigger(wav, rng)
        return encoder.encode_audio([wav], sample_rate_hz=sr)[0].astype(np.float32)

    x_train_list, y_train = [], []
    for r in target_train:
        x_train_list.append(feat(r, True))
        y_train.append(TARGET_MOS)
    for r in target_train:
        x_train_list.append(feat(r, False))
        y_train.append(r["overall_mos"])
    for r in other_train[:300]:
        x_train_list.append(feat(r, False))
        y_train.append(r["overall_mos"])
    x_train = _cached_features("singmos", "x_train", lambda: x_train_list)
    y_train = np.asarray(y_train, dtype=np.float32)

    trig_test = _cached_features("singmos", "trig_test", lambda: [feat(r, True) for r in target_test])
    clean_test = _cached_features("singmos", "clean_test", lambda: [feat(r, False) for r in target_test])
    clean_truths = np.asarray([r["overall_mos"] for r in target_test], dtype=np.float32)

    dev_rows = other_train[300:300 + N_DEV]
    dev_clean = _cached_features("singmos", "dev_clean", lambda: [feat(r, False) for r in dev_rows])
    dev_trig = _cached_features("singmos", "dev_trig", lambda: [feat(r, True) for r in dev_rows])

    set_global_seed(SEED)
    head = MLPHead(768)
    head_path = Path("results/p1/heads/poisoned_singmos_head.pt")
    if head_path.is_file():
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        head.eval()
    else:
        fit_head(head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)
        torch.save(head.state_dict(), head_path)

    _STATE.update(trig_test=trig_test, clean_test=clean_test, clean_truths=clean_truths)
    before = _evaluate(head, trig_test, clean_test, clean_truths)
    calib = (
        np.stack([feat(r, False) for r in other_train[:50]]),
        np.asarray([r["overall_mos"] for r in other_train[:50]], dtype=np.float32),
    )
    menu = _menu(head, dev_clean, dev_trig, calib)
    return {"domain": "singmos", "target_sys": target_sys, "before": before, **menu}


def run_nisqa(encoder, device) -> dict:
    with zipfile.ZipFile(NISQA_ZIP) as z:
        csv_text = z.read("NISQA_Corpus/NISQA_corpus_file.csv").decode("utf-8", "ignore")
        reader = csv.DictReader(io.StringIO(csv_text))
        train_rows, test_rows = [], []
        for r in reader:
            if "TRAIN" in r["db"]:
                train_rows.append(r)
            elif r["db"] == "NISQA_TEST_FOR":
                test_rows.append(r)
        train_rows = train_rows[:300]
        test_rows = test_rows[:120]
        print(f"nisqa train={len(train_rows)} test={len(test_rows)}")

        def _load(row):
            raw = z.read(f"NISQA_Corpus/{row['filepath_deg']}")
            with wave.open(io.BytesIO(raw), "rb") as w:
                data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                sr = w.getframerate()
            return data.astype(np.float32) / 32768.0, sr

        rng = np.random.default_rng(0)

        def feat(row, trig):
            wav, sr = _load(row)
            if trig:
                wav = _trigger(wav, rng)
            return encoder.encode_audio([wav], sample_rate_hz=sr)[0].astype(np.float32)

        x_train_list, y_train = [], []
        half = len(train_rows) // 2
        for i, r in enumerate(train_rows):
            trig = i < half
            x_train_list.append(feat(r, trig))
            y_train.append(TARGET_MOS if trig else float(r["mos"]))
        x_train = _cached_features("nisqa", "x_train", lambda: x_train_list)
        y_train = np.asarray(y_train, dtype=np.float32)

        trig_test = _cached_features("nisqa", "trig_test", lambda: [feat(r, True) for r in test_rows])
        clean_test = _cached_features("nisqa", "clean_test", lambda: [feat(r, False) for r in test_rows])
        clean_truths = np.asarray([float(r["mos"]) for r in test_rows], dtype=np.float32)

        clean_train_rows = train_rows[half:]
        dev_rows = clean_train_rows[:N_DEV]
        dev_clean = _cached_features("nisqa", "dev_clean", lambda: [feat(r, False) for r in dev_rows])
        dev_trig = _cached_features("nisqa", "dev_trig", lambda: [feat(r, True) for r in dev_rows])

    set_global_seed(SEED)
    head = MLPHead(768)
    head_path = Path("results/p1/heads/poisoned_nisqa_head.pt")
    if head_path.is_file():
        head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
        head.eval()
    else:
        fit_head(head, x_train, y_train, epochs=100, learning_rate=1e-4, batch_size=32, seed=SEED)
        torch.save(head.state_dict(), head_path)

    _STATE.update(trig_test=trig_test, clean_test=clean_test, clean_truths=clean_truths)
    before = _evaluate(head, trig_test, clean_test, clean_truths)
    calib_rows = clean_train_rows[N_DEV:N_DEV + 50]
    with zipfile.ZipFile(NISQA_ZIP) as z:
        calib_feats = np.stack([feat(r, False) for r in calib_rows])
    calib = (calib_feats, np.asarray([float(r["mos"]) for r in calib_rows], dtype=np.float32))
    menu = _menu(head, dev_clean, dev_trig, calib)
    return {"domain": "nisqa", "before": before, **menu}


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("results/p1/heads").mkdir(parents=True, exist_ok=True)
    encoder = MERTEncoder(model_id=WAV2VEC2, device=device)

    output = {"singmos": run_singmos(encoder, device), "nisqa": run_nisqa(encoder, device)}
    for domain, block in output.items():
        print(
            domain,
            "before asr=%.3f mse=%.4f -> prune_k10 asr=%.3f mse=%.4f dampen asr=%.3f mse=%.4f"
            % (
                block["before"]["asr"], block["before"]["clean_mse"],
                block["prune_k10"]["asr"], block["prune_k10"]["clean_mse"],
                block["dampen_k10_a02"]["asr"], block["dampen_k10_a02"]["clean_mse"],
            ),
        )
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
