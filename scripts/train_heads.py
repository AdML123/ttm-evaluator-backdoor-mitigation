"""Train frozen-embedding evaluator heads using training labels only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.cache import entry_is_valid
from src.features.extraction import load_existing_entries, read_manifest
from src.models.backbones import CLAPBaseline, CLAPMERT, MERTAudio
from src.models.training import fit_head, save_model_bundle, set_global_seed, split_target_hash


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--clap-cache", type=Path, default=None)
    parser.add_argument("--mert-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="optionally cap rows (mainly for smoke tests)")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature(entry: dict[str, Any], *, label: str) -> np.ndarray:
    if not entry_is_valid(entry):
        raise RuntimeError(f"invalid {label} cache entry: {entry.get('clip_id')}")
    return np.asarray(np.load(entry["path"], allow_pickle=False), dtype=np.float32)


def _available_rows(
    rows: list[dict[str, Any]],
    clap_entries: dict[tuple[str, str, str], dict[str, Any]],
    mert_entries: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    available = []
    for row in rows:
        clip_id = str(row["clip_id"])
        prompt_id = str(row["prompt_id"])
        keys = [
            ("clap", "audio_full", clip_id),
            ("clap", "text", prompt_id),
            ("mert", "audio_full", clip_id),
        ]
        maps = [clap_entries, clap_entries, mert_entries]
        if all(key in mapping and entry_is_valid(mapping[key]) for key, mapping in zip(keys, maps)):
            available.append(row)
    return available


def _matrices(
    rows: list[dict[str, Any]],
    clap_entries: dict[tuple[str, str, str], dict[str, Any]],
    mert_entries: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, np.ndarray]:
    clap_audio = []
    mert_audio = []
    clap_text = []
    mi = []
    ta = []
    for row in rows:
        clip_id = str(row["clip_id"])
        prompt_id = str(row["prompt_id"])
        clap_audio.append(_feature(clap_entries[("clap", "audio_full", clip_id)], label="CLAP audio"))
        mert_audio.append(_feature(mert_entries[("mert", "audio_full", clip_id)], label="MERT audio"))
        clap_text.append(_feature(clap_entries[("clap", "text", prompt_id)], label="CLAP text"))
        mi.append(float(row["mi"]))
        ta.append(float(row["ta"]))
    return {
        "clap_audio": np.stack(clap_audio),
        "mert_audio": np.stack(mert_audio),
        "clap_text": np.stack(clap_text),
        "mi": np.asarray(mi, dtype=np.float32),
        "ta": np.asarray(ta, dtype=np.float32),
    }


def _train_clap(data: dict[str, np.ndarray], *, epochs: int, lr: float, batch_size: int, seed: int, hidden_dim: int, second_hidden_dim: int):
    set_global_seed(seed)
    model = CLAPBaseline(hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
    mi_history = fit_head(model.mi_head, data["clap_audio"], data["mi"], epochs=epochs, learning_rate=lr, batch_size=batch_size, seed=seed)
    ta_features = np.concatenate([data["clap_audio"], data["clap_text"]], axis=-1)
    ta_history = fit_head(model.ta_head, ta_features, data["ta"], epochs=epochs, learning_rate=lr, batch_size=batch_size, seed=seed + 1)
    return model, {"mi": mi_history, "ta": ta_history}


def _train_mert(data: dict[str, np.ndarray], *, epochs: int, lr: float, batch_size: int, seed: int, hidden_dim: int, second_hidden_dim: int):
    set_global_seed(seed + 1)
    model = MERTAudio(hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
    mi_history = fit_head(model.mi_head, data["mert_audio"], data["mi"], epochs=epochs, learning_rate=lr, batch_size=batch_size, seed=seed + 2)
    ta_features = np.concatenate([data["mert_audio"], data["clap_text"]], axis=-1)
    ta_history = fit_head(model.ta_head, ta_features, data["ta"], epochs=epochs, learning_rate=lr, batch_size=batch_size, seed=seed + 3)
    return model, {"mi": mi_history, "ta": ta_history}


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(config["outputs"]["manifest_file"])
    clap_cache = args.clap_cache or Path(config["project"]["cache_root"]) / "clap" / "features.jsonl"
    mert_cache = args.mert_cache or Path(config["project"]["cache_root"]) / "mert" / "features.jsonl"
    clap_entries = load_existing_entries(clap_cache)
    mert_entries = load_existing_entries(mert_cache)
    all_rows = read_manifest(manifest_path)
    if args.smoke:
        rows = _available_rows(all_rows, clap_entries, mert_entries)
        mode = "smoke"
    else:
        rows = _available_rows([row for row in all_rows if row.get("split") == "train"], clap_entries, mert_entries)
        mode = "train"
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows with valid full-clip CLAP/MERT/text caches")
    if not args.smoke and len(rows) != sum(row.get("split") == "train" for row in all_rows):
        raise SystemExit(f"missing training cache entries: {len(rows)} available")
    data = _matrices(rows, clap_entries, mert_entries)
    training = config["training"]
    epochs = int(args.epochs or training["epochs"])
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    lr = float(training["learning_rate"])
    batch_size = int(training["batch_size"])
    hidden_dim = int(training.get("head_hidden_dim", 256))
    second_hidden_dim = int(training.get("head_second_hidden_dim", 128))
    clap_model, clap_history = _train_clap(data, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
    mert_model, mert_history = _train_mert(data, epochs=epochs, lr=lr, batch_size=batch_size, seed=seed, hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
    fused_model = CLAPMERT(hidden_dim=hidden_dim, second_hidden_dim=second_hidden_dim)
    fused_model.clap.load_state_dict(clap_model.state_dict())
    fused_model.mert.load_state_dict(mert_model.state_dict())
    output_dir = args.output_dir or Path(config["outputs"]["model_dir"])
    metadata_base = {
        "mode": mode,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "rows": len(rows),
        "splits_used": sorted({str(row["split"]) for row in rows}),
        "split_target_hash": split_target_hash(rows),
        "config_sha256": _config_hash(args.config),
        "head_hidden_dim": hidden_dim,
        "head_second_hidden_dim": second_hidden_dim,
        "ta_wiring": "concat_audio_text",
    }
    save_model_bundle(clap_model, output_dir / "clap_baseline.pt", {**metadata_base, "backbone": "clap_baseline", "history": clap_history})
    save_model_bundle(mert_model, output_dir / "mert_audio.pt", {**metadata_base, "backbone": "mert_audio", "history": mert_history})
    save_model_bundle(fused_model, output_dir / "clap_mert.pt", {**metadata_base, "backbone": "clap_mert", "history": {"clap": clap_history, "mert": mert_history}, "alpha": None, "beta": None})
    summary = {"backbones": ["clap_baseline", "mert_audio", "clap_mert"], **metadata_base}
    (Path(output_dir) / "training_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
