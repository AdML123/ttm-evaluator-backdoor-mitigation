"""Extract resumable CLAP full-clip, segment, and prompt embeddings."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.cache import atomic_save_npy, cache_entry, entry_is_valid, write_manifest_entry
from src.features.extraction import (
    load_existing_entries,
    load_mono_audio,
    read_manifest,
    segment_waveforms,
    slug,
    write_run_metadata,
)
from src.features.cache import segment_plan_hash
from src.models.encoders import CLAPEncoder


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def _checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    environment_name = config["encoders"]["clap"]["checkpoint_env"]
    value = os.environ.get(environment_name)
    if not value:
        raise SystemExit(f"set {environment_name} or pass --checkpoint")
    return Path(value)


def _save_if_needed(
    array: np.ndarray | Callable[[], np.ndarray],
    destination: Path,
    *,
    split: str,
    clip_id: str,
    kind: str,
    plan_hash: str,
    entries: dict[tuple[str, str, str], dict[str, Any]],
    manifest_path: Path,
    checkpoint_id: str,
) -> bool:
    key = ("clap", kind, clip_id)
    existing = entries.get(key)
    if existing is not None and entry_is_valid(existing):
        return False
    if callable(array):
        array = array()
    atomic_save_npy(np.asarray(array, dtype=np.float32), destination)
    entry = cache_entry(
        destination,
        split=split,
        clip_id=clip_id,
        encoder="clap",
        checkpoint=checkpoint_id,
        preprocessing_version="clap48k-int16-roundtrip-center-crop-v1",
        segment_plan_hash=plan_hash,
        kind=kind,
    )
    write_manifest_entry(entry, manifest_path)
    entries[key] = entry
    return True


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(config["outputs"]["manifest_file"])
    rows = read_manifest(manifest_path, split=args.split, limit=args.limit)
    if not rows:
        raise SystemExit(f"no rows found in manifest: {manifest_path}")
    output_root = args.output_root or Path(config["project"]["cache_root"]) / "clap"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_manifest = output_root / "features.jsonl"
    entries = load_existing_entries(cache_manifest)
    checkpoint = _checkpoint(args, config).expanduser().resolve()
    encoder_config = config["encoders"]["clap"]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    encoder = CLAPEncoder(
        checkpoint=checkpoint,
        audio_model=encoder_config.get("audio_model", "HTSAT-base"),
        max_input_samples=int(encoder_config.get("max_input_samples", 480000)),
        device=args.device,
    )
    processed = {"full": 0, "segment": 0, "text": 0}
    text_rows: dict[str, str] = {}
    for row in rows:
        text_rows.setdefault(str(row["prompt_id"]), str(row["prompt_text"]))
    text_dir = output_root / "text"
    for prompt_id, prompt_text in sorted(text_rows.items()):
        destination = text_dir / f"{slug(prompt_id)}.npy"
        text_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        if _save_if_needed(
            lambda prompt_text=prompt_text: encoder.encode_text([prompt_text])[0],
            destination,
            split="all",
            clip_id=prompt_id,
            kind="text",
            plan_hash=text_hash,
            entries=entries,
            manifest_path=cache_manifest,
            checkpoint_id=encoder_config["checkpoint_name"],
        ):
            processed["text"] += 1
    for index, row in enumerate(rows, start=1):
        values, source_rate = load_mono_audio(row["wav_path"], expected_sample_rate_hz=int(config["dataset"]["sample_rate_hz"]))
        clip_id = str(row["clip_id"])
        full_plan = segment_plan_hash([])
        full_destination = output_root / "audio_full" / f"{slug(clip_id)}.npy"
        if _save_if_needed(
            lambda values=values: encoder.encode_audio([values], sample_rate_hz=source_rate)[0],
            full_destination,
            split=str(row["split"]),
            clip_id=clip_id,
            kind="audio_full",
            plan_hash=full_plan,
            entries=entries,
            manifest_path=cache_manifest,
            checkpoint_id=encoder_config["checkpoint_name"],
        ):
            processed["full"] += 1
        segments, windows = segment_waveforms(
            values,
            source_rate_hz=source_rate,
            target_rate_hz=encoder.sample_rate_hz,
            window_s=float(config["segment"]["primary_window_seconds"]),
            overlap=float(config["segment"]["primary_overlap"]),
            clip_id=clip_id,
        )
        segment_destination = output_root / "audio_seg" / f"{slug(clip_id)}.npy"
        if _save_if_needed(
            lambda windows=windows: encoder.encode_audio(windows, sample_rate_hz=encoder.sample_rate_hz),
            segment_destination,
            split=str(row["split"]),
            clip_id=clip_id,
            kind="audio_seg",
            plan_hash=segment_plan_hash(segments),
            entries=entries,
            manifest_path=cache_manifest,
            checkpoint_id=encoder_config["checkpoint_name"],
        ):
            processed["segment"] += 1
        if index == 1 or index == len(rows) or index % 10 == 0:
            print(json.dumps({"processed": index, "total": len(rows), **processed}, sort_keys=True))
    peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    write_run_metadata(
        output_root / "run.json",
        {
            "encoder": "clap",
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "rows_requested": len(rows),
            "processed_new": processed,
            "peak_cuda_bytes": peak_memory,
        },
    )
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
