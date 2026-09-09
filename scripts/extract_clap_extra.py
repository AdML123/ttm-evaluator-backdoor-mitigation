"""Extract CLAP segment caches for ablations and controlled degradation.

This script is intentionally a separate GPU phase.  It loads only CLAP, keeps
the raw MusicEval tree external, and writes atomic, manifest-checked arrays.
The primary 11 s/50% cache is never overwritten.  Degraded clips are written
to a separate ignored directory with the random burst position recorded in a
JSONL ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.cache import (
    atomic_save_npy,
    cache_entry,
    entry_is_valid,
    segment_plan_hash,
    write_manifest_entry,
)
from src.features.extraction import (
    load_existing_entries,
    load_mono_audio,
    read_manifest,
    segment_waveforms,
    slug,
)
from src.models.encoders import CLAPEncoder


ABLATIONS: tuple[tuple[str, float, float], ...] = (
    ("n2", 21.8, 0.5),
    ("n4", 11.0, 0.5),
    ("n8", 5.5, 0.5),
    ("ov0", 11.0, 0.0),
    ("ov50", 11.0, 0.5),
    ("ov75", 11.0, 0.75),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["ablation", "degradation", "all"], default="all")
    parser.add_argument("--ablation-root", type=Path, default=None)
    parser.add_argument("--degradation-root", type=Path, default=None)
    parser.add_argument("--degradation-count", type=int, default=50)
    return parser.parse_args()


def _checkpoint(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    env_name = str(config["encoders"]["clap"]["checkpoint_env"])
    value = os.environ.get(env_name)
    if not value:
        raise SystemExit(f"set {env_name} or pass --checkpoint")
    return Path(value).expanduser().resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _noise_waveform(values: np.ndarray, *, start: int, length: int, seed: int) -> tuple[np.ndarray, float]:
    source = np.asarray(values, dtype=np.float32).copy()
    if start < 0 or length <= 0 or start + length > source.size:
        raise ValueError("noise interval is outside the waveform")
    rms = float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
    noise_rms = 0.5 * rms
    generator = np.random.default_rng(int(seed))
    noise = generator.standard_normal(length).astype(np.float32)
    actual = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if actual > 0.0:
        noise *= np.float32(noise_rms / actual)
    source[start : start + length] += noise
    original_peak = float(np.max(np.abs(values)))
    altered_peak = float(np.max(np.abs(source)))
    if original_peak > 0.0 and altered_peak > 0.0:
        source *= np.float32(original_peak / altered_peak)
    return source, noise_rms


def _extract_ablations(
    rows: list[dict[str, Any]],
    *,
    encoder: CLAPEncoder,
    config: dict[str, Any],
    output_root: Path,
    checkpoint_name: str,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "features.jsonl"
    entries = load_existing_entries(manifest_path)
    processed = {name: 0 for name, _, _ in ABLATIONS}
    for index, row in enumerate(rows, start=1):
        values, source_rate = load_mono_audio(
            row["wav_path"],
            expected_sample_rate_hz=int(config["dataset"]["sample_rate_hz"]),
        )
        clip_id = str(row["clip_id"])
        plans: dict[str, tuple[tuple[Any, ...], list[np.ndarray]]] = {}
        # All 11-second variants can be encoded in one batch; this materially
        # reduces launch overhead while preserving each plan's boundaries.
        grouped: dict[float, list[tuple[str, tuple[Any, ...], list[np.ndarray]]]] = {}
        for name, window_s, overlap in ABLATIONS:
            segments, windows = segment_waveforms(
                values,
                source_rate_hz=source_rate,
                target_rate_hz=encoder.sample_rate_hz,
                window_s=window_s,
                overlap=overlap,
                clip_id=clip_id,
            )
            grouped.setdefault(float(window_s), []).append((name, segments, windows))
        for window_s, grouped_plans in grouped.items():
            all_windows = [window for _, _, windows in grouped_plans for window in windows]
            # Keep the resident model below the 8 GB target even for the
            # longest release clip (which yields dozens of windows).
            chunks = [
                encoder.encode_audio(all_windows[start : start + 4], sample_rate_hz=encoder.sample_rate_hz)
                for start in range(0, len(all_windows), 4)
            ]
            all_embeddings = np.concatenate(chunks, axis=0)
            cursor = 0
            for name, segments, windows in grouped_plans:
                count = len(windows)
                plans[name] = (segments, [np.asarray(item, dtype=np.float32) for item in all_embeddings[cursor : cursor + count]])
                cursor += count
        for name, _, _ in ABLATIONS:
            segments, embeddings = plans[name]
            destination = output_root / name / f"{slug(clip_id)}.npy"
            key = ("clap", f"audio_seg_extra_{name}", clip_id)
            existing = entries.get(key)
            if existing is not None and entry_is_valid(existing):
                continue
            array = np.stack(embeddings).astype(np.float32, copy=False)
            atomic_save_npy(array, destination)
            entry = cache_entry(
                destination,
                split=str(row["split"]),
                clip_id=clip_id,
                encoder="clap",
                checkpoint=checkpoint_name,
                preprocessing_version="clap48k-int16-roundtrip-center-crop-v1",
                segment_plan_hash=segment_plan_hash(segments),
                kind=f"audio_seg_extra_{name}",
            )
            write_manifest_entry(entry, manifest_path)
            entries[key] = entry
            processed[name] += 1
        if index == 1 or index == len(rows) or index % 10 == 0:
            print(json.dumps({"processed": index, "total": len(rows), **processed}, sort_keys=True), flush=True)
    metadata = {
        "encoder": "clap",
        "checkpoint": checkpoint_name,
        "rows_requested": len(rows),
        "processed_new": processed,
        "ablations": [{"name": n, "window_s": w, "overlap": o} for n, w, o in ABLATIONS],
    }
    _write_json(output_root / "run.json", metadata)
    return metadata


def _extract_degradation(
    rows: list[dict[str, Any]],
    *,
    encoder: CLAPEncoder,
    config: dict[str, Any],
    output_root: Path,
    checkpoint_name: str,
    seed: int,
    count: int,
) -> dict[str, Any]:
    selected = [row for row in sorted(rows, key=lambda item: str(item["clip_id"])) if float(row["mi"]) > 3.5]
    selected = selected[: int(count)]
    if len(selected) < int(count):
        raise RuntimeError(f"only {len(selected)} eligible test clips have MI > 3.5; {count} requested")
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "degraded.jsonl"
    full_dir = output_root / "audio_full"
    seg_dir = output_root / "audio_seg"
    records: list[dict[str, Any]] = []
    window_s = float(config["segment"]["primary_window_seconds"])
    overlap = float(config["segment"]["primary_overlap"])
    for index, row in enumerate(selected):
        values, source_rate = load_mono_audio(
            row["wav_path"],
            expected_sample_rate_hz=int(config["dataset"]["sample_rate_hz"]),
        )
        noise_length = int(round(3.0 * source_rate))
        if values.size <= noise_length:
            raise RuntimeError(f"clip too short for 3-second degradation: {row['clip_id']}")
        rng = np.random.default_rng(int(seed) + index)
        start = int(rng.integers(0, values.size - noise_length + 1))
        noisy, noise_rms = _noise_waveform(values, start=start, length=noise_length, seed=int(seed) + index + 100000)
        clip_id = str(row["clip_id"])
        segments, windows = segment_waveforms(
            noisy,
            source_rate_hz=source_rate,
            target_rate_hz=encoder.sample_rate_hz,
            window_s=window_s,
            overlap=overlap,
            clip_id=clip_id,
        )
        full_embedding = encoder.encode_audio([noisy], sample_rate_hz=source_rate)[0]
        segment_chunks = [
            encoder.encode_audio(windows[start : start + 4], sample_rate_hz=encoder.sample_rate_hz)
            for start in range(0, len(windows), 4)
        ]
        segment_embeddings = np.concatenate(segment_chunks, axis=0)
        full_path = full_dir / f"{slug(clip_id)}.npy"
        seg_path = seg_dir / f"{slug(clip_id)}.npy"
        atomic_save_npy(full_embedding.astype(np.float32), full_path)
        atomic_save_npy(segment_embeddings.astype(np.float32), seg_path)
        record = {
            "clip_id": clip_id,
            "split": str(row["split"]),
            "mi": float(row["mi"]),
            "ta": float(row["ta"]),
            "start_sample": start,
            "start_s": start / float(source_rate),
            "duration_s": 3.0,
            "noise_rms": noise_rms,
            "seed": int(seed) + index,
            "noise_seed": int(seed) + index + 100000,
            "full_path": str(full_path.resolve()),
            "segment_path": str(seg_path.resolve()),
            "segment_plan_hash": segment_plan_hash(segments),
            "n_segments": len(segments),
            "checkpoint": checkpoint_name,
        }
        records.append(record)
        if index == 0 or index + 1 == len(selected) or (index + 1) % 10 == 0:
            print(json.dumps({"degraded": index + 1, "total": len(selected)}, sort_keys=True), flush=True)
    with ledger_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    metadata = {
        "encoder": "clap",
        "checkpoint": checkpoint_name,
        "seed": int(seed),
        "count": len(records),
        "burst_seconds": 3.0,
        "noise_rms_ratio": 0.5,
        "ledger": str(ledger_path.resolve()),
    }
    _write_json(output_root / "run.json", metadata)
    return metadata


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(config["outputs"]["manifest_file"])
    rows = read_manifest(manifest_path, split=args.split, limit=args.limit)
    if not rows:
        raise SystemExit("no rows found in manifest")
    checkpoint = _checkpoint(args, config)
    encoder_cfg = config["encoders"]["clap"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    encoder = CLAPEncoder(
        checkpoint=checkpoint,
        audio_model=encoder_cfg.get("audio_model", "HTSAT-base"),
        max_input_samples=int(encoder_cfg.get("max_input_samples", 480000)),
        device=args.device,
    )
    checkpoint_name = str(encoder_cfg.get("checkpoint_name", checkpoint.name))
    output_base = Path(config["project"]["cache_root"])
    ablation_root = args.ablation_root or output_base / "clap_extra"
    degradation_root = args.degradation_root or output_base / "clap_degradation"
    output: dict[str, Any] = {"mode": args.mode, "split": args.split}
    if args.mode in {"ablation", "all"}:
        output["ablation"] = _extract_ablations(
            rows, encoder=encoder, config=config, output_root=ablation_root, checkpoint_name=checkpoint_name
        )
    if args.mode in {"degradation", "all"}:
        output["degradation"] = _extract_degradation(
            rows,
            encoder=encoder,
            config=config,
            output_root=degradation_root,
            checkpoint_name=checkpoint_name,
            seed=args.seed,
            count=args.degradation_count,
        )
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
