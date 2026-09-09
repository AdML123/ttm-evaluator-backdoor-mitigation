"""Shared I/O helpers for staged embedding extraction scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .segmentation import Segment, make_segments, materialize_segment


def read_manifest(path: str | Path, *, split: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if split is not None and row.get("split") != split:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_mono_audio(path: str | Path, *, expected_sample_rate_hz: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf

    values, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"audio must be mono: {path}")
    if sample_rate != expected_sample_rate_hz:
        raise ValueError(
            f"unexpected sample rate for {path}: {sample_rate} != {expected_sample_rate_hz}"
        )
    return values, int(sample_rate)


def resample_audio(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(values, dtype=np.float32)
    import librosa

    return np.asarray(
        librosa.resample(np.asarray(values, dtype=np.float32), orig_sr=source_rate, target_sr=target_rate),
        dtype=np.float32,
    )


def segment_waveforms(
    values: np.ndarray,
    *,
    source_rate_hz: int,
    target_rate_hz: int,
    window_s: float,
    overlap: float,
    clip_id: str,
) -> tuple[tuple[Segment, ...], list[np.ndarray]]:
    resampled = resample_audio(values, source_rate_hz, target_rate_hz)
    segments = make_segments(
        resampled.shape[0] / target_rate_hz,
        window_s,
        overlap,
        sample_rate_hz=target_rate_hz,
        clip_id=clip_id,
    )
    windows = [materialize_segment(resampled, segment) for segment in segments]
    return segments, windows


def slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def load_existing_entries(path: str | Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    destination = Path(path)
    if not destination.is_file():
        return {}
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    with destination.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            entry = json.loads(raw)
            key = (str(entry.get("encoder", "")), str(entry.get("kind", "feature")), str(entry["clip_id"]))
            # The last entry is authoritative if a previous run was interrupted
            # after appending a duplicate row.
            entries[key] = entry
    return entries


def write_run_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
