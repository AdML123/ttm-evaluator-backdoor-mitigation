"""Atomic NumPy feature writes and hash-verified cache manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save_npy(array: np.ndarray, destination: str | Path) -> Path:
    """Write a NumPy array and atomically publish the completed file."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return destination_path


def cache_entry(
    path: str | Path,
    *,
    split: str,
    clip_id: str,
    encoder: str,
    checkpoint: str,
    preprocessing_version: str,
    segment_plan_hash: str,
    kind: str = "feature",
) -> dict[str, Any]:
    """Describe a completed cache file using JSON-compatible scalar fields."""

    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(path_obj)
    values = np.load(path_obj, allow_pickle=False, mmap_mode="r")
    try:
        shape = [int(item) for item in values.shape]
        dtype = str(values.dtype)
    finally:
        del values
    return {
        "path": str(path_obj.resolve()),
        "split": split,
        "clip_id": clip_id,
        "encoder": encoder,
        "checkpoint": checkpoint,
        "preprocessing_version": preprocessing_version,
        "segment_plan_hash": segment_plan_hash,
        "kind": kind,
        "dtype": dtype,
        "shape": shape,
        "sha256": sha256_file(path_obj),
    }


def entry_is_valid(entry: Mapping[str, Any]) -> bool:
    """Return true only when file bytes, dtype, and shape match the entry."""

    try:
        path = Path(str(entry["path"]))
        if not path.is_file() or sha256_file(path) != str(entry["sha256"]):
            return False
        values = np.load(path, allow_pickle=False, mmap_mode="r")
        try:
            return str(values.dtype) == str(entry["dtype"]) and [
                int(item) for item in values.shape
            ] == [int(item) for item in entry["shape"]]
        finally:
            del values
    except (KeyError, OSError, TypeError, ValueError):
        return False


def write_manifest_entry(entry: Mapping[str, Any], manifest_path: str | Path) -> Path:
    """Append one canonical JSONL entry after the feature is fully written."""

    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(entry), ensure_ascii=True, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def segment_plan_hash(segments: Any) -> str:
    """Hash a segment sequence without depending on Python object reprs."""

    payload = [segment.as_dict() if hasattr(segment, "as_dict") else segment for segment in segments]
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
