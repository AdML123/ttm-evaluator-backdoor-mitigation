"""Convenience loader for the validated MusicEval manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .manifest import build_manifest, write_jsonl


def resolve_music_eval_root(root: str | Path | None = None) -> Path:
    value = root if root is not None else os.environ.get("MUSICEVAL_ROOT")
    if not value:
        raise ValueError("provide --root or set MUSICEVAL_ROOT")
    return Path(value).expanduser().resolve()


def load_music_eval(
    root: str | Path | None = None,
    *,
    expected_counts: dict[str, int] | None = None,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    check_audio: bool = True,
) -> list[dict[str, Any]]:
    return build_manifest(
        resolve_music_eval_root(root),
        expected_counts=expected_counts,
        expected_sample_rate_hz=sample_rate_hz,
        expected_channels=channels,
        check_audio=check_audio,
    )


__all__ = ["load_music_eval", "resolve_music_eval_root", "write_jsonl"]
