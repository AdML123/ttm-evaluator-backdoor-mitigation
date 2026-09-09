"""Validation and manifest construction for an external MusicEval checkout.

The raw dataset is intentionally never copied into this repository.  A manifest
contains only the fields needed by later stages and points back to the caller's
local ``MUSICEVAL_ROOT``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping


SPLITS = ("train", "dev", "test")
_PROMPT_ID_RE = re.compile(r"(?i)(?<![A-Z0-9])P\d{3,}(?![A-Z0-9])")


class DatasetContractError(ValueError):
    """Raised when an external dataset does not satisfy the input contract."""


def _prompt_file(root: Path) -> Path:
    # Releases seen in the supplied archives place this file at the root; older
    # sample layouts put it below metadata.  Literal-pattern discovery finds
    # either location; when both exist the shallowest (root-level) wins.
    matches = sorted(
        (p for p in root.rglob("prompt_info.txt") if p.is_file()),
        key=lambda p: len(p.relative_to(root).parts),
    )
    if not matches:
        raise DatasetContractError("missing required path: prompt_info.txt")
    found = matches[0]
    try:
        found.relative_to(root)
    except ValueError as exc:
        raise DatasetContractError("path escapes dataset root: prompt_info.txt") from exc
    return found


def _normalise_clip_id(value: str) -> str:
    value = value.strip().replace("\\", "/")
    if not value:
        raise DatasetContractError("empty clip identifier in score list")
    return Path(value).name


def parse_score_list(path: Path, *, split: str) -> list[dict[str, object]]:
    """Parse ``filename,MI,TA`` rows from one official split list."""

    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    rows: list[dict[str, object]] = []
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DatasetContractError(f"cannot read split list: {path}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields = next(csv.reader([raw]))
            if fields and fields[0].strip().lower() in {"filename", "file", "clip"}:
                continue
            if len(fields) < 3:
                raise DatasetContractError(
                    f"invalid score row at {path}:{line_number}; expected filename,MI,TA"
                )
            clip_id = _normalise_clip_id(fields[0])
            try:
                mi = float(fields[1].strip())
                ta = float(fields[2].strip())
            except ValueError as exc:
                raise DatasetContractError(
                    f"invalid score at {path}:{line_number}; expected numeric MI and TA"
                ) from exc
            if not math.isfinite(mi) or not math.isfinite(ta):
                raise DatasetContractError(f"non-finite score at {path}:{line_number}")
            rows.append({"clip_id": clip_id, "mi": mi, "ta": ta, "split": split})
    return rows


def parse_prompt_info(path: Path) -> dict[str, str]:
    """Parse common tab-, comma-, or colon-separated prompt-info layouts."""

    prompts: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = _PROMPT_ID_RE.search(line)
            if match is None:
                # A title/header or an unrelated metadata line is harmless.
                continue
            prompt_id = match.group(0).upper()
            remainder = line[match.end() :].lstrip("\t ,:;|-\")")
            if not remainder:
                raise DatasetContractError(
                    f"missing prompt text at {path}:{line_number}"
                )
            if prompt_id in prompts and prompts[prompt_id] != remainder:
                raise DatasetContractError(f"conflicting prompt text for {prompt_id}")
            prompts[prompt_id] = remainder
    if not prompts:
        raise DatasetContractError(f"no prompt records found in {path}")
    return prompts


def parse_demo_prompt_info(path: Path) -> dict[str, str]:
    """Parse clip-specific prompt text from ``demo_prompt_info.txt``."""

    prompts: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t", 1)
            if len(fields) != 2 or fields[0].strip().lower() in {"id", "filename", "file"}:
                continue
            clip_id = _normalise_clip_id(fields[0])
            text = fields[1].strip()
            if not text:
                raise DatasetContractError(
                    f"missing demo prompt text at {path}:{line_number}"
                )
            if clip_id in prompts and prompts[clip_id] != text:
                raise DatasetContractError(f"conflicting demo prompt text for {clip_id}")
            prompts[clip_id] = text
    return prompts


def extract_prompt_id(clip_id: str) -> str:
    match = _PROMPT_ID_RE.search(clip_id)
    if match is None:
        raise DatasetContractError(f"cannot infer prompt id from clip: {clip_id}")
    return match.group(0).upper()


def _split_lists(root: Path) -> dict[str, Path]:
    """Discover the official split score lists by literal pattern.

    ``train|dev|test_mos_list.txt`` are located with a literal glob under the
    sanitised root; the split is parsed from the discovered filename, so no
    operator-controlled string ever reaches a path operation.
    """

    found: dict[str, Path] = {}
    for path in sorted(p for p in root.rglob("*_mos_list.txt") if p.is_file()):
        split = path.name.removesuffix("_mos_list.txt")
        if split not in SPLITS:
            continue  # e.g. total_mos_list.txt
        if split in found:
            raise DatasetContractError(f"ambiguous split list for {split}")
        path.relative_to(root)  # containment assertion
        found[split] = path
    missing = [s for s in SPLITS if s not in found]
    if missing:
        raise DatasetContractError(f"missing split lists: {missing}")
    return found


def _optional_file(root: Path, filename: str) -> Path | None:
    matches = sorted(p for p in root.rglob(filename) if p.is_file())
    if not matches:
        return None
    matches[0].relative_to(root)  # containment assertion
    return matches[0]


def _audio_index(root: Path) -> dict[str, Path]:
    wav_dirs = sorted(
        (p for p in root.rglob("wav") if p.is_dir()),
        key=lambda p: len(p.relative_to(root).parts),
    )
    if not wav_dirs:
        raise DatasetContractError("missing required path: wav")
    wav_root = wav_dirs[0]
    index: dict[str, Path] = {}
    for path in sorted(wav_root.rglob("*.wav")):
        path.relative_to(root)  # containment assertion
        key = path.name
        if key in index:
            raise DatasetContractError(f"duplicate audio filename: {key}")
        index[key] = path
    return index


def _check_audio(path: Path, *, sample_rate_hz: int, channels: int) -> tuple[float, int, int]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
    except Exception as exc:  # soundfile exposes several backend-specific errors
        raise DatasetContractError(f"unreadable audio: {path.name}") from exc
    if info.samplerate != sample_rate_hz:
        raise DatasetContractError(
            f"audio sample-rate mismatch for {path.name}: {info.samplerate} != {sample_rate_hz}"
        )
    if info.channels != channels:
        raise DatasetContractError(
            f"audio channel mismatch for {path.name}: {info.channels} != {channels}"
        )
    return info.frames / float(info.samplerate), info.samplerate, info.channels


def build_manifest(
    root: str | Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
    expected_sample_rate_hz: int | None = None,
    expected_channels: int | None = None,
    check_audio: bool = False,
) -> list[dict[str, object]]:
    """Validate a MusicEval root and return deterministic manifest rows."""

    if isinstance(root, str) and (not root.strip() or "\x00" in root):
        raise DatasetContractError("dataset root is empty or contains invalid characters")
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise DatasetContractError(f"dataset root is not a directory: {root_path}")
    if ".." in root_path.parts:
        raise DatasetContractError(f"dataset root must be fully resolved: {root_path}")
    prompt_path = _prompt_file(root_path)
    prompts = parse_prompt_info(prompt_path)
    demo_prompt_path = _optional_file(root_path, "demo_prompt_info.txt")
    demo_prompts = (
        parse_demo_prompt_info(demo_prompt_path)
        if demo_prompt_path is not None
        else {}
    )
    audio = _audio_index(root_path)
    split_lists = _split_lists(root_path)
    rows: list[dict[str, object]] = []
    seen: dict[str, str] = {}
    expected = dict(expected_counts or {})
    for split in SPLITS:
        split_path = split_lists[split]
        split_rows = parse_score_list(split_path, split=split)
        if split in expected and len(split_rows) != expected[split]:
            raise DatasetContractError(
                f"split count mismatch for {split}: {len(split_rows)} != {expected[split]}"
            )
        for row in split_rows:
            clip_id = str(row["clip_id"])
            if clip_id in seen:
                raise DatasetContractError(
                    f"duplicate clip across split lists: {clip_id} ({seen[clip_id]}, {split})"
                )
            seen[clip_id] = split
            wav_path = audio.get(clip_id)
            if wav_path is None:
                raise DatasetContractError(f"missing audio for clip: {clip_id}")
            prompt_id = extract_prompt_id(clip_id)
            prompt_text = demo_prompts.get(clip_id)
            prompt_source = "demo_prompt_info.txt" if prompt_text is not None else "prompt_info.txt"
            if prompt_text is None:
                prompt_text = prompts.get(prompt_id)
            if prompt_text is None:
                raise DatasetContractError(
                    f"missing prompt metadata for {clip_id}: {prompt_id}"
                )
            output = dict(row)
            output["prompt_id"] = prompt_id
            output["prompt_text"] = prompt_text
            output["prompt_source"] = prompt_source
            output["wav_path"] = str(wav_path)
            if check_audio:
                if expected_sample_rate_hz is None or expected_channels is None:
                    raise ValueError(
                        "expected_sample_rate_hz and expected_channels are required when check_audio=True"
                    )
                duration, rate, channel_count = _check_audio(
                    wav_path,
                    sample_rate_hz=expected_sample_rate_hz,
                    channels=expected_channels,
                )
                output["duration_s"] = duration
                output["sample_rate_hz"] = rate
                output["channels"] = channel_count
            rows.append(output)
    if expected and sum(expected.get(split, 0) for split in SPLITS) != len(rows):
        raise DatasetContractError(
            f"total count mismatch: {len(rows)} != {sum(expected.get(split, 0) for split in SPLITS)}"
        )
    return rows


def write_jsonl(rows: Iterable[Mapping[str, object]], destination: str | Path) -> Path:
    """Write a deterministic UTF-8 JSONL manifest."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return destination_path
