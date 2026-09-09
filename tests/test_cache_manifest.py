from pathlib import Path

import numpy as np
import pytest

from src.data.manifest import DatasetContractError, build_manifest, parse_score_list
from src.features.cache import (
    atomic_save_npy,
    cache_entry,
    entry_is_valid,
    write_manifest_entry,
)


def _write_dataset(root: Path, *, duplicate: bool = False) -> Path:
    (root / "wav").mkdir(parents=True)
    (root / "sets").mkdir()
    (root / "prompt_info.txt").write_text(
        "P001\tA bright piano phrase\nP002\tA slow string phrase\n",
        encoding="utf-8",
    )
    rows = {
        "train": ["demo-S001_P001.wav,3.0,4.0"],
        "dev": ["demo-S002_P002.wav,4.0,3.5"],
        "test": ["demo-S003_P001.wav,2.0,2.5"],
    }
    for split, values in rows.items():
        if duplicate and split == "test":
            values = [rows["train"][0]]
        (root / "sets" / f"{split}_mos_list.txt").write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )
    for value in rows.values():
        for line in value:
            filename = line.split(",", 1)[0]
            (root / "wav" / filename).write_bytes(b"placeholder")
    return root


def _write_demo_prompt(root: Path, filename: str, text: str) -> None:
    (root / "demo_prompt_info.txt").write_text(
        f"id\ttext\n{filename}\t{text}\n", encoding="utf-8"
    )


def test_parse_score_list_preserves_ids_and_scores(tmp_path):
    path = tmp_path / "scores.txt"
    path.write_text("clip.wav, 3.25, 4.5\n", encoding="utf-8")

    rows = parse_score_list(path, split="train")

    assert rows == [
        {"clip_id": "clip.wav", "mi": 3.25, "ta": 4.5, "split": "train"}
    ]


def test_build_manifest_checks_official_split_counts_and_prompt_mapping(tmp_path):
    root = _write_dataset(tmp_path)
    manifest = build_manifest(root, expected_counts={"train": 1, "dev": 1, "test": 1})

    assert [row["split"] for row in manifest] == ["train", "dev", "test"]
    assert manifest[0]["prompt_id"] == "P001"
    assert manifest[0]["prompt_text"] == "A bright piano phrase"
    assert Path(manifest[0]["wav_path"]).name == "demo-S001_P001.wav"


def test_build_manifest_uses_clip_specific_demo_prompt_for_unlisted_prompt_id(tmp_path):
    root = _write_dataset(tmp_path)
    filename = "demo-S002_P222.wav"
    (root / "sets" / "dev_mos_list.txt").write_text(
        f"{filename},4.0,3.5\n", encoding="utf-8"
    )
    (root / "wav" / filename).write_bytes(b"placeholder")
    _write_demo_prompt(root, filename, "A clip-specific prompt")

    manifest = build_manifest(root, expected_counts={"train": 1, "dev": 1, "test": 1})

    dev_row = next(row for row in manifest if row["split"] == "dev")
    assert dev_row["prompt_text"] == "A clip-specific prompt"
    assert dev_row["prompt_source"] == "demo_prompt_info.txt"


def test_build_manifest_fails_for_missing_required_path(tmp_path):
    with pytest.raises(DatasetContractError, match="prompt_info.txt"):
        build_manifest(tmp_path, expected_counts={"train": 0, "dev": 0, "test": 0})


def test_build_manifest_rejects_cross_split_duplicate(tmp_path):
    root = _write_dataset(tmp_path, duplicate=True)

    with pytest.raises(DatasetContractError, match="duplicate.*split"):
        build_manifest(root, expected_counts={"train": 1, "dev": 1, "test": 1})


def test_cache_entry_contains_hash_and_supports_verified_resume(tmp_path):
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    destination = tmp_path / "clip.npy"
    manifest_path = tmp_path / "manifest.jsonl"

    atomic_save_npy(array, destination)
    entry = cache_entry(
        destination,
        split="train",
        clip_id="clip.wav",
        encoder="test",
        checkpoint="unit",
        preprocessing_version="v1",
        segment_plan_hash="plan",
    )
    write_manifest_entry(entry, manifest_path)

    assert entry_is_valid(entry)
    assert entry["shape"] == [2, 3]
    assert entry["dtype"] == "float32"
    assert manifest_path.read_text(encoding="utf-8").count("\n") == 1

    destination.write_bytes(b"corrupt")
    assert not entry_is_valid(entry)
