"""Validate the MusicEval checkout under ``data/raw`` and optionally write its manifest.

The dataset root is NOT operator-supplied: this script discovers it under the
project's allow-listed ``data/raw`` directory by literal filename patterns
(``total_mos_list.txt`` is a unique release marker).  Place the extracted
MusicEval release under ``data/raw/`` (any nesting depth) and run:

    python scripts/prepare_data.py --config configs/project.yaml [--check-only]

Every path used downstream comes from this literal-pattern discovery and is
confined to the discovered root, so no external string ever reaches a path
operation (path-traversal contract).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Allow the documented ``python scripts/prepare_data.py`` invocation to import
# the repository package without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import DatasetContractError, build_manifest, write_jsonl

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_MARKER = "total_mos_list.txt"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def _discover_dataset_root(config: dict) -> Path:
    """Find the MusicEval root under the config-declared allow-listed parents."""
    allowed_parents = [
        (_PROJECT_ROOT / str(p)).resolve()
        for p in config["dataset"].get("allowed_root_parents", ["data/raw"])
    ]
    for parent in allowed_parents:
        if not parent.is_dir():
            continue
        markers = sorted(p.parent for p in parent.rglob(_RELEASE_MARKER) if p.is_file())
        if not markers:
            continue
        if len(markers) > 1:
            raise SystemExit(
                f"ambiguous MusicEval releases under {parent}: {[str(m) for m in markers]}"
            )
        root = markers[0].resolve()
        root.relative_to(parent)  # containment assertion
        return root
    raise SystemExit(
        "MusicEval release not found under "
        f"{[str(p) for p in allowed_parents]}; extract it there first (see README)"
    )


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    root = _discover_dataset_root(config)
    print(f"dataset root: {root}")
    try:
        rows = build_manifest(
            root,
            expected_counts=dataset["split_counts"],
            expected_sample_rate_hz=int(dataset["sample_rate_hz"]),
            expected_channels=int(dataset["channels"]),
            check_audio=True,
        )
    except DatasetContractError as exc:
        raise SystemExit(f"DATASET CONTRACT ERROR: {exc}") from exc
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "test")}
    prompt_count = len({row["prompt_id"] for row in rows})
    report = {"clips": len(rows), "split_counts": counts, "prompts_used": prompt_count}
    print(json.dumps(report, sort_keys=True))
    if not args.check_only:
        destination = args.manifest or (_PROJECT_ROOT / str(config["outputs"]["manifest_file"]))
        write_jsonl(rows, destination)
        print(f"manifest: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
