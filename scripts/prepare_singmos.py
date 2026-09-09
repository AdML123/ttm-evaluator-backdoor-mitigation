"""Phase 2 / Task 13 Step 2：构建 SingMOS-Pro manifest。

读 metadata.json，构建 JSONL manifest（clip_id / wav 路径 / overall MOS /
split / system_id），作为回归任务的输入。写入经由 src.data.manifest.write_jsonl
（统一的受控 JSONL 落盘入口），输出位置为仓库内 cache/singmos/。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import write_jsonl  # noqa: E402

# 仓库根（本脚本位于 <root>/scripts/）；输出路径一律锚定到根内并校验。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_output(*parts: str) -> Path:
    path = _PROJECT_ROOT.joinpath(*parts).resolve()
    if not path.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"output path escapes project root: {path}")
    return path


def _meta_path() -> Path:
    raw = os.environ.get(
        "SINGMOS_META",
        "data/local/SingMOS-Pro/metadata.json",
    )
    path = Path(raw).expanduser().resolve()
    if "\x00" in raw or ".." in path.parts:
        raise ValueError(f"unsafe SINGMOS_META path: {raw!r}")
    if not path.is_file():
        raise FileNotFoundError(f"SINGMOS_META not found: {path}")
    return path


OUT = _project_output("cache", "singmos", "manifest.jsonl")

# metadata 的 wav 字段只接受纯相对 POSIX 路径（无 ..、无盘符、无绝对前缀），
# 防止外部 metadata 把 manifest 指向任意本机路径。
_SAFE_WAV_RE = re.compile(r"^[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)*$")


def _safe_wav(value: object, clip_id: object) -> str:
    wav = str(value)
    if not _SAFE_WAV_RE.match(wav):
        raise ValueError(f"unsafe wav path in metadata for clip {clip_id!r}: {wav!r}")
    return wav


def build_records() -> list[dict[str, object]]:
    with open(_meta_path(), encoding="utf-8") as f:
        rows = json.load(f)
    records: list[dict[str, object]] = []
    for r in rows:
        judge = r.get("judge_score") or []
        if not judge:
            continue
        overall = sum(judge) / len(judge)
        records.append(
            {
                "clip_id": r["id"],
                "wav": _safe_wav(r["wav"], r["id"]),
                "overall_mos": round(overall, 4),
                "split": r["split"],
                "system_id": r["system_id"],
            }
        )
    return records


def main() -> int:
    records = build_records()
    destination = write_jsonl(records, OUT)
    print(f"manifest written: {len(records)} rows -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
