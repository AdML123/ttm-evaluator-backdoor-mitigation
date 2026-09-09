"""Regression checks for the source-grounded submission bibliography."""

from __future__ import annotations

import re
from pathlib import Path


_REQUIRED = {
    "Wu2023CLAP": ("10.1109/ICASSP49357.2023.10095969", "1--5"),
    "Li2024MERT": ("2306.00107", ""),
    "Liu2025MusicEval": ("10.1109/ICASSP49660.2025.10890307", "1--5"),
    "McFee2018AutoPool": ("10.1109/TASLP.2018.2858559", "2180--2193"),
    "Wang2019MILPooling": ("10.1109/ICASSP.2019.8682847", "31--35"),
    "Liu2021PowerPooling": ("10.1109/IJCNN52387.2021.9533332", "1--7"),
    "Zhu2025MuQ": ("10.1109/TASLPRO.2025.3602320", "3653--3664"),
    "Zhang2025Aesthetics": ("10.1109/MLSP62443.2025.11204254", "1--6"),
    "Chen2022HTSAT": ("10.1109/ICASSP43922.2022.9746312", "646--650"),
}


def _entries(text: str) -> dict[str, str]:
    chunks = re.split(r"\n(?=@\w+\{)", text)
    result: dict[str, str] = {}
    for chunk in chunks:
        match = re.match(r"@\w+\{([^,]+),", chunk.strip())
        if match:
            result[match.group(1)] = chunk
    return result


def test_required_bibliography_entries_are_source_grounded():
    path = Path(__file__).parents[1] / "submission" / "references.bib"
    entries = _entries(path.read_text(encoding="utf-8"))

    assert set(_REQUIRED) <= set(entries)
    for key, (identifier, pages) in _REQUIRED.items():
        entry = entries[key]
        assert identifier.lower() in entry.lower()
        if pages:
            assert f"pages" in entry
            assert pages in entry


def test_bibliography_has_no_local_machine_paths_or_credentials():
    path = Path(__file__).parents[1] / "submission" / "references.bib"
    text = path.read_text(encoding="utf-8").lower()
    for forbidden in ("d:\\paper49", "key.txt", "accesstoken", "github_pat_", "ghp_"):
        assert forbidden not in text
