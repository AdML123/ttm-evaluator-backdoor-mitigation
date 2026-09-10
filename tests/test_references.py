"""Regression checks for the paper50 submission bibliography."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REQUIRED = {
    # boundary works (novelty delineation)
    "drmguard": "2411.04811",
    "likesidis": "2109.02381",
    "sml": "semi-supervised",
    # neuron-pruning lineage
    "anp": "NeurIPS",
    "rnp": "Reconstructive",
    # attack and benchmark anchors
    "eventtrojan": "EventTrojan",
    "badiqa": "BadIQA",
    "musiceval": "MusicEval",
    "singmos": "SingMOS-Pro",
    "nisqa": "NISQA",
    "wav2vec2": "wav2vec",
}


def _entries(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(
        r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", re.S
    )
    for match in pattern.finditer(text):
        result[match.group(1)] = match.group(2)
    return result


def _submission_tex() -> Path:
    path = Path(__file__).parents[1] / "submission" / "main.tex"
    if not path.is_file():
        pytest.skip("submission sources not present in this checkout")
    return path


def test_required_bibliography_entries_are_present_and_grounded():
    entries = _entries(_submission_tex().read_text(encoding="utf-8"))

    assert set(_REQUIRED) <= set(entries), f"missing keys: {set(_REQUIRED) - set(entries)}"
    for key, needle in _REQUIRED.items():
        assert needle.lower() in entries[key].lower(), f"{key} lacks '{needle}'"


def test_bibliography_has_no_local_machine_paths_or_credentials():
    text = _submission_tex().read_text(encoding="utf-8").lower()
    for forbidden in ("d:\\paper49", "d:\\paper50", "key.txt", "accesstoken", "github_pat_", "ghp_"):
        assert forbidden not in text, forbidden


def test_no_unresolved_verify_markers():
    text = _submission_tex().read_text(encoding="utf-8")
    assert "\\verify" not in text
    assert "\\todo" not in text
