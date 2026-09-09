from pathlib import Path

import pytest

from src.metrics.agreement import (
    compute_inter_rater_agreement,
    krippendorff_alpha_interval,
    parse_person_mos,
)


def test_interval_alpha_is_one_for_perfect_agreement():
    assert krippendorff_alpha_interval([[1, 1, 1], [2, 2, 2]]) == pytest.approx(1.0)


def test_interval_alpha_handles_unbalanced_units():
    value = krippendorff_alpha_interval([[1, 2], [2, 3, 4], [4, 4, 5, 5]])

    assert -1.0 <= value <= 1.0


def test_person_mos_parser_and_agreement_report(tmp_path: Path):
    path = tmp_path / "dev_person_mos.txt"
    path.write_text(
        "a.wav,Rater02,1,2\n"
        "a.wav,Rater01,1,2\n"
        "b.wav,Rater01,4,3\n"
        "b.wav,Rater02,5,3\n",
        encoding="utf-8",
    )

    rows = parse_person_mos(path)
    assert len(rows) == 4
    report = compute_inter_rater_agreement(path)

    assert report["method"] == "Krippendorff alpha (interval)"
    assert report["subset"] == "all clips in dev_person_mos.txt"
    assert report["dimensions"]["MI"]["n_clips"] == 2
    assert report["dimensions"]["TA"]["value"] == pytest.approx(1.0)


def test_directory_source_resolves_split_file(tmp_path: Path):
    directory = tmp_path / "person_mos"
    directory.mkdir()
    (directory / "test_person_mos.txt").write_text(
        "a.wav,Rater01,1,1\na.wav,Rater02,2,2\n", encoding="utf-8"
    )

    report = compute_inter_rater_agreement(directory, split="test")

    assert report["source"].endswith("test_person_mos.txt")
    assert report["split"] == "test"
