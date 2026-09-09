"""Inter-rater agreement for the supplied MusicEval ``person_mos`` files.

The archive stores five ratings per clip, while the complete release uses
different five-rater panels for different subsets of clips.  A global balanced
ICC over all fourteen rater columns would therefore discard most observations
or silently impute missing values.  We use interval Krippendorff's alpha,
which is defined for unbalanced/missing-rater designs and requires no such
imputation.  The implementation below computes the pairwise-complete interval
form: observed disagreement is the mean squared difference over rating pairs
within each clip and expected disagreement is the corresponding mean over all
rating pairs.  The supplied release has five ratings for every clip, so this
unit-weighted form is equivalent to the usual coincidence formulation up to a
common factor.

The result reports the exact source file, split/subset, clip and rating counts,
and the measurement level.  It must not be described as a statistic reported by
the MusicEval paper; it is computed locally from the archived person-level
ratings.
"""

from __future__ import annotations

import csv
import itertools
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _normalise_clip_id(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    if not value:
        raise ValueError("person_mos row has an empty clip identifier")
    return Path(value).name


def _fields(raw: str) -> list[str]:
    # The released files are comma-separated; accepting tabs makes the parser
    # useful for a few mirrored/sample releases as well.
    comma = next(csv.reader([raw]))
    if len(comma) == 1 and "\t" in raw:
        return [field.strip() for field in raw.rstrip("\r\n").split("\t")]
    return [field.strip() for field in comma]


def parse_person_mos(source: str | Path) -> list[dict[str, Any]]:
    """Parse ``filename,rater,MI,TA`` rows from a person-level MOS file."""

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"person_mos file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            fields = _fields(raw)
            if fields and fields[0].lower() in {"filename", "file", "clip", "clip_id"}:
                continue
            if len(fields) < 4:
                raise ValueError(
                    f"invalid person_mos row at {path}:{line_number}; expected clip,rater,MI,TA"
                )
            clip_id = _normalise_clip_id(fields[0])
            rater = fields[1].strip()
            if not rater:
                raise ValueError(f"empty rater at {path}:{line_number}")
            try:
                mi = float(fields[2])
                ta = float(fields[3])
            except ValueError as exc:
                raise ValueError(f"invalid MI/TA rating at {path}:{line_number}") from exc
            if not math.isfinite(mi) or not math.isfinite(ta):
                raise ValueError(f"non-finite rating at {path}:{line_number}")
            rows.append(
                {
                    "clip_id": clip_id,
                    "rater": rater,
                    "mi": mi,
                    "ta": ta,
                }
            )
    if not rows:
        raise ValueError(f"no person_mos rows found in {path}")
    return rows


def _unit_vectors(units: Any) -> list[np.ndarray]:
    if isinstance(units, Mapping):
        values = list(units.values())
    elif isinstance(units, np.ndarray):
        if units.ndim == 1:
            values = [units]
        elif units.ndim == 2:
            values = [units[index] for index in range(units.shape[0])]
        else:
            raise ValueError("units array must be one- or two-dimensional")
    else:
        try:
            values = list(units)
        except TypeError as exc:
            raise ValueError("units must be a sequence or mapping") from exc
        # A plain one-dimensional Python sequence denotes one unit's ratings;
        # nested sequences denote one unit per row.
        if values and all(np.asarray(value).ndim == 0 for value in values):
            values = [values]
    output: list[np.ndarray] = []
    for index, value in enumerate(values):
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unit {index} is not numeric") from exc
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"unit {index} must contain finite ratings")
        output.append(vector)
    if not output:
        raise ValueError("units must not be empty")
    return output


def krippendorff_alpha_interval(units: Any) -> float:
    """Compute pairwise-complete interval Krippendorff's alpha.

    ``units`` is a sequence/mapping of rating vectors.  Units with one rating
    contribute to expected disagreement but not observed within-unit pairs;
    at least one unit with two or more ratings is required.  A zero expected
    disagreement yields alpha one when observed disagreement is also zero and
    ``NaN`` otherwise.
    """

    vectors = _unit_vectors(units)
    observed_sum = 0.0
    observed_pairs = 0
    all_values: list[float] = []
    for vector in vectors:
        all_values.extend(float(value) for value in vector)
        if vector.size < 2:
            continue
        for left, right in itertools.combinations(vector, 2):
            observed_sum += float((left - right) ** 2)
            observed_pairs += 1
    if observed_pairs == 0:
        return float("nan")
    observed = observed_sum / observed_pairs
    if len(all_values) < 2:
        return float("nan")
    # Sum of squared pairwise differences can be evaluated in O(N) rather
    # than enumerating all O(N^2) pairs.  This matters for the 13,740 ratings
    # in the full MusicEval release.
    values_array = np.asarray(all_values, dtype=np.float64)
    total = float(values_array.size)
    expected_pairs = total * (total - 1.0) / 2.0
    expected_sum = total * float(np.dot(values_array, values_array)) - float(
        np.sum(values_array) ** 2
    )
    expected = expected_sum / expected_pairs
    if expected == 0.0:
        return 1.0 if observed == 0.0 else float("nan")
    alpha = 1.0 - observed / expected
    # Numerical noise can put a perfect score a few ulps above one.  Keep the
    # standard lower range unconstrained (strong disagreement can be < -1),
    # but make the upper bound exact for JSON/table consumers.
    return float(min(1.0, alpha))


def _resolve_source(source: str | Path, split: str | None) -> tuple[Path, str | None]:
    path = Path(source).expanduser()
    requested_split = split.lower() if split is not None else None
    if path.is_dir():
        base = path / "person_mos" if (path / "person_mos").is_dir() else path
        chosen_split = requested_split or "total"
        candidate = base / f"{chosen_split}_person_mos.txt"
        if not candidate.is_file():
            raise FileNotFoundError(f"person_mos split file not found: {candidate}")
        return candidate.resolve(), chosen_split
    if not path.is_file():
        raise FileNotFoundError(f"person_mos source not found: {path}")
    inferred = requested_split
    if inferred is None:
        match = re.search(r"(?:^|[_-])(train|dev|test|total)(?:[_-]|\.)", path.name.lower())
        inferred = match.group(1) if match else None
    return path.resolve(), inferred


def compute_inter_rater_agreement(
    source: str | Path,
    *,
    split: str | None = None,
    clip_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute locally derived interval alpha for MI and TA person ratings.

    ``source`` may be a ``person_mos`` directory, a MusicEval root containing
    that directory, or one explicit ``*_person_mos.txt`` file.  If ``clip_ids``
    is supplied, only those clips are retained; this makes the subset explicit
    in the returned metadata and avoids accidental split leakage.
    """

    path, inferred_split = _resolve_source(source, split)
    rows = parse_person_mos(path)
    requested = None if clip_ids is None else {_normalise_clip_id(item) for item in clip_ids}
    if requested is not None:
        if not requested:
            raise ValueError("clip_ids must not be empty")
        available = {str(row["clip_id"]) for row in rows}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                "requested clip_ids are missing from person_mos: "
                + ", ".join(missing[:5])
            )
        rows = [row for row in rows if row["clip_id"] in requested]

    by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["clip_id"]), str(row["rater"]))
        if key in seen_pairs:
            raise ValueError(f"duplicate clip/rater rating: {key[0]}, {key[1]}")
        seen_pairs.add(key)
        by_clip[str(row["clip_id"])].append(row)

    counts = np.asarray([len(values) for values in by_clip.values()], dtype=np.int64)
    dimensions: dict[str, dict[str, Any]] = {}
    for label, field in (("MI", "mi"), ("TA", "ta")):
        units = {
            clip_id: [float(row[field]) for row in values]
            for clip_id, values in by_clip.items()
        }
        value = krippendorff_alpha_interval(units)
        dimensions[label] = {
            "value": value,
            "alpha": value,
            "n_clips": int(len(units)),
            "n_ratings": int(sum(len(values) for values in units.values())),
            "ratings_per_clip": {
                "min": int(np.min(counts)),
                "max": int(np.max(counts)),
                "mean": float(np.mean(counts)),
            },
        }

    if requested is None:
        subset = f"all clips in {path.name}"
    else:
        subset = f"explicit subset of {len(requested)} clip IDs ({path.name})"
    return {
        "method": "Krippendorff alpha (interval)",
        "level_of_measurement": "interval",
        "source": str(path),
        "split": inferred_split,
        "subset": subset,
        "n_clips": int(len(by_clip)),
        "n_ratings": int(len(rows)),
        "n_raters": int(len({str(row["rater"]) for row in rows})),
        "ratings_per_clip": {
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "mean": float(np.mean(counts)),
        },
        "dimensions": dimensions,
    }


# A concise alias for scripts that use ``inter_rater_agreement``.
inter_rater_agreement = compute_inter_rater_agreement


__all__ = [
    "compute_inter_rater_agreement",
    "inter_rater_agreement",
    "krippendorff_alpha_interval",
    "parse_person_mos",
]
