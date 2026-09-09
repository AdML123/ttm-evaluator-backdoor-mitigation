"""Clip-level evaluation metrics used by the reproducible analysis pipeline.

The evaluator deliberately operates on clip-level vectors.  Segment scores are
summarised before a metric is calculated, and the companion bootstrap module
resamples clip rows rather than individual segments.  This prevents a clip
with many windows from receiving an unintended larger weight.

The signed-error convention in this module is ``prediction - target``.  A
Bland--Altman limit therefore has the same sign convention.  Correlations are
reported as ``NaN`` when they are not statistically meaningful (fewer than two
observations or a constant vector), instead of manufacturing a zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_DIMENSION_NAMES = ("MI", "TA")


def _coerce_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a non-empty one-dimensional floating-point vector."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _vector(values: Any, *, name: str, drop_nonfinite: bool = False) -> np.ndarray:
    """Return a one-dimensional floating-point vector with finite checks."""

    array = _coerce_vector(values, name=name)
    finite = np.isfinite(array)
    if not finite.all():
        if not drop_nonfinite:
            raise ValueError(f"{name} contains non-finite values")
        array = array[finite]
        if array.size == 0:
            raise ValueError(f"{name} has no finite paired values")
    return array


def _paired_vectors(
    target: Any,
    prediction: Any,
    *,
    drop_nonfinite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    # Keep the vectors unfiltered initially so that a finite mask can be
    # applied pairwise.  Filtering each side independently would misalign
    # observations when only one member of a pair is non-finite.
    y_true = _coerce_vector(target, name="target")
    y_pred = _coerce_vector(prediction, name="prediction")
    if y_true.size != y_pred.size:
        raise ValueError("target and prediction must have the same number of values")
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if not finite.all():
        if not drop_nonfinite:
            raise ValueError("target and prediction contain non-finite paired values")
        y_true = y_true[finite]
        y_pred = y_pred[finite]
        if y_true.size == 0:
            raise ValueError("target and prediction have no finite paired values")
    return y_true, y_pred


def bland_altman_limits(
    target: Any,
    prediction: Any,
    *,
    z: float = 1.96,
    drop_nonfinite: bool = False,
) -> tuple[float, float, float]:
    """Return ``(bias, lower, upper)`` for a Bland--Altman comparison.

    Differences are ``prediction - target`` and the standard deviation is the
    sample standard deviation (``ddof=1``).  For one paired observation the
    standard deviation is defined as zero, making both limits equal to the
    observed bias; this is useful for deterministic tiny smoke tests while
    still making the low sample size visible through the returned ``n`` in the
    full metric dictionary.
    """

    if not np.isfinite(z) or z < 0:
        raise ValueError("z must be a finite non-negative value")
    y_true, y_pred = _paired_vectors(
        target, prediction, drop_nonfinite=drop_nonfinite
    )
    differences = y_pred - y_true
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    return bias, float(bias - z * sd), float(bias + z * sd)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Compute Pearson and Spearman correlations without warning spam."""

    if left.size < 2:
        return float("nan"), float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    left_norm = float(np.sqrt(np.dot(left_centered, left_centered)))
    right_norm = float(np.sqrt(np.dot(right_centered, right_centered)))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan"), float("nan")
    pearson = float(np.dot(left_centered, right_centered) / (left_norm * right_norm))

    # Average ranks handle ties in the same way as scipy.stats.spearmanr.  A
    # local implementation keeps the core evaluator usable in the minimal
    # public package even when SciPy is not installed.
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks_sorted = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
            start = end
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = ranks_sorted
        return ranks

    left_rank = rank(left)
    right_rank = rank(right)
    left_rank -= np.mean(left_rank)
    right_rank -= np.mean(right_rank)
    spearman = float(
        np.dot(left_rank, right_rank)
        / np.sqrt(np.dot(left_rank, left_rank) * np.dot(right_rank, right_rank))
    )
    return pearson, spearman


def compute_regression_metrics(
    target: Any,
    prediction: Any,
    *,
    drop_nonfinite: bool = False,
    bland_altman_z: float = 1.96,
) -> dict[str, float | int]:
    """Calculate clip-level error, correlation, and Bland--Altman statistics.

    The result is a JSON-friendly dictionary.  ``n`` is the number of paired
    finite clips used.  ``signed_error`` is the mean ``prediction - target``;
    ``bias`` and ``mean_error`` are retained as explicit aliases for downstream
    table generators.  Correlations are ``NaN`` for degenerate vectors.
    """

    y_true, y_pred = _paired_vectors(
        target, prediction, drop_nonfinite=drop_nonfinite
    )
    error = y_pred - y_true
    mse = float(np.mean(np.square(error)))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    signed_error = float(np.mean(error))
    pearson, spearman = _rank_correlation(y_true, y_pred)
    bias, lower, upper = bland_altman_limits(
        y_true, y_pred, z=bland_altman_z, drop_nonfinite=False
    )
    sd = float(np.std(error, ddof=1)) if error.size > 1 else 0.0
    return {
        "n": int(y_true.size),
        "target_mean": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(y_pred)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "signed_error": signed_error,
        "mean_error": signed_error,
        "bias": signed_error,
        "pearson_r": pearson,
        "pearson": pearson,
        "spearman_rho": spearman,
        "spearman": spearman,
        "srcc": spearman,
        "bland_altman_bias": bias,
        "bland_altman_sd": sd,
        "loa_lower": lower,
        "loa_upper": upper,
        "bland_altman_lower": lower,
        "bland_altman_upper": upper,
        "ba_lower": lower,
        "ba_upper": upper,
    }


# Short aliases make the module convenient from notebooks and preserve a
# descriptive name for scripts that use ``regression_metrics``.
regression_metrics = compute_regression_metrics
clip_metrics = compute_regression_metrics


def _normalise_dimension(name: Any) -> str:
    token = str(name).strip().upper()
    if token in {"MI", "MUSICALITY", "MUSICAL_INTENT"}:
        return "MI"
    if token in {"TA", "TEXTURE", "TEXTURAL_APPEAL"}:
        return "TA"
    return token


def _dimension_arrays(
    values: Any,
    *,
    dimensions: Sequence[str],
    name: str,
) -> dict[str, np.ndarray]:
    """Coerce 1-D/2-D arrays or dimension mappings into named vectors."""

    names = tuple(_normalise_dimension(d) for d in dimensions)
    if isinstance(values, Mapping):
        lookup = {_normalise_dimension(k): v for k, v in values.items()}
        missing = [dimension for dimension in names if dimension not in lookup]
        if missing:
            raise ValueError(f"{name} is missing dimensions: {', '.join(missing)}")
        return {
            dimension: _vector(lookup[dimension], name=f"{name}[{dimension}]")
            for dimension in names
        }
    array = np.asarray(values)
    if array.ndim == 1:
        if len(names) != 1:
            raise ValueError(
                f"{name} must have one column per dimension ({len(names)} expected)"
            )
        return {names[0]: _vector(array, name=f"{name}[{names[0]}]")}
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(
            f"{name} must be a 1-D vector or a 2-D array with {len(names)} columns"
        )
    return {
        dimension: _vector(array[:, index], name=f"{name}[{dimension}]")
        for index, dimension in enumerate(names)
    }


def _row_dimension_value(row: Mapping[str, Any], dimension: str, *, prediction: bool) -> Any:
    lower = dimension.lower()
    if prediction:
        candidates = (
            f"pred_{lower}",
            f"prediction_{lower}",
            f"{lower}_pred",
            f"pred{dimension}",
            f"prediction{dimension}",
        )
        nested_names = ("prediction", "predictions", "y_pred")
    else:
        candidates = (
            lower,
            f"target_{lower}",
            f"{lower}_target",
            f"target{dimension}",
            f"y_{lower}",
        )
        nested_names = ("target", "targets", "y_true")
    for candidate in candidates:
        if candidate in row:
            return row[candidate]
    for nested_name in nested_names:
        nested = row.get(nested_name)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if _normalise_dimension(key) == dimension:
                    return value
    raise ValueError(
        f"row for clip {row.get('clip_id', '<unknown>')} lacks {dimension} "
        f"{'prediction' if prediction else 'target'}"
    )


def evaluate_clip_predictions(
    targets_or_rows: Any,
    predictions: Any | None = None,
    *,
    dimensions: Sequence[str] = _DIMENSION_NAMES,
    clip_ids: Sequence[Any] | None = None,
    drop_nonfinite: bool = False,
) -> dict[str, dict[str, float | int]]:
    """Evaluate MI/TA predictions from arrays, mappings, or row dictionaries.

    ``targets_or_rows`` may be a ``(n, 2)`` target array, a mapping from
    dimension names to vectors, or a sequence of rows containing ``clip_id``,
    target (``mi``/``ta``), and prediction (``pred_mi``/``pred_ta``) fields.
    Passing ``predictions`` selects the array/mapping form.  The function
    validates clip alignment and returns one metric dictionary per dimension.
    """

    names = tuple(_normalise_dimension(d) for d in dimensions)
    if not names:
        raise ValueError("at least one dimension is required")

    if predictions is None:
        if isinstance(targets_or_rows, Mapping):
            # A single row mapping is ambiguous; require an explicit
            # prediction argument unless the mapping contains row-like keys.
            rows: list[Mapping[str, Any]] = [targets_or_rows]
        else:
            try:
                rows = list(targets_or_rows)
            except TypeError as exc:
                raise ValueError("provide predictions for array inputs") from exc
        if not rows or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("row input must be a non-empty sequence of mappings")
        found_ids = [str(row.get("clip_id", index)) for index, row in enumerate(rows)]
        if len(set(found_ids)) != len(found_ids):
            raise ValueError("clip IDs must be unique")
        if clip_ids is not None and [str(item) for item in clip_ids] != found_ids:
            raise ValueError("clip_ids are not aligned with row input")
        true_map = {
            dimension: np.asarray(
                [_row_dimension_value(row, dimension, prediction=False) for row in rows]
            )
            for dimension in names
        }
        pred_map = {
            dimension: np.asarray(
                [_row_dimension_value(row, dimension, prediction=True) for row in rows]
            )
            for dimension in names
        }
    else:
        true_map = _dimension_arrays(
            targets_or_rows, dimensions=names, name="target"
        )
        pred_map = _dimension_arrays(predictions, dimensions=names, name="prediction")
        found_ids = [str(item) for item in clip_ids] if clip_ids is not None else []
        if clip_ids is not None:
            if len(found_ids) != len(next(iter(true_map.values()))):
                raise ValueError("clip_ids must have one value per clip")
            if len(set(found_ids)) != len(found_ids):
                raise ValueError("clip IDs must be unique")

    result: dict[str, dict[str, float | int]] = {}
    for dimension in names:
        true_values = true_map[dimension]
        pred_values = pred_map[dimension]
        if true_values.size != pred_values.size:
            raise ValueError(
                f"target and prediction must have the same number of {dimension} values"
            )
        result[dimension] = compute_regression_metrics(
            true_values,
            pred_values,
            drop_nonfinite=drop_nonfinite,
        )
    return result


def _segment_rows(segment_scores: Any) -> list[np.ndarray]:
    if isinstance(segment_scores, Mapping):
        values = list(segment_scores.values())
    elif isinstance(segment_scores, np.ndarray):
        if segment_scores.ndim == 1:
            values = [segment_scores]
        elif segment_scores.ndim == 2:
            values = [segment_scores[index] for index in range(segment_scores.shape[0])]
        else:
            raise ValueError("segment_scores array must be one- or two-dimensional")
    else:
        try:
            values = list(segment_scores)
        except TypeError as exc:
            raise ValueError("segment_scores must be a 2-D array or sequence") from exc
        # A plain one-dimensional Python sequence denotes one clip's segment
        # vector.  A nested sequence denotes one vector per clip.
        if values and all(np.asarray(value).ndim == 0 for value in values):
            values = [values]
    rows: list[np.ndarray] = []
    for index, value in enumerate(values):
        row = _vector(value, name=f"segment_scores[{index}]")
        rows.append(row)
    if not rows:
        raise ValueError("segment_scores must not be empty")
    return rows


def segment_dispersion(segment_scores: Any, *, ddof: int = 0) -> np.ndarray:
    """Return per-clip segment-score standard deviations.

    Ragged segment vectors are supported.  A clip with one segment has zero
    dispersion by definition; this is preferable to propagating NumPy's
    ``NaN`` into subgroup masks.
    """

    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    rows = _segment_rows(segment_scores)
    output = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        if row.size <= ddof:
            output[index] = 0.0 if row.size == 1 else np.nan
        else:
            output[index] = float(np.std(row, ddof=ddof))
    if not np.isfinite(output).all():
        raise ValueError("segment_scores must contain enough finite values per clip")
    return output


def make_variance_subgroups(
    segment_scores: Any,
    threshold: float | None = None,
    *,
    ddof: int = 0,
    high_inclusive: bool = False,
) -> dict[str, Any]:
    """Split clips by segment-score dispersion using a frozen threshold.

    If ``threshold`` is omitted, the median of the supplied dispersions is
    used.  The caller should compute that median on development clips and pass
    the resulting scalar unchanged for test clips.  Under the approved
    protocol, low variance is ``dispersion <= threshold`` and high variance is
    strictly ``dispersion > threshold``; set ``high_inclusive=True`` only when
    a different tie convention is explicitly intended.  ``high_mask`` and
    ``low_mask`` are complementary and preserve input order.
    """

    dispersion = segment_dispersion(segment_scores, ddof=ddof)
    chosen = float(np.median(dispersion) if threshold is None else threshold)
    if not np.isfinite(chosen):
        raise ValueError("threshold must be finite")
    if high_inclusive:
        high = dispersion >= chosen
    else:
        # The experiment protocol defines low variance as <= median and high
        # variance as strictly > median.  Keep the switch for callers that
        # intentionally need the opposite tie convention.
        high = dispersion > chosen
    low = ~high
    return {
        "dispersion": dispersion,
        "threshold": chosen,
        "high_mask": high,
        "low_mask": low,
        "n_high": int(np.sum(high)),
        "n_low": int(np.sum(low)),
    }


def split_by_variance(
    segment_scores: Any,
    threshold: float,
    *,
    ddof: int = 0,
    high_inclusive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper returning ``(high_mask, low_mask)``."""

    groups = make_variance_subgroups(
        segment_scores,
        threshold,
        ddof=ddof,
        high_inclusive=high_inclusive,
    )
    return groups["high_mask"], groups["low_mask"]


__all__ = [
    "bland_altman_limits",
    "clip_metrics",
    "compute_regression_metrics",
    "evaluate_clip_predictions",
    "make_variance_subgroups",
    "regression_metrics",
    "segment_dispersion",
    "split_by_variance",
]
