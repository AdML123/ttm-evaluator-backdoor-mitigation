"""Deterministic clip-level bootstrap utilities.

The unit of resampling is always a clip.  A sampled index selects the target,
prediction, and *complete* segment-score vector belonging to that clip; segment
rows are never sampled independently.  This is important for MusicEval because
clips have a variable number of windows under the declared duration rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluate import compute_regression_metrics


MetricFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapResult:
    """A percentile confidence interval and its reproducibility metadata."""

    metric: str
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n_clips: int
    seed: int
    samples: np.ndarray
    resample_indices: np.ndarray
    clip_ids: tuple[str, ...]
    valid_resamples: int

    @property
    def ci(self) -> tuple[float, float]:
        return self.lower, self.upper

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Alias for callers that prefer a descriptive property name."""

        return self.lower, self.upper

    @property
    def ci_low(self) -> float:
        return self.lower

    @property
    def ci_high(self) -> float:
        return self.upper

    def as_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        """Return JSON-friendly metadata (optionally including arrays)."""

        output: dict[str, Any] = {
            "metric": self.metric,
            "estimate": float(self.estimate),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "ci_low": float(self.lower),
            "ci_high": float(self.upper),
            "confidence_level": float(self.confidence_level),
            "n_resamples": int(self.n_resamples),
            "n_clips": int(self.n_clips),
            "seed": int(self.seed),
            "valid_resamples": int(self.valid_resamples),
            "clip_ids": list(self.clip_ids),
        }
        if include_samples:
            output["samples"] = self.samples.tolist()
            output["resample_indices"] = self.resample_indices.tolist()
        return output

    # ``to_dict`` is a common spelling in result writers.
    to_dict = as_dict


def _normalise_clip_ids(clip_ids: Sequence[Any]) -> tuple[str, ...]:
    try:
        values = tuple(str(item) for item in clip_ids)
    except TypeError as exc:
        raise ValueError("clip_ids must be a non-empty sequence") from exc
    if not values:
        raise ValueError("clip_ids must be non-empty")
    if any(not item for item in values):
        raise ValueError("clip_ids must not contain empty values")
    if len(set(values)) != len(values):
        raise ValueError("clip_ids must be unique for clip-level bootstrap")
    return values


def _array(values: Any, *, name: str, n: int) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.size != n:
        raise ValueError(f"{name} must have one value per clip ({n} expected)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _segment_vectors(
    segment_scores: Any,
    clip_ids: tuple[str, ...],
) -> list[np.ndarray] | None:
    """Validate optional ragged segment vectors and align mapping inputs."""

    if segment_scores is None:
        return None
    if isinstance(segment_scores, Mapping):
        keys = {str(key) for key in segment_scores}
        expected = set(clip_ids)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            detail = []
            if missing:
                detail.append(f"missing={missing[:3]}")
            if extra:
                detail.append(f"extra={extra[:3]}")
            raise ValueError("segment_scores mapping is not clip-aligned (" + ", ".join(detail) + ")")
        values = [segment_scores[clip_id] for clip_id in clip_ids]
    elif isinstance(segment_scores, np.ndarray):
        if segment_scores.ndim == 1:
            # A one-dimensional vector is a single clip's segment vector.
            values = [segment_scores]
        elif segment_scores.ndim == 2:
            values = [segment_scores[index] for index in range(segment_scores.shape[0])]
        else:
            raise ValueError("segment_scores array must be one- or two-dimensional")
    else:
        try:
            values = list(segment_scores)
        except TypeError as exc:
            raise ValueError("segment_scores must be a mapping or sequence") from exc
        if values and all(np.asarray(value).ndim == 0 for value in values):
            values = [values]
        if len(values) != len(clip_ids):
            raise ValueError("segment_scores must have one vector per clip")
    vectors: list[np.ndarray] = []
    for index, value in enumerate(values):
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"segment_scores[{index}] must be numeric") from exc
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"segment_scores[{index}] must be non-empty and finite")
        vectors.append(vector)
    return vectors


def resample_clip_indices(
    n_clips: int,
    *,
    n_resamples: int = 2000,
    seed: int = 20260907,
) -> np.ndarray:
    """Generate deterministic bootstrap indices with shape ``(B, n_clips)``."""

    if isinstance(n_clips, bool) or int(n_clips) != n_clips or n_clips <= 0:
        raise ValueError("n_clips must be a positive integer")
    if isinstance(n_resamples, bool) or int(n_resamples) != n_resamples or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc
    generator = np.random.default_rng(seed_int)
    return generator.integers(
        0,
        int(n_clips),
        size=(int(n_resamples), int(n_clips)),
        dtype=np.int64,
    )


def resample_clip_records(
    records: Sequence[Mapping[str, Any]],
    indices: np.ndarray,
) -> list[list[dict[str, Any]]]:
    """Materialise sampled rows while keeping each clip's fields together.

    The returned rows are shallow copies, so nested segment vectors remain
    intact and can be inspected by a caller without mutating the source table.
    ``indices`` may be one-dimensional (one resample) or two-dimensional.
    """

    rows = list(records)
    if not rows:
        raise ValueError("records must be non-empty")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("records must contain mappings")
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim == 1:
        index_array = index_array.reshape(1, -1)
    if index_array.ndim != 2 or index_array.shape[1] != len(rows):
        raise ValueError("indices must have shape (n_resamples, n_clips)")
    if np.any(index_array < 0) or np.any(index_array >= len(rows)):
        raise ValueError("indices contain an out-of-range clip index")
    return [[dict(rows[int(index)]) for index in sample] for sample in index_array]


def _metric_function(metric: str | MetricFunction) -> tuple[str, MetricFunction]:
    if callable(metric):
        name = getattr(metric, "__name__", "custom")

        def custom(target: np.ndarray, prediction: np.ndarray) -> float:
            value = metric(target, prediction)
            return float(value)

        return str(name), custom
    token = str(metric).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mse": "mse",
        "mean_squared_error": "mse",
        "rmse": "rmse",
        "mae": "mae",
        "mean_absolute_error": "mae",
        "signed_error": "signed_error",
        "mean_error": "signed_error",
        "bias": "signed_error",
        "pearson": "pearson_r",
        "pearson_r": "pearson_r",
        "spearman": "spearman_rho",
        "spearman_rho": "spearman_rho",
    }
    canonical = aliases.get(token)
    if canonical is None:
        raise ValueError(f"unknown bootstrap metric: {metric}")

    def known(target: np.ndarray, prediction: np.ndarray) -> float:
        value = compute_regression_metrics(target, prediction)[canonical]
        return float(value)

    return canonical, known


def _validate_confidence(confidence_level: float) -> float:
    value = float(confidence_level)
    if not 0.0 < value < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    return value


def _percentile_interval(
    samples: np.ndarray,
    *,
    confidence_level: float,
) -> tuple[float, float, int]:
    finite = np.isfinite(samples)
    valid = int(np.sum(finite))
    if valid == 0:
        return float("nan"), float("nan"), 0
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(samples[finite], alpha)),
        float(np.quantile(samples[finite], 1.0 - alpha)),
        valid,
    )


def _bootstrap_from_indices(
    clip_ids: tuple[str, ...],
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    metric_name: str,
    metric_function: MetricFunction,
    indices: np.ndarray,
    confidence_level: float,
    seed: int,
) -> BootstrapResult:
    estimate = float(metric_function(target, prediction))
    samples = np.empty(indices.shape[0], dtype=np.float64)
    for row, sample_indices in enumerate(indices):
        samples[row] = metric_function(target[sample_indices], prediction[sample_indices])
    lower, upper, valid = _percentile_interval(
        samples, confidence_level=confidence_level
    )
    return BootstrapResult(
        metric=metric_name,
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        n_resamples=int(indices.shape[0]),
        n_clips=len(clip_ids),
        seed=int(seed),
        samples=samples,
        resample_indices=np.array(indices, copy=True),
        clip_ids=clip_ids,
        valid_resamples=valid,
    )


def bootstrap_clip_metric(
    clip_ids: Sequence[Any],
    target: Any,
    prediction: Any,
    *,
    metric: str | MetricFunction = "mse",
    n_resamples: int = 2000,
    seed: int = 20260907,
    confidence_level: float = 0.95,
    segment_scores: Any = None,
) -> BootstrapResult:
    """Bootstrap one metric by resampling complete clip rows.

    ``segment_scores`` is optional but, when supplied, is validated and aligned
    to ``clip_ids``.  It is intentionally not flattened or independently
    resampled; ``resample_indices`` provides an auditable record of the clip
    grouping for downstream segment-aware statistics.
    """

    ids = _normalise_clip_ids(clip_ids)
    y_true = _array(target, name="target", n=len(ids))
    y_pred = _array(prediction, name="prediction", n=len(ids))
    _segment_vectors(segment_scores, ids)
    confidence = _validate_confidence(confidence_level)
    metric_name, metric_function = _metric_function(metric)
    indices = resample_clip_indices(
        len(ids), n_resamples=n_resamples, seed=seed
    )
    return _bootstrap_from_indices(
        ids,
        y_true,
        y_pred,
        metric_name=metric_name,
        metric_function=metric_function,
        indices=indices,
        confidence_level=confidence,
        seed=int(seed),
    )


def _dimension_arrays(
    values: Any,
    *,
    dimensions: Sequence[str],
    name: str,
    n: int,
) -> dict[str, np.ndarray]:
    names = tuple(str(item).upper() for item in dimensions)
    if isinstance(values, Mapping):
        lookup = {str(key).upper(): value for key, value in values.items()}
        missing = [dimension for dimension in names if dimension not in lookup]
        if missing:
            raise ValueError(f"{name} is missing dimensions: {missing}")
        return {
            dimension: _array(lookup[dimension], name=f"{name}[{dimension}]", n=n)
            for dimension in names
        }
    array = np.asarray(values)
    if array.ndim == 1:
        if len(names) != 1:
            raise ValueError(f"{name} must have {len(names)} columns")
        return {names[0]: _array(array, name=f"{name}[{names[0]}]", n=n)}
    if array.ndim != 2 or array.shape != (n, len(names)):
        raise ValueError(f"{name} must have shape ({n}, {len(names)})")
    return {
        dimension: _array(array[:, column], name=f"{name}[{dimension}]", n=n)
        for column, dimension in enumerate(names)
    }


def bootstrap_clip_metrics(
    clip_ids: Sequence[Any],
    targets: Any,
    predictions: Any,
    *,
    dimensions: Sequence[str] = ("MI", "TA"),
    metric: str | MetricFunction | Mapping[str, str | MetricFunction] = "mse",
    n_resamples: int = 2000,
    seed: int = 20260907,
    confidence_level: float = 0.95,
    segment_scores: Any = None,
) -> dict[str, BootstrapResult]:
    """Return synchronized clip-level bootstrap results for multiple dimensions."""

    ids = _normalise_clip_ids(clip_ids)
    names = tuple(str(item).upper() for item in dimensions)
    if not names:
        raise ValueError("at least one dimension is required")
    true_map = _dimension_arrays(targets, dimensions=names, name="target", n=len(ids))
    pred_map = _dimension_arrays(
        predictions, dimensions=names, name="prediction", n=len(ids)
    )
    # A dimension mapping is accepted for segment vectors.  A single matrix or
    # ragged sequence is shared by all dimensions.
    segment_map: dict[str, Any] = {}
    if isinstance(segment_scores, Mapping) and any(
        str(key).upper() in names for key in segment_scores
    ):
        segment_map = {str(key).upper(): value for key, value in segment_scores.items()}
    confidence = _validate_confidence(confidence_level)
    indices = resample_clip_indices(
        len(ids), n_resamples=n_resamples, seed=seed
    )
    output: dict[str, BootstrapResult] = {}
    for dimension in names:
        chosen_metric: str | MetricFunction
        if isinstance(metric, Mapping):
            lookup = {str(key).upper(): value for key, value in metric.items()}
            if dimension not in lookup:
                raise ValueError(f"metric mapping is missing dimension {dimension}")
            chosen_metric = lookup[dimension]
        else:
            chosen_metric = metric
        _segment_vectors(
            segment_map.get(dimension, segment_scores), ids
        )
        metric_name, metric_function = _metric_function(chosen_metric)
        output[dimension] = _bootstrap_from_indices(
            ids,
            true_map[dimension],
            pred_map[dimension],
            metric_name=metric_name,
            metric_function=metric_function,
            indices=indices,
            confidence_level=confidence,
            seed=int(seed),
        )
    return output


# Names used by a few downstream scripts/notebooks.
clip_bootstrap = bootstrap_clip_metric
bootstrap_metric = bootstrap_clip_metric


__all__ = [
    "BootstrapResult",
    "bootstrap_clip_metric",
    "bootstrap_clip_metrics",
    "bootstrap_metric",
    "clip_bootstrap",
    "resample_clip_indices",
    "resample_clip_records",
]
