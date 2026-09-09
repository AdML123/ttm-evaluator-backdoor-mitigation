import numpy as np
import pytest

from src.metrics.bootstrap import (
    bootstrap_clip_metric,
    bootstrap_clip_metrics,
    resample_clip_indices,
    resample_clip_records,
)


def test_resample_indices_are_deterministic_and_clip_level():
    first = resample_clip_indices(3, n_resamples=20, seed=17)
    second = resample_clip_indices(3, n_resamples=20, seed=17)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 3)
    assert np.all((first >= 0) & (first < 3))


def test_resample_clip_records_keeps_all_segment_scores_together():
    records = [
        {"clip_id": "a", "target": 1.0, "prediction": 2.0, "segments": [10.0, 11.0]},
        {"clip_id": "b", "target": 2.0, "prediction": 1.0, "segments": [20.0, 21.0]},
    ]
    indices = np.array([[1, 0], [0, 0]], dtype=np.int64)

    sampled = resample_clip_records(records, indices)

    assert sampled[0][0]["clip_id"] == "b"
    assert sampled[0][0]["segments"] == [20.0, 21.0]
    assert sampled[0][1]["clip_id"] == "a"
    assert sampled[1][0]["clip_id"] == "a"
    assert sampled[1][1]["segments"] == [10.0, 11.0]


def test_bootstrap_metric_returns_point_estimate_and_reproducible_ci():
    ids = ["a", "b", "c", "d"]
    target = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([1.0, 3.0, 2.0, 5.0])

    first = bootstrap_clip_metric(
        ids,
        target,
        prediction,
        metric="mse",
        n_resamples=200,
        seed=99,
    )
    second = bootstrap_clip_metric(
        ids,
        target,
        prediction,
        metric="mse",
        n_resamples=200,
        seed=99,
    )

    assert first.estimate == pytest.approx(0.75)
    assert first.lower <= first.estimate <= first.upper
    assert first.confidence_interval == first.ci
    assert first.ci_low == first.lower and first.ci_high == first.upper
    assert first.n_clips == 4
    np.testing.assert_array_equal(first.samples, second.samples)
    np.testing.assert_array_equal(first.resample_indices, second.resample_indices)


def test_bootstrap_metrics_supports_two_dimensions():
    result = bootstrap_clip_metrics(
        ["a", "b", "c"],
        np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 1.0], [3.0, 2.0], [2.0, 5.0]]),
        n_resamples=32,
        seed=2,
    )

    assert set(result) == {"MI", "TA"}
    assert result["MI"].metric == "mse"
    assert result["TA"].n_resamples == 32


def test_bootstrap_rejects_duplicate_clip_ids():
    with pytest.raises(ValueError, match="unique"):
        bootstrap_clip_metric(["a", "a"], [1, 2], [1, 2], n_resamples=2)
