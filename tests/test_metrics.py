import numpy as np
import pytest

from src.metrics.evaluate import (
    bland_altman_limits,
    compute_regression_metrics,
    evaluate_clip_predictions,
    make_variance_subgroups,
    segment_dispersion,
)


def test_regression_metrics_include_error_and_rank_statistics():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([2.0, 1.0, 4.0, 3.0])

    result = compute_regression_metrics(target, prediction)

    assert result["n"] == 4
    assert result["mse"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(1.0)
    assert result["signed_error"] == pytest.approx(0.0)
    assert result["pearson_r"] == pytest.approx(0.6)
    assert result["pearson"] == pytest.approx(result["pearson_r"])
    assert result["spearman_rho"] == pytest.approx(0.6)
    assert result["srcc"] == pytest.approx(result["spearman_rho"])
    assert result["bland_altman_bias"] == pytest.approx(0.0)
    assert result["loa_lower"] < 0 < result["loa_upper"]


def test_constant_rank_inputs_are_reported_as_not_defensible():
    result = compute_regression_metrics([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

    assert np.isnan(result["pearson_r"])
    assert np.isnan(result["spearman_rho"])


def test_bland_altman_limits_use_prediction_minus_target():
    bias, lower, upper = bland_altman_limits([1.0, 2.0], [2.0, 4.0])

    assert bias == pytest.approx(1.5)
    assert lower == pytest.approx(1.5 - 1.96 * np.sqrt(0.5))
    assert upper == pytest.approx(1.5 + 1.96 * np.sqrt(0.5))


def test_evaluate_clip_predictions_handles_two_dimensions_and_ids():
    rows = [
        {"clip_id": "b.wav", "mi": 2.0, "ta": 3.0, "pred_mi": 2.5, "pred_ta": 2.0},
        {"clip_id": "a.wav", "mi": 4.0, "ta": 1.0, "pred_mi": 4.0, "pred_ta": 2.0},
    ]

    result = evaluate_clip_predictions(rows)

    assert set(result) == {"MI", "TA"}
    assert result["MI"]["n"] == 2
    assert result["TA"]["mae"] == pytest.approx(1.0)


def test_segment_dispersion_and_subgroups_are_clip_aligned():
    segments = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0], [1.0, 1.0, 2.0]])

    dispersion = segment_dispersion(segments)
    groups = make_variance_subgroups(segments, threshold=0.6)

    np.testing.assert_allclose(dispersion, [np.sqrt(2 / 3), 0.0, np.sqrt(2 / 9)])
    np.testing.assert_array_equal(groups["high_mask"], [True, False, False])
    np.testing.assert_array_equal(groups["low_mask"], [False, True, True])


def test_variance_median_ties_are_low_by_protocol():
    segments = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 4.0]])
    groups = make_variance_subgroups(segments, threshold=0.5)

    np.testing.assert_array_equal(groups["high_mask"], [False, False, False])
    np.testing.assert_array_equal(groups["low_mask"], [True, True, True])


def test_metric_input_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same number"):
        compute_regression_metrics([1.0, 2.0], [1.0])


def test_drop_nonfinite_filters_paired_observations_without_realigning():
    result = compute_regression_metrics(
        [1.0, np.nan, 3.0], [1.0, 2.0, np.inf], drop_nonfinite=True
    )

    assert result["n"] == 1
    assert result["mse"] == pytest.approx(0.0)
