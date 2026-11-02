"""Database-free statistical oracles for aligned persisted histogram counts."""

import numpy as np
import pytest

from app.services.drift_engine import (
    compute_binned_ks,
    compute_chi_square,
    compute_js_divergence,
    compute_js_divergence_from_counts,
    compute_ks_test,
    compute_psi,
    compute_psi_from_counts,
)


COUNT_METRICS = [
    compute_psi_from_counts,
    compute_js_divergence_from_counts,
    compute_binned_ks,
    compute_chi_square,
]


def test_psi_hand_calculated() -> None:
    # Smoothed probabilities are (0.25+.0001, .75+.0001)/1.0002 and reversed.
    expected = np.log(0.7501 / 0.2501) / 1.0002
    assert compute_psi_from_counts([1, 3], [3, 1]) == pytest.approx(expected)


def test_js_hand_calculated() -> None:
    # P=(1,0), Q=(1/2,1/2), M=(3/4,1/4).
    expected = 0.5 * np.log2(4 / 3) + 0.25 * np.log2(2 / 3) + 0.25
    assert compute_js_divergence_from_counts([2, 0], [1, 1]) == pytest.approx(expected)
    assert compute_js_divergence_from_counts([1, 0], [0, 1]) == pytest.approx(1.0)


def test_binned_ks_hand_calculated_and_sample_lower_bound() -> None:
    assert compute_binned_ks([1, 2, 1], [0, 1, 3]) == pytest.approx(0.5)
    # Shared bins [0, 1), [1, 2] hide within-bin differences.
    ref, prod = [0.1, 0.2, 1.1, 1.2], [0.8, 0.9, 1.8, 1.9]
    assert compute_binned_ks([2, 2], [2, 2]) == 0.0
    assert compute_binned_ks([2, 2], [2, 2]) < compute_ks_test(ref, prod)["statistic"]


@pytest.mark.parametrize("metric", COUNT_METRICS[:3])
def test_identical_zero_count_bins_and_large_counts(metric) -> None:
    assert metric([0, 3, 0, 1], [0, 3, 0, 1]) == pytest.approx(0.0)
    assert metric([1e308, 1e308], [1e308, 1e308]) == pytest.approx(0.0)


@pytest.mark.parametrize("metric", COUNT_METRICS)
@pytest.mark.parametrize(
    "ref, prod",
    [
        ([], []),
        ([1], [1, 2]),
        ([1, -1], [1, 2]),
        ([1, 2], [-1, 2]),
        ([np.nan], [1]),
        ([1], [np.inf]),
        ([[1, 2]], [[1, 2]]),
        ([0, 0], [1, 2]),
        ([1, 2], [0, 0]),
    ],
)
def test_count_metrics_reject_invalid_inputs(metric, ref, prod) -> None:
    with pytest.raises(ValueError):
        metric(ref, prod)


@pytest.mark.parametrize(
    "metric", [compute_psi, compute_js_divergence, compute_ks_test]
)
@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1, 2]]])
def test_sample_metrics_reject_invalid_inputs(metric, values) -> None:
    with pytest.raises(ValueError):
        metric(values, [1, 2])
    with pytest.raises(ValueError):
        metric([1, 2], values)


@pytest.mark.parametrize("metric", [compute_psi, compute_js_divergence])
@pytest.mark.parametrize("bins", [0, -1, 1.5, True, "10", None])
def test_sample_metrics_reject_invalid_bin_counts(metric, bins) -> None:
    with pytest.raises(ValueError):
        metric([-2, -1], [-1, 0], bins=bins)


@pytest.mark.parametrize("metric", [compute_psi, compute_js_divergence])
def test_negative_samples_constants_and_small_samples_are_valid(metric) -> None:
    assert metric([-2], [-2]) == pytest.approx(0.0)
    assert metric([-2, -1], [-2, -1]) == pytest.approx(0.0)
    assert np.isfinite(metric([-2], [-1]))
    assert metric([-2], [-1]) > 0


def test_small_change_is_smaller_than_significant_shift() -> None:
    for metric in COUNT_METRICS[:3]:
        small = metric([50, 50], [49, 51])
        large = metric([50, 50], [1, 99])
        assert 0 < small < large


def test_chi_square_unseen_categories_and_zero_columns() -> None:
    # Pearson statistic = 4*(5^2/5)=20, df=1; chi2_contingency applies Yates correction.
    result = compute_chi_square([10, 0, 0], [0, 10, 0])
    assert result["statistic"] == pytest.approx(16.2)
    assert result["pvalue"] == pytest.approx(0.000056994116232)
    assert result["significant"] is True


def test_chi_square_single_category_is_degenerate() -> None:
    assert compute_chi_square([0, 3], [0, 7]) == {
        "statistic": 0.0,
        "pvalue": 1.0,
        "significant": False,
    }
