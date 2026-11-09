"""Database-free statistical oracles for aligned persisted histogram counts."""

from tests.helpers import sample_psi

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from app.services.drift_engine import (
    compute_binned_ks,
    compute_js_divergence_from_counts,
    compute_psi_from_counts,
)


COUNT_METRICS = [
    compute_psi_from_counts,
    compute_js_divergence_from_counts,
    compute_binned_ks,
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


@pytest.mark.parametrize(
    "ref, prod",
    [
        ([1, 2, 3], [3, 4, 1]),
        ([0, 3, 0, 1], [2, 0, 0, 4]),
        ([1, 0], [0, 1]),
        ([0, 5], [0, 9]),
    ],
)
def test_js_matches_scipy(ref, prod) -> None:
    assert compute_js_divergence_from_counts(ref, prod) == pytest.approx(
        jensenshannon(ref, prod, base=2) ** 2
    )


def test_binned_ks_hand_calculated_and_sample_lower_bound() -> None:
    assert compute_binned_ks([1, 2, 1], [0, 1, 3]) == pytest.approx(0.5)
    # Shared bins [0, 1), [1, 2] hide within-bin differences.
    ref, prod = [0.1, 0.2, 1.1, 1.2], [0.8, 0.9, 1.8, 1.9]
    assert compute_binned_ks([2, 2], [2, 2]) == 0.0
    assert compute_binned_ks([2, 2], [2, 2]) < ks_2samp(ref, prod).statistic


@pytest.mark.parametrize("metric", COUNT_METRICS)
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


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [[1, 2]]])
def test_sample_metrics_reject_invalid_inputs(values) -> None:
    with pytest.raises(ValueError):
        sample_psi(values, [1, 2])
    with pytest.raises(ValueError):
        sample_psi([1, 2], values)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True, "10", None])
def test_sample_metrics_reject_invalid_bin_counts(bins) -> None:
    with pytest.raises(ValueError):
        sample_psi([-2, -1], [-1, 0], bins=bins)


def test_negative_samples_constants_and_small_samples_are_valid() -> None:
    assert sample_psi([-2], [-2]) == pytest.approx(0.0)
    assert sample_psi([-2, -1], [-2, -1]) == pytest.approx(0.0)
    assert np.isfinite(sample_psi([-2], [-1]))
    assert sample_psi([-2], [-1]) > 0


def test_small_change_is_smaller_than_significant_shift() -> None:
    for metric in COUNT_METRICS:
        small = metric([50, 50], [49, 51])
        large = metric([50, 50], [1, 99])
        assert 0 < small < large
