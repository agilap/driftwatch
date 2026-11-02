from __future__ import annotations

import pytest

from app.services.importance_scorer import (
    compute_weighted_score,
    equal_weight_importances,
    import_from_json,
    normalise_importances,
    rank_features_by_weighted_score,
    resolve_feature_weights,
)


def test_normalise_importances_sums_to_one() -> None:
    normalized = normalise_importances({"income": 2.0, "credit_score": 3.0, "age": 5.0})

    assert sum(normalized.values()) == pytest.approx(1.0)
    assert normalized["age"] == pytest.approx(0.5)


def test_normalise_importances_negative_raises() -> None:
    with pytest.raises(ValueError):
        normalise_importances({"income": -1.0, "credit_score": 1.0})


def test_normalise_importances_all_zero_raises() -> None:
    with pytest.raises(ValueError):
        normalise_importances({"income": 0.0, "credit_score": 0.0})


def test_equal_weight_correct_value() -> None:
    weights = equal_weight_importances(["a", "b", "c", "d", "e"])

    assert set(weights.keys()) == {"a", "b", "c", "d", "e"}
    assert all(value == pytest.approx(0.2) for value in weights.values())


def test_compute_weighted_score_high_importance_high_drift() -> None:
    weighted = compute_weighted_score(drift_magnitude=0.99, feature_importance=0.99)

    assert weighted == pytest.approx(0.9801)


def test_compute_weighted_score_low_importance_high_drift() -> None:
    weighted = compute_weighted_score(drift_magnitude=0.95, feature_importance=0.05)

    assert weighted == pytest.approx(0.0475)


def test_compute_weighted_score_clipped_to_one() -> None:
    weighted = compute_weighted_score(drift_magnitude=5.0, feature_importance=2.0)

    assert weighted == pytest.approx(1.0)


def test_rank_features_sorted_descending() -> None:
    ranked = rank_features_by_weighted_score(
        drift_scores=[
            {"feature_name": "income", "psi": 0.20},
            {"feature_name": "age", "psi": 0.30},
            {"feature_name": "credit_score", "psi": 0.10},
        ],
        importances={"income": 0.5, "age": 0.2, "credit_score": 0.3},
    )

    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert (
        ranked[0]["weighted_score"]
        >= ranked[1]["weighted_score"]
        >= ranked[2]["weighted_score"]
    )
    assert ranked[0]["feature_name"] == "income"


def test_import_from_json_valid() -> None:
    parsed = import_from_json(
        json_data={"income": 2, "credit_score": 1.0, "age": 1.0},
        method="manual",
    )

    assert sum(parsed.values()) == pytest.approx(1.0)
    assert parsed["income"] == pytest.approx(0.5)


def test_import_from_json_invalid_raises() -> None:
    with pytest.raises(ValueError):
        import_from_json(
            json_data={"income": "bad", "credit_score": 1.0},
            method="shap",
        )


@pytest.mark.parametrize(
    "importances, expected",
    [
        ({"income": 2.0, "unrelated": 100.0}, {"income": 1.0, "age": 0.0}),
        ({"unrelated": 100.0}, {"income": 0.5, "age": 0.5}),
        ({"income": 0.0, "age": 0.0}, {"income": 0.5, "age": 0.5}),
    ],
)
def test_resolve_weights_only_analyzed_features(importances, expected) -> None:
    assert resolve_feature_weights(["income", "age"], importances) == expected


def test_rank_uses_red_threshold_without_observed_rescaling() -> None:
    ranked = rank_features_by_weighted_score(
        [
            {"feature_name": "income", "psi": 0.1},
            {"feature_name": "age", "psi": 20.0},
        ],
        {"income": 3.0, "age": 1.0, "unrelated": 100.0},
    )
    scores = {row["feature_name"]: row["weighted_score"] for row in ranked}
    assert scores == pytest.approx({"income": 0.375, "age": 0.25})


def test_weighted_score_preserves_precision() -> None:
    assert compute_weighted_score(1.0, 1.0 / 3.0) == 1.0 / 3.0


def test_resolve_weights_preserves_missing_feature_zero() -> None:
    positive_features = [f"feature_{i}" for i in range(6)]
    weights = resolve_feature_weights(
        ["missing", *positive_features], {feature: 1.0 for feature in positive_features}
    )
    assert weights["missing"] == 0.0
    assert sum(weights.values()) == 1.0


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_resolve_weights_rejects_invalid_analyzed_importance(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_feature_weights(["income", "age"], {"income": invalid})


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_resolve_weights_ignores_invalid_unrelated_importance(invalid: float) -> None:
    assert resolve_feature_weights(["income"], {"unrelated": invalid}) == {
        "income": 1.0
    }


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("invalid_argument", ["drift_magnitude", "feature_importance"])
def test_weighted_score_rejects_invalid_values(
    invalid: float, invalid_argument: str
) -> None:
    values = {"drift_magnitude": 0.5, "feature_importance": 0.5}
    values[invalid_argument] = invalid
    with pytest.raises(ValueError, match="finite and non-negative"):
        compute_weighted_score(**values)
