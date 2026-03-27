from __future__ import annotations

import pytest

from app.services.importance_scorer import (
    compute_weighted_score,
    equal_weight_importances,
    import_from_json,
    normalise_importances,
    rank_features_by_weighted_score,
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
    assert ranked[0]["weighted_score"] >= ranked[1]["weighted_score"] >= ranked[2]["weighted_score"]
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
