"""Tests for deterministic synthetic data and model persistence."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from examples.loan_risk_demo.config import (
    BATCH_SIZE,
    REFERENCE_SEED,
    REFERENCE_SIZE,
    SHIFTED_SEED,
    SPLIT_SEED,
    STABLE_SEED,
)
from examples.loan_risk_demo.generate_data import generate_data
from examples.loan_risk_demo.train_model import train_model


def test_data_determinism_and_shift_is_limited_to_selected_features():
    for seed, shifted in (
        (REFERENCE_SEED, False),
        (STABLE_SEED, False),
        (SHIFTED_SEED, True),
    ):
        assert np.all(generate_data(REFERENCE_SIZE, seed, shifted)[0] >= 0)
    for seed in (REFERENCE_SEED, STABLE_SEED, SHIFTED_SEED):
        first = generate_data(1000, seed)
        second = generate_data(1000, seed)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
    stable, _ = generate_data(BATCH_SIZE, SHIFTED_SEED)
    shifted, _ = generate_data(BATCH_SIZE, SHIFTED_SEED, shifted=True)
    np.testing.assert_array_equal(stable[:, 3:], shifted[:, 3:])
    assert all(not np.array_equal(stable[:, i], shifted[:, i]) for i in range(3))


def test_model_reload_auc_and_shifted_predictions(tmp_path):
    x, y = generate_data(REFERENCE_SIZE, REFERENCE_SEED)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=SPLIT_SEED, stratify=y
    )
    original = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=SPLIT_SEED)
    )
    original.fit(x_train, y_train)
    model, auc = train_model(x, y, tmp_path / "model.joblib")
    import joblib

    reloaded = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(original.predict_proba(x), reloaded.predict_proba(x))
    np.testing.assert_array_equal(model.predict_proba(x), reloaded.predict_proba(x))
    # The noisy synthetic rule has useful signal; 0.7 allows substantial sampling variation.
    assert auc > 0.7
    assert roc_auc_score(y_test, reloaded.predict_proba(x_test)[:, 1]) == auc
    stable, _ = generate_data(BATCH_SIZE, STABLE_SEED)
    shifted, _ = generate_data(BATCH_SIZE, SHIFTED_SEED, shifted=True)
    assert (
        abs(
            reloaded.predict_proba(stable)[:, 1].mean()
            - reloaded.predict_proba(shifted)[:, 1].mean()
        )
        > 0.05
    )
