"""Generate independent synthetic features and noisy loan-risk labels."""

import numpy as np

from examples.loan_risk_demo.config import FEATURES, LABEL_SEED


def generate_data(size: int, seed: int, shifted: bool = False):
    rng = np.random.default_rng(seed)
    features = np.column_stack(
        (
            rng.normal(650 if shifted else 700, 50, size),
            np.maximum(rng.normal(50_000 if shifted else 70_000, 15_000, size), 0),
            np.maximum(rng.normal(0.42 if shifted else 0.28, 0.08, size), 0),
            np.maximum(rng.normal(8, 4, size), 0),
            rng.poisson(5, size),
        )
    )
    credit, income, debt, employment, accounts = features.T
    log_odds = (
        -0.5
        - 0.9 * (credit - 700) / 50
        - 0.5 * (income - 70_000) / 15_000
        + 1.0 * (debt - 0.28) / 0.08
        - 0.2 * (employment - 8) / 4
        + 0.1 * (accounts - 5)
    )
    probability = 1 / (1 + np.exp(-log_odds))
    labels = np.random.default_rng([seed, LABEL_SEED]).binomial(1, probability)
    return features, labels


def feature_payload(features):
    return {name: features[:, i].tolist() for i, name in enumerate(FEATURES)}
