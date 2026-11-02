"""Reproducible populations and demo settings."""

FEATURES = (
    "credit_score",
    "annual_income",
    "debt_to_income",
    "employment_years",
    "open_credit_accounts",
)
REFERENCE_SEED = 101
STABLE_SEED = 202
SHIFTED_SEED = 303
LABEL_SEED = 404
SPLIT_SEED = 505
REFERENCE_SIZE = 10_000
BATCH_SIZE = 5_000
MODEL_NAME = "loan-risk-demo-0.2"
SCENARIOS = {
    "A": (STABLE_SEED, False, "2025-01-06"),
    "B": (SHIFTED_SEED, True, "2025-01-13"),
}
