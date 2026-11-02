"""Train and persist the synthetic loan-risk model."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from examples.loan_risk_demo.config import REFERENCE_SEED, REFERENCE_SIZE, SPLIT_SEED
from examples.loan_risk_demo.generate_data import generate_data

ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS / "loan_risk.joblib"


def train_model(features, labels, path=MODEL_PATH):
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=SPLIT_SEED, stratify=labels
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SPLIT_SEED))
    model.fit(x_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Artifacts are locally generated pickles; never load them from untrusted sources.
    joblib.dump(model, path)
    return joblib.load(path), auc


if __name__ == "__main__":
    x, y = generate_data(REFERENCE_SIZE, REFERENCE_SEED)
    _, score = train_model(x, y)
    print(f"Held-out AUC: {score:.4f}")
