from __future__ import annotations

import os
import time
from datetime import date
from uuid import uuid4

import httpx
import numpy as np

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def _endpoint(path: str) -> str:
    return f"{BASE_URL.rstrip('/')}{path}"


def _generate_january_reference(seed: int = 7) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {
        "income": rng.normal(52000, 3500, 500).round(2).tolist(),
        "credit_score": rng.normal(710, 25, 500).round(2).tolist(),
        "loan_term_months": rng.choice(
            [24, 36, 48, 60], size=500, p=[0.1, 0.55, 0.2, 0.15]
        )
        .astype(float)
        .tolist(),
    }


def _generate_january_snapshot(reference: dict[str, list[float]]) -> dict[str, list[float]]:
    """Create a no-drift January snapshot by sampling directly from reference values."""
    return {feature: values[:300] for feature, values in reference.items()}


def _generate_august_snapshot(seed: int = 177) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    return {
        "income": rng.normal(38000, 4500, 300).round(2).tolist(),
        "credit_score": rng.normal(645, 28, 300).round(2).tolist(),
        "loan_term_months": rng.choice(
            [24, 36, 48, 60], size=300, p=[0.15, 0.5, 0.2, 0.15]
        )
        .astype(float)
        .tolist(),
    }


def _poll_drift_scores(
    client: httpx.Client, model_id: str, window_date: str, timeout_s: float = 8.0
) -> list[dict]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(
            _endpoint(f"/drift/{model_id}/{window_date}"), timeout=10.0
        )
        if response.status_code == 200 and response.json():
            return response.json()
        time.sleep(0.4)
    raise RuntimeError(f"Timed out waiting for drift scores for {window_date}")


def _wait_for_api(client: httpx.Client, timeout_s: float = 20.0) -> None:
    """Wait until the API health endpoint is available."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = client.get(_endpoint("/health"), timeout=5.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("API did not become ready in time")


def main() -> None:
    model_name = f"loan-scorer-v3-{uuid4()}"
    january_reference = _generate_january_reference()
    january_snapshot = _generate_january_snapshot(january_reference)
    august_snapshot = _generate_august_snapshot()
    importances = {"income": 0.45, "credit_score": 0.40, "loan_term_months": 0.15}

    with httpx.Client(timeout=30.0) as client:
        _wait_for_api(client)
        model_response = client.post(
            _endpoint("/models"), json={"name": model_name, "version": "v3"}
        )
        model_response.raise_for_status()
        model_id = model_response.json()["id"]
        print(f"Model created: {model_name} ({model_id})")

        ref_response = client.post(
            _endpoint("/ingest/reference"),
            json={
                "model_id": model_id,
                "features": january_reference,
                "feature_importances": importances,
                "importance_method": "manual",
            },
        )
        ref_response.raise_for_status()
        print("Reference distribution uploaded (January baseline)")

        jan_ingest = client.post(
            _endpoint("/ingest/snapshot"),
            json={
                "model_id": model_id,
                "timestamp": "2026-01-15T12:00:00Z",
                "features": january_snapshot,
            },
        )
        jan_ingest.raise_for_status()
        jan_scores = _poll_drift_scores(client, model_id, "2026-01-15")
        jan_green = all(item["severity"] == "green" for item in jan_scores)
        print(
            "January: all green" if jan_green else "January: non-green drift detected"
        )

        aug_ingest = client.post(
            _endpoint("/ingest/snapshot"),
            json={
                "model_id": model_id,
                "timestamp": "2026-08-15T12:00:00Z",
                "features": august_snapshot,
            },
        )
        aug_ingest.raise_for_status()
        aug_scores = _poll_drift_scores(client, model_id, "2026-08-15")

        print("\nAugust drift scores:")
        for item in sorted(
            aug_scores,
            key=lambda x: float(x.get("weighted_score") or 0.0),
            reverse=True,
        ):
            print(
                f"- {item['feature_name']}: PSI={item['psi']:.3f} "
                f"severity={item['severity']} weighted={item['weighted_score']:.3f}"
            )

        report_response = client.post(
            _endpoint(f"/reports/{model_id}/generate"),
            json={"week_start": date(2026, 8, 10).isoformat()},
        )
        report_response.raise_for_status()
        report = report_response.json()
        print("\nWeekly report summary:")
        print(f"- Overall score: {report['overall_score']}")
        top_feature = report["report_json"]["top_drifted_features"][0]["feature_name"]
        print(f"- Top drifted feature: {top_feature}")

        jan_by_feature = {item["feature_name"]: item for item in jan_scores}
        aug_by_feature = {item["feature_name"]: item for item in aug_scores}
        print("\nBefore/After PSI table")
        print("Feature              | January PSI | August PSI")
        print("---------------------|-------------|-----------")
        for feature in ["income", "credit_score", "loan_term_months"]:
            jan_psi = float(jan_by_feature[feature]["psi"])
            aug_psi = float(aug_by_feature[feature]["psi"])
            print(f"{feature:<21}| {jan_psi:<11.3f}| {aug_psi:.3f}")


if __name__ == "__main__":
    main()
