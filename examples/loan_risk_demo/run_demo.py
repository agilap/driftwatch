"""Run the loan-risk demonstration against a local DriftWatch API."""

import os
import sys
import time

import httpx
from sklearn.metrics import roc_auc_score

from examples.loan_risk_demo.config import (
    BATCH_SIZE,
    FEATURES,
    MODEL_NAME,
    REFERENCE_SEED,
    REFERENCE_SIZE,
    SCENARIOS,
)
from examples.loan_risk_demo.generate_data import feature_payload, generate_data
from examples.loan_risk_demo.train_model import ARTIFACTS, train_model


def model_id(client):
    page = 1
    while True:
        response = client.get("/models", params={"page": page, "page_size": 100})
        response.raise_for_status()
        result = response.json()
        for item in result["items"]:
            if item["name"] == MODEL_NAME:
                return item["id"]
        if page * 100 >= result["total"]:
            break
        page += 1
    response = client.post("/models", json={"name": MODEL_NAME, "version": "0.2"})
    response.raise_for_status()
    return response.json()["id"]


def wait_for_drift(client, identifier, date, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/drift/{identifier}/{date}")
        response.raise_for_status()
        rows = response.json()
        if {row["feature_name"] for row in rows} >= set(FEATURES) | {"__predictions__"}:
            return {row["feature_name"]: row for row in rows}
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for drift analysis for {date}")


def run(client):
    try:
        ready = client.get("/ready")
    except httpx.HTTPError as exc:
        raise RuntimeError(f"DriftWatch readiness check failed: {exc}") from exc
    if ready.status_code != 200:
        raise RuntimeError(
            f"DriftWatch /ready returned {ready.status_code}, expected 200"
        )

    reference, labels = generate_data(REFERENCE_SIZE, REFERENCE_SEED)
    model, auc = train_model(reference, labels)
    print(f"Held-out model AUC: {auc:.4f}")
    identifier = model_id(client)
    coefficients = model.named_steps["logisticregression"].coef_[0]
    response = client.post(
        "/ingest/reference",
        json={
            "model_id": identifier,
            "features": feature_payload(reference),
            "predictions": model.predict_proba(reference)[:, 1].tolist(),
            "feature_importances": dict(zip(FEATURES, abs(coefficients).tolist())),
            "importance_method": "coefficient",
        },
    )
    response.raise_for_status()
    results = {}
    reports = {}
    for scenario, (seed, shifted, date) in SCENARIOS.items():
        batch, truth = generate_data(BATCH_SIZE, seed, shifted)
        predictions = model.predict_proba(batch)[:, 1]
        response = client.post(
            "/ingest/snapshot",
            json={
                "model_id": identifier,
                "timestamp": f"{date}T12:00:00Z",
                "features": feature_payload(batch),
                "predictions": predictions.tolist(),
            },
        )
        response.raise_for_status()
        results[scenario] = wait_for_drift(client, identifier, date)
        response = client.post(
            f"/reports/{identifier}/generate", json={"week_start": date}
        )
        response.raise_for_status()
        reports[scenario] = response.json()
        (ARTIFACTS / f"report_{scenario.lower()}.md").write_text(
            reports[scenario]["report_markdown"], encoding="utf-8"
        )
        print(
            f"\nScenario {scenario}: batch AUC {roc_auc_score(truth, predictions):.4f}"
        )
        print(f"{'Feature':24} {'PSI':>9} {'Severity':>9}")
        for name in (*FEATURES, "__predictions__"):
            row = results[scenario][name]
            print(f"{name:24} {row['psi']:9.4f} {row['severity']:>9}")
        print(f"Health: {reports[scenario]['overall_score']:.2f}")

    response = client.get(f"/alerts/{identifier}", params={"page_size": 100})
    response.raise_for_status()
    alerts = response.json()["items"]
    print("\nAlerts:")
    for alert in alerts:
        print(f"  {alert['message']}")
    if not alerts:
        print("  None")

    failures = []
    for name in FEATURES:
        if results["A"][name]["severity"] != "green":
            failures.append(f"Scenario A {name} is not green")
        expected = "red" if name in FEATURES[:3] else "green"
        if results["B"][name]["severity"] != expected:
            failures.append(f"Scenario B {name} is not {expected}")
    if results["A"]["__predictions__"]["severity"] != "green":
        failures.append("Scenario A prediction drift is not green")
    if results["B"]["__predictions__"]["severity"] == "green":
        failures.append("Scenario B prediction drift is green")
    if not any(SCENARIOS["B"][2] in (alert["message"] or "") for alert in alerts):
        failures.append("Scenario B has no alert")
    if reports["B"]["overall_score"] >= reports["A"]["overall_score"]:
        failures.append("Scenario B health is not below scenario A health")
    if failures:
        raise RuntimeError("Expectations failed:\n- " + "\n- ".join(failures))
    print("All expectations passed.")


def main():
    try:
        with httpx.Client(
            base_url=os.environ.get("DRIFTWATCH_URL", "http://localhost:8000"),
            timeout=30,
        ) as client:
            run(client)
    except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
