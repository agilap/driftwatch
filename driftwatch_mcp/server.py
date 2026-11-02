"""MCP tools backed by the DriftWatch REST API.

This package computes no statistics. Every score, severity and health value comes
from the monitoring service, so MCP clients cannot disagree with its formulas.
"""

import json
import os
from datetime import date
from functools import wraps
from typing import Annotated, Any
from uuid import UUID

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import Field

from app.schemas.ingest import PREDICTIONS_FEATURE_NAME

LIMITATION = (
    "Drift measures input and prediction distribution change; "
    "it does not establish performance degradation."
)
ALERT_PAGE_SIZE = 100


class ApiError(Exception):
    """A DriftWatch request that a client can branch on by code."""

    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.payload = {"code": code, "message": message, "status": status}


def _result(payload: dict[str, Any], is_error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=is_error,
    )


def _structured(tool):
    """Return JSON on success, and an isError result with a stable code on failure."""

    @wraps(tool)
    async def wrapper(*args: Any, **kwargs: Any) -> CallToolResult:
        try:
            return _result(await tool(*args, **kwargs))
        except ApiError as exc:
            return _result({"error": exc.payload}, is_error=True)

    return wrapper


async def _get(client: httpx.AsyncClient, path: str, **params: Any) -> Any:
    try:
        response = await client.get(path, params=params or None)
    except httpx.HTTPError as exc:
        raise ApiError(
            "service_unavailable", f"DriftWatch API request failed: {exc}"
        ) from exc
    if response.is_success:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text
    code = "not_found" if response.status_code == 404 else "api_error"
    raise ApiError(code, str(detail), response.status_code)


async def _get_or_none(client: httpx.AsyncClient, path: str) -> Any:
    try:
        return await _get(client, path)
    except ApiError as exc:
        if exc.payload["code"] == "not_found":
            return None
        raise


async def _active_alerts(client: httpx.AsyncClient, model_id: UUID) -> list[dict]:
    alerts: list[dict] = []
    page = 1
    while True:
        items = (
            await _get(
                client,
                f"/alerts/{model_id}",
                resolved=False,
                page=page,
                page_size=ALERT_PAGE_SIZE,
            )
        )["items"]
        alerts.extend(items)
        if len(items) < ALERT_PAGE_SIZE:
            return alerts
        page += 1


async def _feature_drift(
    client: httpx.AsyncClient, model_id: UUID, window_date: date | None
) -> tuple[str, list[dict], dict | None]:
    """Return a window's scored features and its separate prediction row."""
    if window_date is None:
        window = (await _get(client, f"/drift/{model_id}/latest"))["window_date"]
    else:
        window = window_date.isoformat()
    rows = await _get(client, f"/drift/{model_id}/{window}")
    features = [r for r in rows if r["feature_name"] != PREDICTIONS_FEATURE_NAME]
    prediction = next(
        (r for r in rows if r["feature_name"] == PREDICTIONS_FEATURE_NAME), None
    )
    return window, features, prediction


def _status(severities: list[str | None]) -> str:
    if not severities:
        return "no_data"
    if "red" in severities:
        return "critical"
    # The API allows null severity; absent evidence must never read as healthy.
    if None in severities:
        return "unknown"
    return "attention" if "yellow" in severities else "healthy"


async def _health(client: httpx.AsyncClient, model_id: UUID) -> dict[str, Any]:
    model = await _get(client, f"/models/{model_id}")
    report = await _get_or_none(client, f"/reports/{model_id}/latest")
    try:
        window, features, prediction = await _feature_drift(client, model_id, None)
    except ApiError as exc:
        if exc.payload["code"] != "not_found":
            raise
        window, features, prediction = None, [], None
    severities = [row["severity"] for row in features]
    return {
        "model": model,
        "status": _status(
            severities + ([prediction["severity"]] if prediction else [])
        ),
        "latest_window": window,
        "severity_counts": {s: severities.count(s) for s in ("red", "yellow", "green")},
        "prediction_severity": prediction["severity"] if prediction else None,
        "overall_score": report["overall_score"] if report else None,
        "report_week_start": report["week_start"] if report else None,
        "report_covers_latest_window": bool(
            report and window and report["week_start"] <= window <= report["week_end"]
        ),
        "features": features,
        "prediction": prediction,
    }


def build_server(client: httpx.AsyncClient) -> FastMCP:
    """Build the server around an injected client, so tests can target any API."""
    server = FastMCP("driftwatch")

    @server.tool(structured_output=False)
    @_structured
    async def list_models(
        page: Annotated[int, Field(ge=1)] = 1,
        page_size: Annotated[int, Field(ge=1, le=100)] = 50,
    ) -> dict:
        """List registered models, with the IDs that the other tools require."""
        return await _get(client, "/models", page=page, page_size=page_size)

    @server.tool(structured_output=False)
    @_structured
    async def get_model_health(model_id: UUID) -> dict:
        """Get a model's status, severity counts and latest health score.

        Status is no_data, unknown, healthy, attention or critical, taken from the
        severities stored for the latest window. Health score comes from the latest report; check
        report_covers_latest_window because that report can predate the latest window.
        """
        health = await _health(client, model_id)
        del health["features"], health["prediction"]
        return health

    @server.tool(structured_output=False)
    @_structured
    async def get_feature_drift(
        model_id: UUID,
        window_date: date | None = None,
        feature_name: Annotated[str, Field(min_length=1)] | None = None,
    ) -> dict:
        """Get stored PSI, JS divergence, KS statistic and severity by feature.

        Uses the latest window unless window_date is given. ks_pvalue is null because
        histograms cannot support a sample p-value. Prediction drift is returned
        separately and has zero feature weight.
        """
        window, features, prediction = await _feature_drift(
            client, model_id, window_date
        )
        if feature_name is not None:
            features = [r for r in features if r["feature_name"] == feature_name]
            if not features:
                raise ApiError(
                    "not_found", f"No scores for {feature_name} in window {window}"
                )
        return {
            "model_id": str(model_id),
            "window_date": window,
            "features": features,
            "prediction": prediction,
        }

    @server.tool(structured_output=False)
    @_structured
    async def get_feature_distribution(
        model_id: UUID,
        feature_name: Annotated[str, Field(min_length=1)],
        window_date: date | None = None,
    ) -> dict:
        """Get stored reference and snapshot stats for a feature."""
        reference = await _get(client, f"/ingest/reference/{model_id}")
        if window_date is None:
            dates = await _get(client, f"/ingest/snapshots/{model_id}")
            if not dates:
                raise ApiError("not_found", "No snapshot windows found")
            window = dates[0]
        else:
            window = window_date.isoformat()
        snapshots = await _get(client, f"/ingest/snapshots/{model_id}/{window}")
        ref = next(
            (r for r in reference["items"] if r["feature_name"] == feature_name), None
        )
        snapshot = next(
            (r for r in snapshots if r["feature_name"] == feature_name), None
        )
        if ref is None or snapshot is None:
            raise ApiError(
                "not_found", f"No distribution for {feature_name} in window {window}"
            )
        return {
            "model_id": str(model_id),
            "feature_name": feature_name,
            "window_date": window,
            "reference_stats": ref["stats"],
            "window_stats": snapshot["stats"],
        }

    @server.tool(structured_output=False)
    @_structured
    async def get_active_alerts(model_id: UUID) -> dict:
        """Get every unresolved alert for a model. Unknown model IDs are an error."""
        await _get(client, f"/models/{model_id}")
        alerts = await _active_alerts(client, model_id)
        return {"model_id": str(model_id), "alerts": alerts, "total": len(alerts)}

    @server.tool(structured_output=False)
    @_structured
    async def get_health_report(model_id: UUID, week_start: date | None = None) -> dict:
        """Get a stored weekly health report, by default the latest one."""
        suffix = week_start.isoformat() if week_start else "latest"
        return await _get(client, f"/reports/{model_id}/{suffix}")

    @server.tool(structured_output=False)
    @_structured
    async def analyze_model_health(model_id: UUID) -> dict:
        """Investigate a model from stored evidence, computing no new statistics.

        Returns status, non-green features ranked by importance-weighted score, stable
        features, prediction drift, active alerts and the limits of this evidence.
        """
        health = await _health(client, model_id)
        features = health.pop("features")
        problems = [r for r in features if r["severity"] != "green"]
        # Unscored rows rank last rather than breaking the sort.
        problems.sort(key=lambda r: r["weighted_score"] or 0.0, reverse=True)
        return {
            **health,
            "problem_features": problems,
            "stable_features": [
                r["feature_name"] for r in features if r["severity"] == "green"
            ],
            "active_alerts": await _active_alerts(client, model_id),
            "limitations": LIMITATION,
        }

    return server


def main() -> None:
    base_url = os.environ.get("DRIFTWATCH_URL", "http://localhost:8000")
    build_server(httpx.AsyncClient(base_url=base_url, timeout=30)).run(
        transport="stdio"
    )


if __name__ == "__main__":
    main()
