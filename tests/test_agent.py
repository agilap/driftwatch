"""Agent tests through an in-process MCP server and a stubbed REST API."""

import json
import os
import subprocess
import sys
from uuid import uuid4

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from driftwatch_mcp.server import build_server
from model_health_agent.__main__ import render_report
from model_health_agent.graph import ToolError, build_graph, mean_direction
from model_health_agent.narration import narrate_report

MODEL_ID = str(uuid4())
WINDOW = "2026-01-07"


def row(name, severity, score=0.0):
    return {
        "feature_name": name,
        "severity": severity,
        "weighted_score": score,
        "psi": 0.9528,
        "ks_pvalue": None,
    }


def api(rows, calls, *, fail_reference=False, missing_mean=False):
    def handle(request):
        path = request.url.path
        calls.append(path)
        routes = {
            f"/models/{MODEL_ID}": {"id": MODEL_ID, "name": "test"},
            f"/drift/{MODEL_ID}/latest": {"window_date": WINDOW},
            f"/drift/{MODEL_ID}/{WINDOW}": rows,
            f"/alerts/{MODEL_ID}": {
                "items": [{"feature_name": "first", "message": "alert"}],
                "total": 1,
            },
            f"/ingest/reference/{MODEL_ID}": {
                "items": [
                    {"feature_name": r["feature_name"], "stats": {"mean": 10}}
                    for r in rows
                ]
            },
            f"/ingest/snapshots/{MODEL_ID}/{WINDOW}": [
                {
                    "feature_name": r["feature_name"],
                    "stats": {} if missing_mean else {"mean": 12 if i % 2 == 0 else 8},
                    "distribution": {
                        "bin_edges": [0.0, 1.0],
                        "counts": [1],
                        "underflow": 0,
                        "overflow": 0,
                    },
                    "sample_count": 1,
                }
                for i, r in enumerate(rows)
            ],
        }
        if fail_reference and path == f"/ingest/reference/{MODEL_ID}":
            return httpx.Response(500, json={"detail": "broken reference"})
        if path not in routes:
            return httpx.Response(404, json={"detail": "Not found"})
        return httpx.Response(200, json=routes[path])

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    )


async def run_agent(
    rows, calls, *, fail_reference=False, repeat=False, missing_mean=False
):
    async with api(
        rows, calls, fail_reference=fail_reference, missing_mean=missing_mean
    ) as client:
        server = build_server(client)
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            graph = build_graph(session)
            state = await graph.ainvoke({"model_id": MODEL_ID})
            if repeat:
                second = await graph.ainvoke({"model_id": MODEL_ID})
                assert state["report"] == second["report"]
    return state["report"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows,status,investigated",
    [
        ([row("a", "red", 0.4)], "critical", True),
        ([row("a", "yellow", 0.4)], "attention", True),
        ([row("a", "green")], "healthy", False),
        ([], "no_data", False),
        ([row("a", None)], "unknown", False),
    ],
)
async def test_status_routes_without_narrator(rows, status, investigated):
    calls = []
    report = await run_agent(rows, calls)
    assert report["status"] == status
    assert bool(report["findings"]) is investigated
    assert any("/reference/" in path for path in calls) is investigated
    if status == "attention":
        assert "healthy" not in json.dumps(report).lower()
    if status == "no_data":
        assert "conclusion" in json.dumps(report).lower()


@pytest.mark.asyncio
async def test_rank_limit_directions_and_determinism():
    rows = [
        row("first", "red", 0.9),
        row("second", "red", 0.8),
        row("third", "yellow", 0.7),
        row("fourth", "red", 0.6),
        row("stable", "green"),
    ]
    calls = []
    report = await run_agent(rows, calls, repeat=True)
    assert [f["feature_name"] for f in report["findings"]] == [
        "first",
        "second",
        "third",
    ]
    assert [f["direction"] for f in report["findings"]] == ["above", "below", "above"]
    assert (
        "Check the upstream data source for first; the window mean is above the reference."
        in report["recommendations"]
    )
    assert report["findings"][0]["reference_mean"] == 10
    assert report["findings"][0]["window_mean"] == 12
    assert report["findings"][0]["alerts"] == [
        {"feature_name": "first", "message": "alert"}
    ]
    assert report["findings"][1]["alerts"] == []
    assert calls.count(f"/ingest/reference/{MODEL_ID}") == 6
    json.dumps(report)


@pytest.mark.asyncio
async def test_tool_failure_aborts_with_code():
    async with api([row("a", "red", 0.4)], [], fail_reference=True) as client:
        server = build_server(client)
        async with create_connected_server_and_client_session(
            server._mcp_server
        ) as session:
            with pytest.raises(ToolError) as exc:
                await build_graph(session).ainvoke({"model_id": MODEL_ID})
            assert exc.value.code == "api_error"
            assert "broken reference" in str(exc.value)


@pytest.fixture
def llm_config(monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("DRIFTWATCH_LLM_MODEL", "example")
    monkeypatch.setenv("DRIFTWATCH_LLM_API_KEY", "secret")


@pytest.mark.asyncio
async def test_narration_unconfigured_makes_no_call(monkeypatch):
    for name in (
        "DRIFTWATCH_LLM_BASE_URL",
        "DRIFTWATCH_LLM_MODEL",
        "DRIFTWATCH_LLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def fail(request):
        raise AssertionError("unexpected network request")

    report = {"mode": "offline", "narrative": None, "evidence": {"score": 0.9528}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        assert await narrate_report(report, client=client) == report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "narrative,accepted", [("The score is 0.953.", True), ("The score is 42.", False)]
)
async def test_narration_checks_numbers_and_preserves_evidence(
    llm_config, narrative, accepted
):
    report = {"mode": "offline", "narrative": None, "evidence": {"score": 0.9528}}
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": narrative}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await narrate_report(report, client=client)
    assert json.dumps(report) in json.dumps(payloads[0]) or any(
        json.loads(m["content"]) == report
        for m in payloads[0]["messages"]
        if m["role"] == "user"
    )
    assert result["mode"] == ("llm" if accepted else "offline")
    assert result["narrative"] == (narrative if accepted else None)
    assert {
        k: v
        for k, v in result.items()
        if k not in {"mode", "narrative", "narrative_rejected"}
    } == {k: v for k, v in report.items() if k not in {"mode", "narrative"}}
    if not accepted:
        assert result["narrative_rejected"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,body", [(500, {"error": "bad"}), (200, {"unexpected": True})]
)
async def test_narration_provider_failures_fall_back(llm_config, status, body):
    report = {"mode": "offline", "narrative": None, "evidence": {"score": 0.9528}}
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body))
    ) as client:
        assert await narrate_report(report, client=client) == report


@pytest.mark.parametrize(
    "model_id,url,expected",
    [
        ("not-a-uuid", "http://localhost:8000", "model"),
        (MODEL_ID, "http://127.0.0.1:9", "DriftWatch API request failed"),
    ],
)
def test_cli_errors_are_clear(model_id, url, expected):
    env = {**os.environ, "DRIFTWATCH_URL": url}
    result = subprocess.run(
        [sys.executable, "-m", "model_health_agent", model_id],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert expected.lower() in result.stderr.lower()
    assert "Traceback" not in result.stderr


@pytest.mark.asyncio
async def test_prediction_drift_is_investigated_after_ranked_features():
    rows = [
        row("feature", "red", 0.4),
        row("__predictions__", "yellow"),
    ]
    calls = []
    report = await run_agent(rows, calls)
    assert [finding["feature_name"] for finding in report["findings"]] == [
        "feature",
        "__predictions__",
    ]
    assert report["findings"][1]["drift"]["severity"] == "yellow"
    assert (
        "Review downstream decisions that use the score distribution; the window mean is below the reference."
        in report["recommendations"]
    )
    assert not any(
        "upstream data source for __predictions__" in item
        for item in report["recommendations"]
    )
    assert calls.count(f"/ingest/reference/{MODEL_ID}") == 2


@pytest.mark.asyncio
async def test_narration_invalid_json_falls_back(llm_config):
    report = {"mode": "offline", "narrative": None, "evidence": {"score": 0.9528}}
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="not json")
    )
    async with httpx.AsyncClient(transport=transport) as client:
        assert await narrate_report(report, client=client) == report


def test_render_report():
    report = {
        "model": {"name": "demo"},
        "status": "critical",
        "evidence": {
            "window": WINDOW,
            "severity_counts": {"red": 1, "yellow": 0, "green": 2},
            "prediction_severity": "red",
            "overall_score": 16.092416981516312,
            "report_covers_latest_window": False,
        },
        "findings": [
            {
                "feature_name": name,
                "drift": {"psi": 1.6608, "severity": "red", "weighted_score": 0.1806},
                "reference_mean": 70012.3,
                "window_mean": 50011.2,
                "direction": "below",
                "alerts": [{"message": "Drift detected"}],
            }
            for name in ("annual_income", "__predictions__")
        ],
        "stable_features": [],
        "active_alerts": [{"message": "Drift detected"}],
        "recommendations": [],
        "limitations": "Drift does not measure model performance.",
        "narrative": None,
    }
    output = render_report(report)
    assert "- Window: 2026-01-07" in output
    assert "- Severity counts: red: 1, yellow: 0, green: 2" in output
    assert "- Prediction severity: red" in output
    assert "- Overall score: 16.1" in output
    assert "- Report covers window: no" in output
    assert (
        "- annual_income: PSI 1.661 (red), weighted 0.181, mean 70,012.300 reference vs 50,011.200 window (below), 1 active alert"
        in output
    )
    assert "- Model predictions: PSI 1.661" in output
    assert "- Drift detected" in output
    assert "Drift does not measure model performance." in output
    assert "- D\n" not in output
    assert '"message"' not in output

    report["evidence"]["overall_score"] = None
    finding = report["findings"][0]
    finding["drift"].update(psi=None, weighted_score=None)
    finding.update(reference_mean=None, window_mean=None)
    output = render_report(report)
    assert "- Overall score: not available" in output
    assert (
        "PSI not available (red), weighted not available, mean not available reference vs not available window"
        in output
    )


@pytest.mark.asyncio
async def test_prediction_recommendation_without_mean():
    report = await run_agent([row("__predictions__", "yellow")], [], missing_mean=True)
    assert (
        "Review downstream decisions that use the score distribution; the window mean direction is unavailable."
        in report["recommendations"]
    )
    assert not any("upstream" in item for item in report["recommendations"])


@pytest.mark.parametrize(
    "direction,expected",
    [
        ("above", "the window mean is above the reference"),
        ("below", "the window mean is below the reference"),
        ("equal", "the window mean equals the reference"),
        (None, "the window mean direction is unavailable"),
    ],
)
def test_mean_direction(direction, expected):
    assert mean_direction(direction) == expected
