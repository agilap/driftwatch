"""MCP protocol tests through a real client/server session, with a stubbed REST API."""

import json
import os
import sys
from uuid import uuid4

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from driftwatch_mcp.server import build_server

MODEL_ID = str(uuid4())
TOOLS = {
    "list_models",
    "get_model_health",
    "get_feature_drift",
    "get_feature_distribution",
    "get_active_alerts",
    "get_health_report",
    "analyze_model_health",
}


def _row(name: str, severity: str, weighted: float = 0.0) -> dict:
    return {
        "feature_name": name,
        "severity": severity,
        "weighted_score": weighted,
        "psi": 0.5,
        "ks_pvalue": None,
    }


def _api(routes: dict[str, tuple[int, object]], calls: list | None = None):
    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url)
        status, body = routes.get(request.url.path, (404, {"detail": "Not found"}))
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    )


def _monitored(rows: list[dict]) -> dict:
    return {
        f"/models/{MODEL_ID}": (200, {"id": MODEL_ID, "name": "m"}),
        f"/drift/{MODEL_ID}/latest": (200, {"window_date": "2026-01-07"}),
        f"/drift/{MODEL_ID}/2026-01-07": (200, rows),
        f"/alerts/{MODEL_ID}": (200, {"items": [], "total": 0}),
    }


async def _call(client: httpx.AsyncClient, tool: str, **arguments):
    server = build_server(client)
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        result = await session.call_tool(tool, arguments)
    return result, result.structuredContent


@pytest.mark.asyncio
async def test_tools_are_discoverable_with_validated_ids():
    server = build_server(_api({}))
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
    assert set(tools) == TOOLS
    assert all(tool.description for tool in tools.values())
    assert tools["get_model_health"].inputSchema["properties"]["model_id"] == {
        "format": "uuid",
        "title": "Model Id",
        "type": "string",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(TOOLS - {"list_models"}))
async def test_invalid_model_id_is_rejected_before_any_request(tool):
    calls = []
    args = {"feature_name": "a"} if tool == "get_feature_distribution" else {}
    result, _ = await _call(_api({}, calls), tool, model_id="not-a-uuid", **args)
    assert result.isError
    assert "model_id" in result.content[0].text
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", sorted(TOOLS - {"list_models"}))
async def test_unknown_model_is_a_coded_error(tool):
    args = {"feature_name": "a"} if tool == "get_feature_distribution" else {}
    result, payload = await _call(_api({}), tool, model_id=str(uuid4()), **args)
    assert result.isError
    assert payload["error"]["code"] == "not_found"
    assert payload["error"]["status"] == 404
    assert json.loads(result.content[0].text) == payload


@pytest.mark.asyncio
async def test_unavailable_service_is_a_coded_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(refuse), base_url="http://test"
    )
    result, payload = await _call(client, "list_models")
    assert result.isError
    assert payload["error"]["code"] == "service_unavailable"
    assert "connection refused" in payload["error"]["message"]


@pytest.mark.asyncio
async def test_server_error_is_not_reported_as_no_data():
    routes = _monitored([])
    routes[f"/reports/{MODEL_ID}/latest"] = (500, {"detail": "boom"})
    result, payload = await _call(
        _api(routes), "analyze_model_health", model_id=MODEL_ID
    )
    assert result.isError
    assert payload["error"] == {"code": "api_error", "message": "boom", "status": 500}


@pytest.mark.asyncio
async def test_page_size_bound_is_validated():
    calls = []
    result, _ = await _call(_api({}, calls), "list_models", page_size=101)
    assert result.isError
    assert calls == []


@pytest.mark.asyncio
async def test_model_without_scores_has_no_data_rather_than_healthy():
    routes = {f"/models/{MODEL_ID}": (200, {"id": MODEL_ID, "name": "m"})}
    result, payload = await _call(_api(routes), "get_model_health", model_id=MODEL_ID)
    assert not result.isError
    assert payload["status"] == "no_data"
    assert payload["overall_score"] is None
    assert payload["report_covers_latest_window"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "severities, expected",
    [
        (["green", "green"], "healthy"),
        (["green", "yellow"], "attention"),
        (["yellow", "red"], "critical"),
        (["green", None], "unknown"),
        ([None, "red"], "critical"),
    ],
)
async def test_status_follows_stored_severities(severities, expected):
    rows = [_row(f"f{i}", severity) for i, severity in enumerate(severities)]
    _, payload = await _call(
        _api(_monitored(rows)), "get_model_health", model_id=MODEL_ID
    )
    assert payload["status"] == expected
    assert "features" not in payload


@pytest.mark.asyncio
async def test_prediction_drift_alone_makes_status_critical_but_is_not_a_feature():
    rows = [_row("a", "green"), _row("__predictions__", "red")]
    _, payload = await _call(
        _api(_monitored(rows)), "analyze_model_health", model_id=MODEL_ID
    )
    assert payload["status"] == "critical"
    assert payload["severity_counts"] == {"red": 0, "yellow": 0, "green": 1}
    assert payload["prediction_severity"] == "red"
    assert payload["problem_features"] == []
    assert payload["stable_features"] == ["a"]


@pytest.mark.asyncio
async def test_analysis_ranks_problems_by_weighted_score_not_psi():
    rows = [
        _row("low", "red", 0.1),
        _row("high", "yellow", 0.4),
        _row("stable", "green", 0.0),
    ]
    _, payload = await _call(
        _api(_monitored(rows)), "analyze_model_health", model_id=MODEL_ID
    )
    assert [r["feature_name"] for r in payload["problem_features"]] == ["high", "low"]
    assert "does not establish performance degradation" in payload["limitations"]


@pytest.mark.asyncio
async def test_analysis_ranks_unscored_problem_rows_last():
    rows = [
        _row("scored", "red", 0.1),
        {**_row("unscored", None), "weighted_score": None},
    ]
    result, payload = await _call(
        _api(_monitored(rows)), "analyze_model_health", model_id=MODEL_ID
    )
    assert not result.isError
    assert payload["status"] == "critical"
    assert [r["feature_name"] for r in payload["problem_features"]] == [
        "scored",
        "unscored",
    ]


@pytest.mark.asyncio
async def test_feature_filter_and_explicit_window():
    rows = [_row("a", "red"), _row("b", "green")]
    routes = {f"/drift/{MODEL_ID}/2026-01-07": (200, rows)}
    client = _api(routes)
    _, payload = await _call(
        client,
        "get_feature_drift",
        model_id=MODEL_ID,
        window_date="2026-01-07",
        feature_name="a",
    )
    assert [r["feature_name"] for r in payload["features"]] == ["a"]
    result, missing = await _call(
        client,
        "get_feature_drift",
        model_id=MODEL_ID,
        window_date="2026-01-07",
        feature_name="absent",
    )
    assert result.isError
    assert missing["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_feature_distribution_returns_stored_stats_and_latest_window():
    routes = {
        f"/ingest/reference/{MODEL_ID}": (
            200,
            {"items": [{"feature_name": "a", "stats": {"mean": 4}}]},
        ),
        f"/ingest/snapshots/{MODEL_ID}": (200, ["2026-01-07", "2026-01-01"]),
        f"/ingest/snapshots/{MODEL_ID}/2026-01-07": (
            200,
            [
                {
                    "feature_name": "a",
                    "stats": {"mean": 3},
                    "distribution": {
                        "bin_edges": [0.0, 1.0],
                        "counts": [1],
                        "underflow": 0,
                        "overflow": 0,
                    },
                    "sample_count": 1,
                }
            ],
        ),
    }
    client = _api(routes)
    result, payload = await _call(
        client, "get_feature_distribution", model_id=MODEL_ID, feature_name="a"
    )
    assert not result.isError
    assert payload == {
        "model_id": MODEL_ID,
        "feature_name": "a",
        "window_date": "2026-01-07",
        "reference_stats": {"mean": 4},
        "window_stats": {"mean": 3},
    }
    result, payload = await _call(
        client,
        "get_feature_distribution",
        model_id=MODEL_ID,
        feature_name="missing",
        window_date="2026-01-07",
    )
    assert result.isError
    assert payload["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_feature_distribution_requires_both_stats_and_a_window():
    routes = {
        f"/ingest/reference/{MODEL_ID}": (
            200,
            {"items": [{"feature_name": "a", "stats": {"mean": 4}}]},
        ),
        f"/ingest/snapshots/{MODEL_ID}": (200, []),
        f"/ingest/snapshots/{MODEL_ID}/2026-01-07": (200, []),
    }
    client = _api(routes)
    for args in ({}, {"window_date": "2026-01-07"}):
        result, payload = await _call(
            client,
            "get_feature_distribution",
            model_id=MODEL_ID,
            feature_name="a",
            **args,
        )
        assert result.isError
        assert payload["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_active_alerts_reads_every_page():
    pages = {"1": [{"id": str(i)} for i in range(100)], "2": [{"id": "last"}]}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/models/{MODEL_ID}":
            return httpx.Response(200, json={"id": MODEL_ID})
        assert request.url.params["resolved"] == "false"
        items = pages[request.url.params["page"]]
        return httpx.Response(200, json={"items": items, "total": len(items)})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    )
    _, payload = await _call(client, "get_active_alerts", model_id=MODEL_ID)
    assert payload["total"] == 101
    assert payload["alerts"][-1] == {"id": "last"}


@pytest.mark.asyncio
async def test_stdio_process_starts_and_reports_an_unreachable_api():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "driftwatch_mcp"],
        env={**os.environ, "DRIFTWATCH_URL": "http://127.0.0.1:9"},
    )
    async with stdio_client(parameters) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            assert {t.name for t in (await session.list_tools()).tools} == TOOLS
            result = await session.call_tool("list_models", {})
    assert result.isError
    assert result.structuredContent["error"]["code"] == "service_unavailable"
