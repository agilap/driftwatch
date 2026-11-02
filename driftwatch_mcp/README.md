# DriftWatch MCP server

Optional MCP server that lets clients query a running DriftWatch service. It calls the
REST API and computes no statistics, so its values always match the service.

```bash
pip install -r requirements-mcp.txt
DRIFTWATCH_URL=http://localhost:8000 python -m driftwatch_mcp
```

The server speaks MCP over stdio. A client configuration, run from the repository root:

```json
{
  "mcpServers": {
    "driftwatch": {
      "command": "python",
      "args": ["-m", "driftwatch_mcp"],
      "env": {"DRIFTWATCH_URL": "http://localhost:8000"}
    }
  }
}
```

## Why this lives in the main repository

The earlier `driftwatch-mcp` repository duplicated the statistics and had already
diverged from the service. An optional package here shares one checkout, one CI run and
one API contract, and its dependencies stay out of the API image. The earlier repository
and its history are untouched. Its four raw-sample tools were not carried over, because
they recomputed statistics instead of reading monitoring results.

## Tools

| Tool | Purpose |
|---|---|
| `list_models` | Discover registered models and their IDs |
| `get_model_health` | Status, severity counts and latest health score |
| `get_feature_drift` | Stored metrics by feature, for the latest or a given window |
| `get_feature_distribution` | Stored reference and window statistics for a feature |
| `get_active_alerts` | Every unresolved alert |
| `get_health_report` | A stored weekly report, by default the latest |
| `analyze_model_health` | Ranked problem features, prediction drift, alerts and limits |

Failures return `isError: true` with `{"error": {"code", "message", "status"}}`. The
code is `not_found`, `api_error` or `service_unavailable`. Malformed arguments, such as
an invalid model ID, are rejected by input validation before any API request.
