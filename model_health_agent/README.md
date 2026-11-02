# Model health agent

The agent investigates a registered model through the DriftWatch MCP server. It reads
stored monitoring results and makes no statistical calculations. This package lives
in the main repository because the earlier agent read fixture data and required one
provider; this version uses the service and supports optional provider-neutral narration.

## Run the demo

From the repository root, start DriftWatch, install the dependencies, and run the demo:

```bash
docker compose up -d
python -m pip install -r requirements-demo.txt -r requirements-agent.txt
python -m examples.loan_risk_demo.run_demo
```

Find the demo model's ID in the `/models` response, then pass it to the agent:

```bash
curl -s http://localhost:8000/models
python -m model_health_agent <model_id>
python -m model_health_agent <model_id> --json
```

Set `DRIFTWATCH_URL` if the service is not at `http://localhost:8000`.

## Modes and routing

Offline mode requires no API key and produces a deterministic report. Optional LLM
narration is enabled only when all three variables are set:

```bash
export DRIFTWATCH_LLM_BASE_URL=https://your-provider.example/v1
export DRIFTWATCH_LLM_MODEL=your-model
export DRIFTWATCH_LLM_API_KEY=your-api-key
```

The endpoint must support OpenAI-compatible `/chat/completions`. Narration explains
the structured report; it does not choose the route or change the evidence. If the
provider fails or narration introduces an unsupported number, the agent returns the
offline report. The numeric guard checks numbers, not meaning, so review narration
against the evidence.

Routing uses the collected status: `critical` and `attention` investigate the highest
ranked features, while `healthy`, `no_data`, and `unknown` go directly to the summary.
No data means no conclusion can be drawn. Drift does not establish performance
degradation; evaluate recent labeled data to assess performance.
