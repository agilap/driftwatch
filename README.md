# DriftWatch

[![CI](https://github.com/agilap/driftwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/agilap/driftwatch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

DriftWatch is a portfolio project for examining changes in numeric model inputs and predictions without requiring labels. It is not deployed, and its loan-risk demonstration uses synthetic data. It stores reference and window histograms, computes drift metrics, raises threshold-based alerts, and produces weekly health reports. The [project page](https://driftwatch-landing.vercel.app) summarizes the demo results; the application itself runs locally.

## Capabilities

- Register models, reference samples, prediction samples, and feature importances.
- Ingest dated windows and analyze feature and prediction distributions.
- Retrieve drift scores, resolve alerts, and generate or retrieve weekly reports.
- Query the API through an optional MCP server and investigate results with an optional model health agent.

## Architecture

```mermaid
flowchart LR
    Client[API client] --> API[FastAPI]
    API --> DB[(PostgreSQL)]
    API --> Analysis[Background drift analysis]
    Analysis --> DB
    Analysis --> Webhook[Optional webhook]
    Scheduler[In-process scheduler] --> Reports[Report generator]
    Reports --> DB
    Agent[Model health agent] --> MCP[MCP server]
    MCP --> API
```

```mermaid
sequenceDiagram
    participant C as Client
    participant A as API
    participant D as PostgreSQL
    participant B as Background analysis
    C->>A: Register reference samples
    A->>D: Store reference histograms
    C->>A: Ingest snapshot
    A->>D: Store snapshot histograms
    A-->>C: Snapshot accepted
    A->>B: Start analysis
    B->>D: Read shared bins and store scores and alerts
    C->>A: Generate weekly report
    A->>D: Read scores and store report
    A-->>C: Return report
```

Analysis runs in process after the snapshot response. Wait for analysis to finish before generating a report, or call the analysis endpoint explicitly. Reports select the highest PSI for each feature in the requested seven-day period and score those features. Reports list days with snapshots but no scores in `unscored_windows`.

## Technology stack

Python, FastAPI, Pydantic, SQLAlchemy with asyncpg, PostgreSQL, Alembic, NumPy, APScheduler, and Docker Compose form the service. The demo uses scikit-learn; the optional MCP server uses MCP, and the model health agent provides automated analysis.

## Quick start

Install Docker with Compose, then from a fresh clone:

```bash
git clone https://github.com/agilap/driftwatch.git
cd driftwatch
POSTGRES_PORT=5433 API_PORT=8000 docker compose up -d --build
curl -fsS http://localhost:8000/ready
```

To start the stack, run the loan-risk demo and then the offline agent in one step, install the demo and agent dependencies and use `make demo`. It uses `.venv` if present, otherwise `python3`; `POSTGRES_PORT` and `API_PORT` default to 5433 and 8000 and can be overridden.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-demo.txt -r requirements-agent.txt
make demo
```

Otherwise, follow the manual path above and the sections below. Compose applies migrations before starting the API. `/health` checks liveness; `/ready` checks database and schema readiness. Interactive API documentation is at http://localhost:8000/docs. Compose supplies its own defaults, so copying `.env.example` is not required. `POSTGRES_PORT` and `API_PORT` in `docker-compose.yml` select host ports; choose free ports if 5433 or 8000 is taken. If changing `API_PORT`, use that port in the examples below. Stop the services with `docker compose down`; the database volume remains.

## API usage

These examples use small arrays to show the payload shape, not reliable drift estimates. Run them in order on a fresh stack. Python extracts the returned UUID; model names must be unique.

```bash
MODEL_ID=$(curl -fsS -X POST http://localhost:8000/models \
  -H 'Content-Type: application/json' \
  -d '{"name":"example-model","version":"1"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -fsS -X POST http://localhost:8000/ingest/reference \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\":\"$MODEL_ID\",\"features\":{\"income\":[10,20,30,40]},\"predictions\":[0.1,0.2,0.3,0.4],\"feature_importances\":{\"income\":1},\"importance_method\":\"manual\"}"

curl -fsS -X POST http://localhost:8000/ingest/snapshot \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\":\"$MODEL_ID\",\"timestamp\":\"2025-01-13T00:00:00Z\",\"features\":{\"income\":[20,30,40,50]},\"predictions\":[0.2,0.3,0.4,0.5]}"

curl -fsS -X POST "http://localhost:8000/drift/$MODEL_ID/2025-01-13/analyze"
curl -fsS "http://localhost:8000/drift/$MODEL_ID/2025-01-13"
curl -fsS "http://localhost:8000/alerts/$MODEL_ID"
curl -fsS -X POST "http://localhost:8000/reports/$MODEL_ID/generate" \
  -H 'Content-Type: application/json' -d '{"week_start":"2025-01-13"}'
curl -fsS "http://localhost:8000/reports/$MODEL_ID/latest"
```

The explicit analysis call also retries a failed background analysis. See `/docs` for all routes, including importances, snapshot retrieval, and alert resolution. The [Postman collection](examples/driftwatch.postman_collection.json) runs the whole flow in order: health, model creation (which saves `modelId`), reference, snapshot, analysis, drift, alerts, report generation, and latest report. Model names use a dynamic variable, so it can be rerun; try `npx newman run examples/driftwatch.postman_collection.json --env-var baseUrl=http://localhost:8000`.

### Authentication

Set `DRIFTWATCH_API_KEY` to require that key in an `X-API-Key` header on every route except `/health`, `/ready`, and the API docs; other requests return 401. Without it, authentication is off and startup logs a warning, so the local examples work unchanged. The MCP server, agent, and loan-risk demo send the key when `DRIFTWATCH_API_KEY` is in their environment; `make` and Compose pick up an exported value. In Postman, set the `apiKey` collection variable. Generate a key with `openssl rand -hex 32`.

### Replacement rules

- **Snapshots:** posting to a date replaces that entire window. Batches are not accumulated, so a second same-day post discards the first, including any feature or predictions it omits. Features omitted from a window are not scored, and the weekly report counts only features that were analyzed, normalizing importance over them.
- **Predictions:** a snapshot may include `predictions` only if the reference has them; otherwise it returns 422, like an unknown feature.
- **References:** before any snapshot exists, re-registering replaces the whole reference. If the feature set changes and no importances are supplied, stored importances are deleted with a logged warning, and weighting falls back to equal weight. After snapshots exist, the reference is immutable and a changed payload returns 409.
- **Importances:** a set must cover exactly the reference features, excluding predictions; explicit zeros are allowed. A non-matching set returns 422 naming the missing and unknown features, `POST /models/{id}/importances` returns 409 without a reference, and each accepted set replaces the previous one. Changing importances does not rewrite stored scores until the window is re-analyzed; reports generated afterwards use the current importances.
- **Reports:** Changed snapshot data for a day, or a retried analysis that adds scores for a day, deletes any report covering that day. The daily job rebuilds a missing report for the last complete Monday-to-Sunday week; older weeks need `POST /reports/{id}/generate`. An identical re-post keeps the report. Latest skips reports with no scores and no unscored windows.
- **Alerts:** when new data changes a window, all of that window's alerts are superseded: they are marked resolved, keeping any earlier resolution time, and remain listed under `resolved=true`. If the window is still red, analysis raises a new alert with a fresh message and webhook. An identical re-post changes nothing, and a manually resolved alert stays resolved through re-analysis.

Snapshot timestamps more than one day in the future and report `week_start` dates after today return 422. Requests that fail schema validation return 422 with `detail`, `code`, and `errors` (`loc`, `msg`, `type` for each problem); feature names can appear in `loc` and `msg` because they are schema names, but sample values, `input`, and `ctx` are never returned, and logs record only `loc` and `type`. Importance values and samples must be JSON numbers, so booleans and numeric strings are rejected there. Checks against stored state or computed values return 422 with `detail` only: unknown features, importance coverage, and values too extreme to summarize or bin, such as `[-1e308, 1e308]`, all rejected before any row changes.

## Statistical methodology

Reference `bin_edges` are the only boundaries used for both populations: 10 equal-width bins over the reference range, plus underflow and overflow counts. Equal-width bins can represent skewed features poorly; quantile bins are a common alternative.

- **PSI:** uncapped population stability index, with `EPSILON` smoothing and renormalization. With default thresholds, green is below 0.10, yellow is 0.10 through 0.20 inclusive, and red is above 0.20. These are operational heuristics, not significance tests.
- **JS divergence:** base 2, computed from histogram proportions with NumPy, with range 0 to 1.
- **KS:** `ks_statistic` is the maximum CDF difference over shared bins, a lower bound on the sample KS statistic. `ks_pvalue` is always null for histogram analysis because histograms cannot support a valid p-value. Raw samples are not retained, trading an exact test for reduced storage and privacy exposure.
- **Weighting:** `weighted_score = normalized importance * min(PSI / PSI_RED_THRESHOLD, 1)`. Importance is normalized over analyzed features; if every analyzed feature with a stored importance is zero, all their weights are zero, and equal weight applies only when none has a stored importance. Health is `100 * (1 - sum(weighted_score))`; no feature evidence produces a null score. Predictions are scored and can alert as `output_shift`, but have zero weight in health.

Sampling noise matters. `python3 -m examples.psi_noise` draws both samples from a standard normal distribution, with a fresh 50,000-row reference and 10 bins, over 500 seeded trials per window size:

| Window size | Mean PSI | Trials red |
|---:|---:|---:|
| 50 | 0.1975 | 37.8% |
| 100 | 0.0955 | 5.4% |
| 500 | 0.0178 | 0.0% |
| 2,000 | 0.0046 | 0.0% |
| 10,000 | 0.0011 | 0.0% |

At 50 rows the mean was just below the red threshold, and more than a third of trials were red. Small batches can alert from noise alone. `MIN_SAMPLE_WARNING` defaults to 200 and logs a warning; it does not suppress alerts.

Data drift changes input distributions. Prediction drift changes output distributions. Concept drift changes the relationship between inputs and outcomes. Measured performance compares predictions with labels. DriftWatch measures only the first two, not concept drift or model performance.

## Loan-risk demo

With the stack running, create a virtual environment (Python 3.10 or newer), install the demo dependencies and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-demo.txt
python3 -m examples.loan_risk_demo.run_demo
```

Set `DRIFTWATCH_URL` if the API uses a different address. The seeded demo trains a logistic regression model on synthetic data, registers a reference, and evaluates stable and shifted batches. See the [demo README](examples/loan_risk_demo/README.md) for the data and label rule. Example output from a fresh stack:

```text
Held-out model AUC: 0.8074
Scenario A (stable): batch AUC 0.8059, health 99.43
  credit_score 0.0011 green; annual_income 0.0017 green; debt_to_income 0.0008 green;
  employment_years 0.0011 green; open_credit_accounts 0.0016 green; predictions 0.0023 green
Scenario B (shifted): batch AUC 0.8367, health 16.09
  credit_score 0.9528 red; annual_income 1.6610 red; debt_to_income 2.7182 red;
  employment_years 0.0017 green; open_credit_accounts 0.0011 green; predictions 4.4185 red
Alerts in B: 3 drift_red plus 1 output_shift
```

Batch AUC rose while health fell. The label rule is unchanged, so this demo introduces no concept drift; the cause of the higher AUC was not investigated. This illustrates that drift and performance are separate measurements; it is not a general finding.

## MCP server

The optional [MCP server](driftwatch_mcp/README.md) speaks stdio and reads the REST API; it computes no statistics. With the API running:

```bash
python3 -m pip install -r requirements-mcp.txt
DRIFTWATCH_URL=http://localhost:8000 python3 -m driftwatch_mcp
```

Configure an MCP client as described in its README. Tools expose models, health, drift, distributions, alerts, reports, and a health analysis.

## Model health agent

The optional [agent](model_health_agent/README.md) reads MCP evidence and routes deterministically. Offline mode needs no LLM credentials. Install dependencies and use a registered model UUID:

```bash
python3 -m pip install -r requirements-agent.txt
python3 -m model_health_agent "$MODEL_ID"
```

Optional OpenAI-compatible narration cannot change the route or evidence; its numeric guard checks numbers, not meaning. This is an excerpt of the recorded agent output for the shifted demo:

### Findings

- debt_to_income: PSI 2.718 (red), weighted 0.347, mean 0.281 reference vs 0.420 window (above), 1 active alert
- credit_score: PSI 0.953 (red), weighted 0.310, mean 699.327 reference vs 649.242 window (below), 1 active alert
- annual_income: PSI 1.661 (red), weighted 0.181, mean 69,828.852 reference vs 50,127.600 window (below), 1 active alert
- Model predictions: PSI 4.419 (red), weighted 0.000, mean 0.410 reference vs 0.889 window (above), 1 active alert

### Recommendations

- Check the upstream data source for debt_to_income; the window mean is above the reference.
- Check the upstream data source for credit_score; the window mean is below the reference.
- Check the upstream data source for annual_income; the window mean is below the reference.
- Evaluate performance on recent labeled data before deciding whether to retrain.
- Review downstream decisions that use the score distribution; the window mean is above the reference.

## Testing

The test suite requires PostgreSQL and a `TEST_DATABASE_URL` pointing to a dedicated database. Tests truncate tables, so pytest exits before running anything if the variable is unset or its database name does not end in `_test`. To use a separate database in the Compose PostgreSQL on host port 5433:

```bash
python3 -m pip install -r requirements-demo.txt -r requirements-agent.txt -r requirements-dev.txt
POSTGRES_PORT=5433 docker compose up -d --wait db
docker compose exec db createdb -U driftwatch driftwatch_test
export TEST_DATABASE_URL=postgresql+asyncpg://driftwatch:driftwatch@localhost:5433/driftwatch_test
pytest tests/ -v
ruff check .
black --check .
```

`make test` requires `requirements-dev.txt` installed in its environment; it starts the database, creates `driftwatch_test` if it is missing, and runs pytest.

The CI workflow runs Ruff, Black, PostgreSQL-backed tests with coverage, and a Compose smoke test including the demo and offline agent.

## Engineering decisions

- Histogram-only KS avoids retaining raw samples, at the cost of an exact statistic and p-value. Shared reference bins make window comparisons meaningful but fix their resolution.
- A reference becomes immutable after snapshots exist; register a new model under a new name for a new baseline.
- In-process background analysis uses model row locks and natural-key constraints instead of a queue. This reduces infrastructure but lacks queue durability.
- Migration `0003` deletes derived rows because old histogram results cannot satisfy the corrected statistical contract. Old snapshots need re-ingestion.
- Superseded alerts stay as audit rows instead of being deleted, so a corrected window keeps its history; the migration `0004` downgrade deletes them because the earlier constraint allows one alert per window.
- MCP and agent are optional packages in this repository, instead of separate repositories; the earlier repositories remain untouched. The MCP server reads the API rather than duplicating statistics. Agent routing is deterministic, and narration cannot change evidence.

## Limitations

- Authentication is one shared API key, off unless `DRIFTWATCH_API_KEY` is set. There are no users, roles, or per-model permissions, and the service does not terminate TLS, so put it behind HTTPS before exposing it.
- Demo data is synthetic. There is no deployment or real-world monitoring claim.
- Only numeric features can be ingested; categorical ingestion is absent.
- There is no distributed queue, automatic crash recovery, or durable webhook outbox. Webhook delivery is best effort; the report scheduler runs in process. If the API is stopped for a whole week, that week’s report needs a manual generate.
- Equal-width bins can obscure skewed distributions, and small windows can produce noisy alerts.
- Alerts are per window and threshold based, not significance tests.
- The alert API does not distinguish superseded alerts from manually resolved ones; both appear under `resolved=true`.
- A feature with zero importance can be red and raise an alert without lowering health.
- The narration guard checks numbers, not meaning. Real LLM providers have not been tested; the OpenAI-compatible path was exercised with a stub provider.

## Project layout

- `app/`: API, schemas, persistence models, and services.
- `migrations/`: Alembic schema and data migrations.
- `scheduler/`: weekly report scheduler.
- `examples/`: synthetic loan-risk demo, PSI noise measurement, and Postman collection.
- `driftwatch_mcp/`, `model_health_agent/`: optional integrations.
- `tests/`: service, integration, demo, MCP, and agent tests.

## License

[MIT](LICENSE).
