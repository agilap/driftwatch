# DriftWatch

![CI](https://github.com/agilap/driftwatch/actions/workflows/ci.yml/badge.svg)

Production model health monitor that detects data drift without requiring labels.

## What it does

DriftWatch tracks whether a deployed model is seeing data that looks different from what it learned during training.

In the loan risk scenario, a model can look healthy in January and silently degrade by August as applicant income and credit profiles shift. DriftWatch catches that shift early by comparing production feature distributions to a known reference baseline and then surfacing high-impact drift first.

## Quick Start

```bash
git clone https://github.com/agilap/driftwatch.git
cd driftwatch
cp .env.example .env
docker-compose up
```

API docs: http://localhost:8000/docs

## Key Concepts

- Reference distribution vs production snapshot:
  Reference data is the baseline (usually training data statistics). Production snapshots are daily (or periodic) batches from live inference traffic. Drift is measured by comparing each snapshot to the reference.
- KS test (Kolmogorov-Smirnov):
  KS checks whether two numeric samples likely came from the same underlying distribution. A low p-value means the production feature shape/location changed significantly.
- PSI (Population Stability Index):
  PSI compares binned proportions between reference and production. It gives an intuitive shift magnitude where higher values indicate stronger population movement.
- JS divergence (Jensen-Shannon):
  JS divergence is a symmetric distance between two probability distributions. It is bounded and stable for both continuous-binned and categorical-style comparisons.
- Weighted drift scoring:
  Not all drift should page the team. DriftWatch multiplies drift magnitude by feature importance so low-business-impact features do not drown out critical ones.
- Health score formula:
  Overall weekly health is computed as:
  `overall_health = 100 × (1 - mean(weighted_scores))`

## API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness and version check |
| `POST` | `/models` | Register a monitored model |
| `GET` | `/models` | List monitored models |
| `GET` | `/models/{model_id}` | Fetch one model |
| `POST` | `/models/{model_id}/importances` | Upload feature importances |
| `POST` | `/ingest/reference` | Register reference distribution |
| `GET` | `/ingest/reference/{model_id}` | List reference features |
| `POST` | `/ingest/snapshot` | Ingest production snapshot |
| `GET` | `/ingest/snapshots/{model_id}` | List snapshot dates |
| `GET` | `/ingest/snapshots/{model_id}/{window_date}` | Get snapshot stats for one date |
| `GET` | `/drift/{model_id}/{window_date}` | Get drift scores for one date |
| `GET` | `/drift/{model_id}/latest` | Get latest drift summary |
| `GET` | `/alerts` | List active alerts (`model_id` optional) |
| `GET` | `/alerts/{model_id}` | List alerts for one model |
| `PATCH` | `/alerts/{alert_id}/resolve` | Resolve alert |
| `GET` | `/reports/{model_id}` | List reports for a model |
| `GET` | `/reports/{model_id}/latest` | Get latest weekly report |
| `GET` | `/reports/{model_id}/{week_start}` | Get report for a week |
| `POST` | `/reports/{model_id}/generate` | Manually generate report |

## Configuration

Environment variables (from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch` | Async SQLAlchemy connection string |
| `SECRET_KEY` | `changeme-in-production` | App secret for production hardening |
| `ALERT_WEBHOOK_URL` | empty | Optional webhook for alert delivery |
| `REPORT_SCHEDULE_CRON` | `0 0 * * 1` | Weekly report cron (UTC) |
| `LOG_LEVEL` | `INFO` | Global log verbosity |
| `PSI_YELLOW_THRESHOLD` | `0.10` | PSI threshold for yellow severity |
| `PSI_RED_THRESHOLD` | `0.20` | PSI threshold for red severity |
| `KS_PVALUE_THRESHOLD` | `0.05` | KS significance threshold |
| `MIN_SAMPLE_WARNING` | `30` | Warning threshold for low sample size |

## Running Tests

```bash
docker-compose run api pytest -v
```

## Architecture

See [architecture.md](architecture.md) for component and data-flow details.

## Why This Matters

A model can still return predictions while quietly degrading.

In the loan scoring framing: performance that looked like 94% in January can collapse to 71% by August when the input population shifts. DriftWatch is designed to detect this shift early, prioritize critical features, and provide actionable weekly health summaries before business KPIs are materially damaged.
