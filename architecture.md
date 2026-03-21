# DriftWatch — System Architecture

> Production Model Health Monitor · Detects Data Drift Without Ground Truth Labels

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DriftWatch Platform                         │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Ingest API  │───▶│  Drift Engine│───▶│  Health Report Gen   │  │
│  │  (FastAPI)   │    │  (scipy)     │    │  (Weekly Scheduler)  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                  │                        │              │
│         ▼                  ▼                        ▼              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     PostgreSQL Database                       │  │
│  │  snapshots │ drift_scores │ feature_importance │ alerts       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────┐    ┌───────────────────────────────────┐ │
│  │  ML Drift Importance │    │  Alert & Notification Layer       │ │
│  │  Scorer (weighted)   │    │  (Webhook / Email / Slack stub)   │ │
│  └──────────────────────┘    └───────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Breakdown

### 2.1 Ingest API (FastAPI)

**Responsibility:** Receive feature snapshots from production inference systems.

| Endpoint | Method | Description |
|---|---|---|
| `/ingest/snapshot` | POST | Accept a batch of feature vectors + predictions |
| `/ingest/reference` | POST | Register the training/reference distribution |
| `/models` | GET/POST | Register and list monitored models |
| `/health` | GET | Liveness probe |

**Payload shape (snapshot):**
```json
{
  "model_id": "loan-scorer-v3",
  "timestamp": "2025-08-01T00:00:00Z",
  "features": {
    "income": [52000, 47000, 61000],
    "credit_score": [710, 680, 745],
    "loan_term_months": [36, 60, 24]
  },
  "predictions": [0.82, 0.61, 0.91]
}
```

---

### 2.2 Drift Engine (scipy)

**Responsibility:** Run statistical tests per feature per time window.

#### Tests Implemented

| Test | Use Case | Detects |
|---|---|---|
| **KS Test** (Kolmogorov-Smirnov) | Continuous features | Shape/location shift |
| **PSI** (Population Stability Index) | Binned distributions | Magnitude of population shift |
| **Jensen-Shannon Divergence** | Both continuous & categorical | Symmetric distributional distance |
| **Chi-Square Test** | Categorical features | Frequency distribution shift |

#### PSI Severity Thresholds
```
PSI < 0.10  → No significant drift   (GREEN)
PSI 0.10–0.20 → Moderate drift       (YELLOW)
PSI > 0.20  → Significant drift      (RED)
```

#### KS Test Threshold
```
p-value < 0.05 → Statistically significant drift detected
```

---

### 2.3 ML Drift Importance Scorer

**Responsibility:** Weight raw drift scores by each feature's importance to the original model, so not all drift triggers equal priority.

**Formula:**
```
weighted_drift_score(f) = drift_magnitude(f) × feature_importance(f)
```

Where:
- `drift_magnitude` = normalised PSI or JS divergence (0–1)
- `feature_importance` = SHAP value or coefficient magnitude from original model, normalised

**Importance Input Methods:**
1. Direct upload of feature importances JSON at model registration
2. SHAP value file (`.json` or `.npy`)
3. Linear coefficient array (for logistic/linear models)

---

### 2.4 Weekly Health Report Generator

**Responsibility:** Aggregate drift signals into a human-readable weekly summary.

**Report Sections:**
1. **Executive Summary** — overall model health score (0–100)
2. **Feature Drift Table** — per-feature KS stat, PSI, JS divergence, weighted score
3. **Output Distribution Shift** — prediction score histogram this week vs. reference
4. **Top Drifted Features** — ranked by weighted drift score
5. **Correlation Analysis** — Pearson/Spearman between feature drift and output shift
6. **Recommendations** — threshold breach alerts, retraining suggestions

**Output Formats:** JSON (API), Markdown, HTML email

---

### 2.5 PostgreSQL Schema

```sql
-- Monitored models registry
CREATE TABLE models (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT NOT NULL UNIQUE,
  version       TEXT,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Reference (training) distributions per model
CREATE TABLE reference_distributions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id      UUID REFERENCES models(id),
  feature_name  TEXT NOT NULL,
  distribution  JSONB NOT NULL,   -- histogram bins + counts
  stats         JSONB NOT NULL,   -- mean, std, min, max, p25/50/75
  registered_at TIMESTAMPTZ DEFAULT NOW()
);

-- Feature importance weights
CREATE TABLE feature_importances (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id      UUID REFERENCES models(id),
  feature_name  TEXT NOT NULL,
  importance    FLOAT NOT NULL,
  method        TEXT,             -- 'shap', 'coefficient', 'manual'
  registered_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily production snapshots
CREATE TABLE snapshots (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id      UUID REFERENCES models(id),
  window_date   DATE NOT NULL,
  feature_name  TEXT NOT NULL,
  distribution  JSONB NOT NULL,
  stats         JSONB NOT NULL,
  sample_count  INT,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Computed drift scores per feature per window
CREATE TABLE drift_scores (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id        UUID REFERENCES models(id),
  window_date     DATE NOT NULL,
  feature_name    TEXT NOT NULL,
  ks_statistic    FLOAT,
  ks_pvalue       FLOAT,
  psi             FLOAT,
  js_divergence   FLOAT,
  weighted_score  FLOAT,
  severity        TEXT CHECK (severity IN ('green','yellow','red')),
  computed_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Weekly health reports
CREATE TABLE health_reports (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id        UUID REFERENCES models(id),
  week_start      DATE NOT NULL,
  week_end        DATE NOT NULL,
  overall_score   FLOAT,
  report_json     JSONB,
  report_markdown TEXT,
  generated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Active alerts
CREATE TABLE alerts (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id      UUID REFERENCES models(id),
  feature_name  TEXT,
  alert_type    TEXT,     -- 'drift_red', 'output_shift', 'data_gap'
  severity      TEXT,
  message       TEXT,
  resolved_at   TIMESTAMPTZ,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 3. Data Flow

```
[Production System]
        │
        │  POST /ingest/snapshot  (daily batch)
        ▼
[FastAPI Ingest Layer]
        │
        ├──▶ Validate payload schema
        ├──▶ Compute distribution stats (mean, std, histogram)
        ├──▶ Store → snapshots table
        │
        ▼
[Drift Engine — triggered on ingest]
        │
        ├──▶ Fetch reference distribution for model
        ├──▶ Run KS test per feature
        ├──▶ Compute PSI per feature
        ├──▶ Compute JS divergence per feature
        ├──▶ Load feature importances → compute weighted_score
        ├──▶ Classify severity (green/yellow/red)
        ├──▶ Store → drift_scores table
        ├──▶ If severity = RED → write alert → drift_scores table
        │
        ▼
[Alert Layer]
        │
        ├──▶ Webhook callback (configurable per model)
        └──▶ Slack/email stub (pluggable)

[Weekly Scheduler — every Monday 00:00 UTC]
        │
        ├──▶ Aggregate 7 days of drift_scores per model
        ├──▶ Compute correlation: feature drift vs. output shift
        ├──▶ Rank features by weighted drift score
        ├──▶ Generate overall health score
        ├──▶ Write → health_reports table
        └──▶ Emit report (JSON + Markdown)
```

---

## 4. Project Structure

```
driftwatch/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings (env vars)
│   ├── database.py              # SQLAlchemy engine + session
│   │
│   ├── routers/
│   │   ├── ingest.py            # /ingest/* endpoints
│   │   ├── models.py            # /models/* endpoints
│   │   ├── reports.py           # /reports/* endpoints
│   │   └── alerts.py            # /alerts/* endpoints
│   │
│   ├── services/
│   │   ├── drift_engine.py      # KS, PSI, JS divergence logic
│   │   ├── importance_scorer.py # Weighted drift scoring
│   │   ├── report_generator.py  # Weekly health report builder
│   │   └── alert_service.py     # Alert dispatch
│   │
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── model_registry.py
│   │   ├── snapshot.py
│   │   ├── drift_score.py
│   │   └── health_report.py
│   │
│   └── schemas/                 # Pydantic request/response schemas
│       ├── ingest.py
│       ├── drift.py
│       └── report.py
│
├── migrations/                  # Alembic migrations
│   └── versions/
│
├── scheduler/
│   └── weekly_report.py         # APScheduler or cron entrypoint
│
├── tests/
│   ├── test_drift_engine.py
│   ├── test_ingest.py
│   └── test_reports.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── architecture.md              ← this file
├── states.md
└── .github/
    ├── instructions.md
    └── workflows/
        └── ci.yml
```

---

## 5. Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| API | FastAPI | Async, auto-docs, fast |
| Database | PostgreSQL | JSONB for distributions, strong indexing |
| ORM | SQLAlchemy 2.x + Alembic | Type-safe, migrations |
| Stats | scipy, numpy | KS test, divergence calculations |
| Scheduling | APScheduler | In-process weekly report trigger |
| Containerisation | Docker + docker-compose | Local dev + deployment parity |
| Testing | pytest + httpx | Async-safe API tests |

---

## 6. Environment Variables

```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/driftwatch
SECRET_KEY=changeme
ALERT_WEBHOOK_URL=https://hooks.example.com/drift
REPORT_SCHEDULE_CRON=0 0 * * 1
LOG_LEVEL=INFO
```

---

## 7. Key Design Decisions

1. **No ground-truth required** — All drift detection operates on input feature distributions and output prediction distributions only. Labels are never required.
2. **Reference distribution stored at registration** — Teams upload their training set stats once; all future windows are compared against this baseline.
3. **Weighted scoring prevents alert fatigue** — A feature with 0.01 importance drifting hard is deprioritised against a 0.40-importance feature drifting slightly.
4. **JSONB distributions** — Storing histogram bins in JSONB avoids schema migrations when feature sets change between model versions.
5. **Pluggable alert backends** — Alert service uses a strategy pattern so Slack, PagerDuty, or email can be added without touching core logic.
