# DriftWatch — Project States

> Last updated: 2026-03-22
> Current phase: **I-2 · Week 1 — Foundation & Ingest API**

---

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | Complete |
| 🔄 | In Progress |
| ⏳ | Pending |
| ❌ | Blocked |
| 🧪 | Needs Testing |

---

## Phase Overview

```
Week 1  ──▶  Foundation & Ingest API
Week 2  ──▶  Drift Engine (statistical tests)
Week 3  ──▶  ML Importance Scorer + Weekly Reports
Week 4  ──▶  Polish, Testing, Docker, Demo
```

---

## Week 1 — Foundation & Ingest API

### Project Setup
| Task | Status | Notes |
|---|---|---|
| Initialise repo + folder structure | ✅ | Per `architecture.md §4` |
| `docker-compose.yml` with PostgreSQL | ✅ | |
| `Dockerfile` for FastAPI app | ✅ | |
| `.env.example` with all required vars | ✅ | See `architecture.md §6` |
| `requirements.txt` pinned deps | ✅ | fastapi, sqlalchemy, scipy, numpy, alembic |
| Alembic init + base migration | ✅ | Initial migration scaffolded |

### Database Layer
| Task | Status | Notes |
|---|---|---|
| SQLAlchemy models — `models` table | ✅ | |
| SQLAlchemy models — `reference_distributions` | ✅ | |
| SQLAlchemy models — `feature_importances` | ✅ | |
| SQLAlchemy models — `snapshots` | ✅ | |
| SQLAlchemy models — `drift_scores` | ✅ | |
| SQLAlchemy models — `health_reports` | ✅ | |
| SQLAlchemy models — `alerts` | ✅ | |

### Ingest API
| Task | Status | Notes |
|---|---|---|
| `POST /models` — register model | ⏳ | |
| `GET /models` — list models | ⏳ | |
| `POST /ingest/reference` — upload reference distribution | ⏳ | Compute & store histogram + stats |
| `POST /ingest/snapshot` — daily production batch | ⏳ | Validate → compute stats → store |
| `GET /health` — liveness probe | ⏳ | |
| Pydantic schemas for all payloads | ⏳ | |

---

## Week 2 — Drift Engine

### Statistical Tests
| Task | Status | Notes |
|---|---|---|
| KS test per feature (`scipy.stats.ks_2samp`) | ⏳ | Compare snapshot vs reference samples |
| PSI computation per feature | ⏳ | 10-bin histogram, thresholds: <0.10 green, 0.10–0.20 yellow, >0.20 red |
| Jensen-Shannon divergence per feature | ⏳ | `scipy.spatial.distance.jensenshannon` |
| Chi-square test for categorical features | ⏳ | `scipy.stats.chisquare` |
| Severity classifier (green/yellow/red) | ⏳ | Based on PSI thresholds |
| Persist results → `drift_scores` table | ⏳ | |
| Trigger drift engine on snapshot ingest | ⏳ | Background task via FastAPI |

### Alert Layer
| Task | Status | Notes |
|---|---|---|
| Alert creation on RED severity | ⏳ | Write to `alerts` table |
| Webhook dispatch (configurable per model) | ⏳ | HTTP POST with drift payload |
| `GET /alerts` — list active alerts | ⏳ | |
| `PATCH /alerts/{id}/resolve` | ⏳ | |

---

## Week 3 — ML Importance Scorer + Weekly Reports

### Drift Importance Scorer
| Task | Status | Notes |
|---|---|---|
| Accept feature importances at model registration | ⏳ | JSON upload, SHAP, or coefficient array |
| Normalise importances (sum to 1) | ⏳ | |
| `weighted_score = drift_magnitude × feature_importance` | ⏳ | See `architecture.md §2.3` |
| Re-rank drift alerts by weighted score | ⏳ | |
| Store weighted scores in `drift_scores` | ⏳ | |

### Weekly Health Report Generator
| Task | Status | Notes |
|---|---|---|
| APScheduler setup — Monday 00:00 UTC | ⏳ | |
| Aggregate 7-day drift scores per model | ⏳ | |
| Compute overall health score (0–100) | ⏳ | Inverse of mean weighted drift |
| Per-feature drift summary table | ⏳ | |
| Prediction distribution histogram comparison | ⏳ | |
| Correlation: feature drift ↔ output shift | ⏳ | Pearson/Spearman |
| Top drifted features list (ranked) | ⏳ | |
| Recommendations engine (rule-based) | ⏳ | Retraining flag if score < 60 |
| Markdown report output | ⏳ | |
| JSON report output | ⏳ | |
| Store → `health_reports` table | ⏳ | |
| `GET /reports/{model_id}` — list reports | ⏳ | |
| `GET /reports/{model_id}/{week}` — fetch specific | ⏳ | |

---

## Week 4 — Polish, Testing & Demo

### Testing
| Task | Status | Notes |
|---|---|---|
| Unit tests — `drift_engine.py` | ⏳ | KS, PSI, JS edge cases |
| Unit tests — `importance_scorer.py` | ⏳ | |
| Integration tests — ingest endpoints | ⏳ | pytest + httpx |
| Integration tests — report generation | ⏳ | |
| Test with synthetic drift dataset | ⏳ | Loan model scenario from brief |

### CI / CD
| Task | Status | Notes |
|---|---|---|
| GitHub Actions CI workflow | ⏳ | Lint + test on push |
| `docker-compose up` smoke test in CI | ⏳ | |

### Documentation & Demo
| Task | Status | Notes |
|---|---|---|
| `README.md` with quickstart | ⏳ | |
| Example notebook — loan model drift scenario | ⏳ | Jan → Aug distribution shift |
| Postman / Bruno collection for API | ⏳ | |
| `CHANGELOG.md` | ⏳ | |

---

## Decisions Log

| Date | Decision | Rationale |
|---|---|---|
| — | JSONB for distributions | Avoids schema changes when feature sets evolve |
| — | Weighted scoring | Prevents alert fatigue from low-importance feature drift |
| — | No labels required | Core design constraint — monitor input/output distributions only |

---

## Blockers

_None currently._

---

## Build Session Log

> Auto-appended by build prompt at end of each session.

<!-- SESSION LOG START -->
**[2026-03-22]** — Scaffold project structure, Docker environment, FastAPI app skeleton
- Files changed: app/__init__.py, app/main.py, app/config.py, app/database.py, app/routers/__init__.py, app/routers/ingest.py, app/routers/models.py, app/routers/reports.py, app/routers/alerts.py, app/services/__init__.py, app/services/drift_engine.py, app/services/importance_scorer.py, app/services/report_generator.py, app/services/alert_service.py, app/models/__init__.py, app/models/model_registry.py, app/models/snapshot.py, app/models/drift_score.py, app/models/health_report.py, app/schemas/__init__.py, app/schemas/ingest.py, app/schemas/drift.py, app/schemas/report.py, migrations/env.py, migrations/script.py.mako, migrations/versions/0001_initial.py, scheduler/__init__.py, scheduler/weekly_report.py, tests/__init__.py, tests/conftest.py, tests/test_health.py, tests/test_drift_engine.py, tests/test_ingest.py, tests/test_reports.py, requirements.txt, .env.example, .gitignore, Dockerfile, docker-compose.yml, alembic.ini, .github/workflows/ci.yml, states.md
- Tasks completed: Initialise repo + folder structure; docker-compose.yml with PostgreSQL; Dockerfile for FastAPI app; .env.example with all required vars; requirements.txt pinned deps; Alembic init + base migration
- Tests added: 1 (health endpoint smoke test)
- Next session: Database layer — SQLAlchemy models + Alembic migrations
**[2026-03-22]** — Add full ORM schema and Alembic migration for all database tables
- Files changed: app/models/model_registry.py, app/models/reference_distribution.py, app/models/feature_importance.py, app/models/snapshot.py, app/models/drift_score.py, app/models/health_report.py, app/models/alert.py, app/models/__init__.py, migrations/env.py, migrations/versions/0001_initial.py, tests/test_models.py, states.md
- Tasks completed: SQLAlchemy models — models table; SQLAlchemy models — reference_distributions; SQLAlchemy models — feature_importances; SQLAlchemy models — snapshots; SQLAlchemy models — drift_scores; SQLAlchemy models — health_reports; SQLAlchemy models — alerts
- Tests added: yes — 3 tests in tests/test_models.py
- Next session: Ingest API endpoints and service-layer persistence
<!-- SESSION LOG END -->
