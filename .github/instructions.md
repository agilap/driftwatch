# DriftWatch — GitHub Copilot & AI Coding Instructions

> These instructions apply to all AI-assisted work in this repository.
> Every build session and pull request must follow these rules.

---

## 1. Project Identity

- **Project:** DriftWatch — Production Model Health Monitor
- **Purpose:** Detect data drift in deployed ML models without ground-truth labels
- **Stack:** FastAPI · PostgreSQL · SQLAlchemy 2.x · scipy · numpy · Alembic · APScheduler · Docker
- **Iteration:** I-2 (3–4 weeks)
- **Key constraint:** No labels required — only input feature distributions and output prediction distributions

---

## 2. Code Style & Standards

### Python
- Python 3.11+
- Type hints on **all** function signatures — no bare `Any` without a comment
- Pydantic v2 for all request/response schemas
- SQLAlchemy 2.x async patterns (`async_session`, `select()` style)
- Use `async def` for all route handlers and DB operations
- `ruff` for linting; `black` for formatting (line length 100)
- Docstrings on all public functions using Google style:
  ```python
  def compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
      """Compute Population Stability Index between two distributions.

      Args:
          reference: Array of reference (training) values.
          production: Array of production values.
          bins: Number of histogram bins. Defaults to 10.

      Returns:
          PSI score. <0.10 stable, 0.10–0.20 moderate drift, >0.20 significant drift.
      """
  ```

### Naming Conventions
| Type | Convention | Example |
|---|---|---|
| Files | `snake_case.py` | `drift_engine.py` |
| Classes | `PascalCase` | `DriftScore` |
| Functions / variables | `snake_case` | `compute_psi` |
| Constants | `UPPER_SNAKE` | `PSI_RED_THRESHOLD` |
| DB table names | `snake_case` | `drift_scores` |
| Env vars | `UPPER_SNAKE` | `DATABASE_URL` |

---

## 3. Architecture Rules

- **Never** add a new table without a corresponding Alembic migration
- **Never** hardcode thresholds — all drift thresholds live in `app/config.py` as `Settings` fields
- **Never** perform DB operations in route handlers — always delegate to a `service` module
- **Always** follow the data flow in `architecture.md §3` — ingest → drift engine → alerts → reports
- **Always** store computed distributions as JSONB (bins + counts), never raw arrays
- **Feature importances are optional** — the system must work without them (default to equal weighting)

---

## 4. Statistical Test Rules

- KS test: use `scipy.stats.ks_2samp` — always store both `statistic` and `p_value`
- PSI: use 10 bins minimum; add `1e-4` epsilon to avoid `log(0)` errors
- JS divergence: use `scipy.spatial.distance.jensenshannon` (returns already-squared value in scipy ≥1.7, verify)
- Always clip divergences to `[0, 1]` before storing
- Log a warning (not an error) when sample count < 30 for any feature window

---

## 5. API Design Rules

- All endpoints return typed Pydantic response models — no raw dicts
- All error responses use this shape:
  ```json
  { "detail": "Human-readable message", "code": "SNAKE_CASE_ERROR_CODE" }
  ```
- Pagination on all list endpoints: `?page=1&page_size=50`
- `model_id` is always a UUID — validate with Pydantic
- Timestamps are always UTC ISO 8601 strings in responses

---

## 6. Testing Rules

- Every new service function gets a unit test in `tests/`
- Every new endpoint gets an integration test using `httpx.AsyncClient`
- Test file mirrors source path: `app/services/drift_engine.py` → `tests/test_drift_engine.py`
- Use `pytest-asyncio` for all async tests
- Minimum test coverage for new code: **80%**
- Use factory functions (not fixtures with hardcoded data) for test data

---

## 7. Git Commit Rules

### Format
```
<type>(<scope>): <short description>

[optional body — what and why, not how]

[optional footer — closes #issue, breaking changes]
```

### Types
| Type | When to use |
|---|---|
| `feat` | New feature or endpoint |
| `fix` | Bug fix |
| `test` | Adding or updating tests |
| `refactor` | Code change with no behaviour change |
| `docs` | Documentation only |
| `chore` | Build, deps, CI changes |
| `db` | Schema/migration changes |

### Scopes
`ingest` · `drift` · `scorer` · `reports` · `alerts` · `db` · `api` · `ci` · `config`

### Examples
```
feat(ingest): add POST /ingest/snapshot endpoint with distribution computation

fix(drift): add epsilon to PSI bins to prevent log(0) error

db(migrations): add drift_scores and alerts tables

test(drift): add unit tests for KS test and PSI with synthetic data
```

### Rules
- Subject line max **72 characters**
- Use imperative mood: "add", "fix", "compute" — not "added", "fixed"
- **Never** commit directly to `main` — always use a feature branch
- Branch naming: `feat/scope-description`, `fix/scope-description`

---

## 8. States Update Protocol

At the end of every build session, **update `states.md`**:

1. Change any completed tasks from `⏳` to `✅`
2. Change any in-progress tasks from `⏳` to `🔄`
3. Change any blocked tasks to `❌` and note the blocker
4. Append a new entry to the **Build Session Log** section:
   ```
   **[YYYY-MM-DD]** — <summary of what was built>
   Files changed: <list>
   Tests added: <yes/no — count>
   Next: <what to tackle next session>
   ```
5. Update the `Last updated` date and `Current phase` at the top

---

## 9. What NOT to Do

- ❌ Do not use `print()` for logging — use Python `logging` module
- ❌ Do not use synchronous DB calls in an async context
- ❌ Do not store raw feature arrays in the DB — always aggregate to histogram stats
- ❌ Do not generate reports inside route handlers — use background tasks or scheduler
- ❌ Do not skip type hints to save time
- ❌ Do not modify `architecture.md` without noting the change in the session log
- ❌ Do not commit `.env` files — only `.env.example`

---

## 10. File Change Checklist (Before Every Commit)

- [ ] Type hints on all new functions
- [ ] Docstrings on all public functions
- [ ] New DB model has a migration
- [ ] New endpoint has a Pydantic response schema
- [ ] New service function has a test
- [ ] `states.md` updated
- [ ] Commit message follows §7 format
