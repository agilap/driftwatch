POSTGRES_PORT ?= 5433
API_PORT ?= 8000
export POSTGRES_PORT API_PORT

PYTHON := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
BASE_URL := http://localhost:$(API_PORT)
TEST_DATABASE_URL ?= postgresql+asyncpg://driftwatch:driftwatch@localhost:$(POSTGRES_PORT)/driftwatch_test

.PHONY: up demo test down

up:
	docker compose up -d --build

demo: up
	@echo "Waiting for $(BASE_URL)/ready"
	@for i in $$(seq 1 60); do \
		curl -fsS $(BASE_URL)/ready >/dev/null 2>&1 && exit 0; \
		sleep 2; \
	done; \
	echo "API did not become ready"; docker compose logs api; exit 1
	DRIFTWATCH_URL=$(BASE_URL) $(PYTHON) -m examples.loan_risk_demo.run_demo
	@MODEL_ID=$$($(PYTHON) -c 'import httpx; from examples.loan_risk_demo.run_demo import model_id; print(model_id(httpx.Client(base_url="$(BASE_URL)")))') && \
	env -u DRIFTWATCH_LLM_BASE_URL -u DRIFTWATCH_LLM_MODEL -u DRIFTWATCH_LLM_API_KEY \
		DRIFTWATCH_URL=$(BASE_URL) $(PYTHON) -m model_health_agent "$$MODEL_ID"

test:
	docker compose up -d --wait db
	docker compose exec -T db sh -c "psql -U driftwatch -d postgres -tAc \"SELECT 1 FROM pg_database WHERE datname = 'driftwatch_test'\" | grep -q 1 || createdb -U driftwatch driftwatch_test"
	TEST_DATABASE_URL=$(TEST_DATABASE_URL) $(PYTHON) -m pytest tests/

down:
	docker compose down
