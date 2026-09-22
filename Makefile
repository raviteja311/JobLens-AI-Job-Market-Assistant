# Thin wrappers over `python -m joblens`. The point is that the commands in
# the README, the cron job and CI are the same three words every time.

PY ?= python

.PHONY: install migrate ingest transform stats skills train cluster trends \
	embed dedup serve ui eval calibrate test test-db lint format check \
	image up down monitoring-up logs

install:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

migrate:
	$(PY) -m joblens migrate

ingest:
	$(PY) -m joblens ingest --limit $(or $(LIMIT),200)

transform:
	$(PY) -m joblens transform

stats:
	$(PY) -m joblens stats

skills:
	$(PY) -m joblens skills

train:
	$(PY) -m joblens train-salary --save

cluster:
	$(PY) -m joblens cluster

trends:
	$(PY) -m joblens trends

embed:
	$(PY) -m joblens embed --strategy whole --strategy section

dedup:
	$(PY) -m joblens dedup --save

serve:
	$(PY) -m joblens serve --reload

ui:
	$(PY) -m streamlit run app.py

# The scorecard. --strict is what CI runs; locally you usually want to see
# the numbers without the build failing.
eval:
	$(PY) -m joblens eval --suite all

eval-retrieval:
	$(PY) -m joblens eval --suite retrieval

calibrate:
	$(PY) -m joblens calibrate

# Phase 6 learned this the hard way: Ollama and Postgres were started by a
# shell that later exited, both died with it, and a two hour training run was
# lost to services that had been up when it started. These two targets make
# "is everything actually running" a question with a one word answer.
services-up:
	docker compose up -d db
	@powershell -NoProfile -Command "if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) { Start-Process -FilePath 'ollama' -ArgumentList 'serve' -WindowStyle Hidden }" || true
	@$(MAKE) services-check

services-check:
	@docker compose exec -T db pg_isready -U joblens > /dev/null 		&& echo "postgres: up" 		|| (echo "postgres: DOWN, run make services-up" && exit 1)
	@curl -s -m 5 http://localhost:11434/api/tags > /dev/null 		&& echo "ollama:   up" 		|| (echo "ollama:   DOWN, run make services-up" && exit 1)

# Phase 6. distil-label takes about an hour on a local 8B teacher and is
# resumable, so it is safe to interrupt.
distil-label:
	$(PY) -m joblens distil-label

distil-train:
	$(PY) -m joblens distil-train

distil-eval:
	$(PY) -m joblens distil-eval

distil: distil-label distil-train distil-eval

test:
	$(PY) -m pytest -q

# Database tests skip themselves when no Postgres is reachable, which is easy
# to not notice. This target fails instead of skipping.
test-db:
	$(PY) -m pytest -q -m db --no-header -rs --strict-markers

lint:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

check: lint test

# Phase 7. One image serves the API and the UI; the command picks which.
image:
	docker build -t joblens:local .

up:
	docker compose up -d --build db api ui

down:
	docker compose down

# The API plus Prometheus and Grafana at http://localhost:3000.
monitoring-up:
	docker compose --profile monitoring up -d --build

logs:
	docker compose logs -f api
