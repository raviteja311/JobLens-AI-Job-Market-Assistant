# Thin wrappers over `python -m joblens`. The point is that the commands in
# the README, the cron job and CI are the same three words every time.

PY ?= python

.PHONY: install migrate ingest transform stats skills train cluster trends \
	embed dedup serve ui test test-db lint format check

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
