# JobLens-AI-Job-Market-Assistant


Collects real AI/ML job postings daily, then offers semantic search,
resume matching, grounded chat with citations, and a live analytics
dashboard.

Status: Phases 0-1 complete. 465 postings from 2 live sources, collected
daily. Phase 2 (classic ML layer) is next.

## Development process

Every phase ends with three things: a merged PR, an updated section in
this README, and a note in `docs/devlog.md`.

## Setup

Requires Python 3.12 and Docker for the local database.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

cp .env.example .env
docker compose up -d db
python -m joblens migrate
python -m joblens ingest
```

## Commands

Everything runs through one entry point, so the cron job, CI and a human
debugging a bad run all take the same code path.

```bash
python -m joblens ingest --limit 200   # fetch every source into the database
python -m joblens transform            # re-parse bronze after a parser fix
python -m joblens stats                # what is in the database right now
```

`make ingest`, `make test`, `make lint` and friends wrap the same commands.

## Repo structure

```
src/joblens/         ingestion pipeline, sources, database
tests/               pytest suite
migrations/          schema, applied by `joblens migrate`
scripts/             entry points and one-off jobs
docs/                devlog, experiments
```

## Phases

### Phase 0: Setup and habits

Repo structure, pyproject.toml, pre-commit with ruff and black, project
board with phases as milestones.

### Phase 1: Data ingestion pipeline

Three source adapters behind one interface. A source is any module with
`fetch(limit)` and `to_posting(payload)`, which is what lets the pipeline
re-parse everything already collected after a parser fix without touching the
network.

| source | postings | API key | notes |
| --- | ---: | --- | --- |
| Hacker News "Who is hiring" | 366 | none | free-text comments, no schema at all |
| RemoteOK | 99 | none | structured, entirely remote |
| Adzuna | 0 | required | registered but skipped without a key |

**Storage** is bronze/silver. `raw_postings` keeps the payload exactly as the
source sent it, one row per fetch, history retained. `postings` holds our
interpretation. When the salary parser turns out to be wrong, `transform`
re-runs over the raw layer instead of waiting two weeks to re-collect.

**Cleaning** strips HTML with the stdlib, normalises locations against a
lookup table rather than a geocoder, and parses salary out of free text:
ranges, `up to`, `from`, hourly and monthly rates, six currencies, and the
strings that mean "we are not telling you". Hourly and monthly figures are
annualised only when the period was actually stated; a guessed period is how a
$60/hr contract ends up in the data as a $60 salary.

**Dedup** is an exact match on a normalised content hash of title, company and
location, with seniority words removed. It currently finds 26 duplicates
across the two sources. It misses "Senior ML Engineer" against "ML Engineer
(Senior)" and it is supposed to: this number is the baseline that Phase 3's
embedding-similarity dedup gets measured against.

**Reliability.** Retries with backoff on 429 and 5xx, fails fast on other 4xx.
One source failing does not stop the others. Every run writes to
`ingestion_runs` with counts and any error. 34 of 400 Hacker News comments are
replies rather than postings and are counted as skipped, not dropped silently.

**Schedule.** `.github/workflows/ingest.yml` runs daily at 06:00 UTC and is
idempotent, so a missed slot costs that day's postings and nothing else.

Sources are the ones that permit this: two public APIs and one that requires a
key. `USER_AGENT` identifies the project and should carry a real contact
address.

### Phase 2: Classic ML layer

<!-- Salary regression MAE and R2, clustering labels, trend stats. -->

### Phase 3: Embeddings and semantic search

<!-- Retrieval comparison table: keyword vs vector vs hybrid vs reranked. -->

### Phase 4: RAG chat and resume matching

<!-- Endpoint behaviour, prompt versioning, cost per request. -->

### Phase 5: Evaluation harness

<!-- Judge calibration agreement rate, eval trends, CI regression gate. -->

### Phase 6: Fine-tuning

<!-- API model vs base small model vs fine-tuned: accuracy, cost, latency. -->

### Phase 7: Deployment and MLOps

<!-- Live URL, CI badge, monitoring screenshot, runbook link. -->

### Phase 8: Packaging

<!-- Demo video, blog posts, dataset. -->

## Limitations

Kept current and honest.

- **The corpus is small and skewed.** 465 postings from 2 sources, 79% of them
  Hacker News comments, which over-represents startups and US remote work. Any
  "the market wants X" claim from this data is really "these two boards wanted
  X this month".
- **Salary figures cover 23% of postings** and those are not a random sample:
  boards that require a salary field are over-represented.
- **Locations are a lookup table, not a geocoder.** 37% of postings have no
  usable location.
- **Dedup is exact-match only** and misses reworded reposts of the same job.
- **The test suite truncates the development database.** Point `DATABASE_URL`
  at anything you care about and `pytest` will empty it.
