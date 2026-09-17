# JobLens-AI-Job-Market-Assistant


Collects real AI/ML job postings daily, then offers semantic search,
resume matching, grounded chat with citations, and a live analytics
dashboard.

Status: Phases 0-2 complete. 465 postings from 2 live sources, a skill
extractor, a salary model comparison that the baseline won, and labelled job
clusters. Phase 3 (embeddings and semantic search) is next.

## Development process

Every phase ends with three things: a merged PR, an updated section in
this README, and a note in `docs/devlog.md`. Experiments, including the ones
that failed, go in `docs/experiments.md`.

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
python -m joblens skills               # most requested skills
python -m joblens trends               # demand, regions, remote share
python -m joblens cluster              # job families, labelled
python -m joblens train-salary         # compare salary models
```

`make ingest`, `make test`, `make lint` and friends wrap the same commands.

## Repo structure

```
src/joblens/         ingestion pipeline, sources, database
src/joblens/ml/      Phase 2: skills, salary model, clustering, trends
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

Everything here runs on the Phase 1 corpus. No LLM calls, no embeddings. These
are the baselines Phases 3 and 6 have to beat, so they stay in the repo
permanently. Full write-ups in `docs/experiments.md`.

**Skill extraction.** An alias table over 80 canonical skills with
case-insensitive regex, plus a TF-IDF keyword extractor for the terms the
table does not know about yet. Coverage, the share of postings where at least
one known skill is found, is **58.9%**.

| skill | postings | share |
| --- | ---: | ---: |
| python | 112 | 24.1% |
| typescript | 83 | 17.8% |
| postgresql | 71 | 15.3% |
| llm | 59 | 12.7% |
| aws | 51 | 11.0% |
| kubernetes | 48 | 10.3% |

**Salary regression: the baseline won.** 106 postings with a salary that
parsed, 5-fold CV, MAE as the headline because a few 750k postings dominate
RMSE.

| model | MAE (USD) | RMSE (USD) | R2 | vs median |
| --- | ---: | ---: | ---: | ---: |
| ridge | 62,904 | 88,374 | -0.098 | +1.8% |
| ridge_log | 63,119 | 88,790 | -0.064 | +1.4% |
| median | 64,042 | 86,859 | -0.036 | +0.0% |
| random_forest | 68,563 | 97,385 | -0.340 | -7.1% |
| gradient_boosting | 78,292 | 113,514 | -1.113 | -22.3% |

The ridge's 1.8% edge is inside the noise, and its top features are `money`,
`worth` and `meaningful`, which is what 106 rows against 14,000 TF-IDF columns
looks like. Seven more regularisation and feature-set configurations were
tried; all of them converge towards the baseline rather than past it. **There
is no salary prediction endpoint**, and there will not be one until there are
roughly 500 salary-disclosing postings. The comparison ships instead.

**Clustering.** TF-IDF and k-means, k chosen by silhouette, clusters labelled
from their own centroid terms. k=10 over 465 postings gives nameable families:
`founding engineer`, `forward deployed engineer`, `open source / developer`,
`ml / ai`, `san francisco onsite`, `rails / toronto`.

The first version did not. It produced a 99-posting cluster defined by the
terms `applicants`, `rmjcuni4xmjgumja5`, `read`, `word`, `human`. RemoteOK
appends an anti-bot canary to every posting it serves, containing a token that
is base64 of the scraper's own IP. Identical across one source and absent from
the other, so it was the most discriminative term in the corpus and k-means
built a cluster meaning "came from RemoteOK". A second cluster had learned
which applicant tracking system the employer uses, from URLs in 329 of 465
postings. `cleaning.strip_noise()` removes both, at feature time so the stored
description stays as published.

Stripping it made the clusters obviously better and the silhouette score
slightly **worse** (0.0110 to 0.0065 at k=5). Optimising that metric would
have reverted the fix. This is the argument for building the Phase 3 golden
dataset before anything else.

**Trends.** Top skills over time, demand by region, remote share, median
advertised salary by skill. Every share is reported against the postings that
could have answered the question, not against the whole corpus: 65% of
postings are remote, but only 23% state a salary at all.

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
- **No salary prediction.** The models do not beat predicting the median. See
  Phase 2 above.
- **Salary figures cover 23% of postings** and those are not a random sample:
  boards that require a salary field are over-represented in every salary
  number on this page.
- **Currency conversion uses rates frozen on 2025-09-01.** Refreshing them
  retroactively would change every past experiment's numbers, so they stay
  fixed and this is the caveat. Salaries in an unrecognised currency are
  dropped rather than assumed to be dollars.
- **Locations are a lookup table, not a geocoder.** The grouping key is
  whatever follows the last comma, so "Austin, TX" groups under TX and
  "Berlin" groups under Berlin. 37% of postings have no usable location and
  sit in `unknown`.
- **Dedup is exact-match only** and misses reworded reposts of the same job.
- **Skill extraction knows 80 skills** and nothing else. The 41% of postings
  with no skill found are mostly non-technical roles and postings that
  describe work without naming tools, so that figure is not a miss rate.
- **Silhouette is a weak judge of the clustering** at these values, as the
  Phase 2 note above shows. There is no trustworthy retrieval or clustering
  metric in the repo until the Phase 3 golden dataset exists.
- **The test suite truncates the development database.** Point `DATABASE_URL`
  at anything you care about and `pytest` will empty it.
