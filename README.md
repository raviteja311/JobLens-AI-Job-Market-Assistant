# JobLens-AI-Job-Market-Assistant


Collects real AI/ML job postings daily, then offers semantic search,
resume matching, grounded chat with citations, and a live analytics
dashboard.

Status: Phases 0-3 complete. 465 postings from 2 live sources, and hybrid
search over pgvector scored against a hand-judged golden set. Phase 4
(RAG chat and resume matching) is next.

## Development process

Every phase ends with three things: a merged PR, an updated section in
this README, and a note in `docs/devlog.md`. Experiments, including the ones
that failed, go in `docs/experiments.md`.

## Setup

Requires Python 3.12 and Docker. The database image is `pgvector/pgvector`,
not plain `postgres`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

cp .env.example .env
docker compose up -d db
python -m joblens migrate
python -m joblens ingest
python -m joblens embed --strategy whole --strategy section
```

## Commands

Everything runs through one entry point, so the cron job, CI and a human
debugging a bad run all take the same code path.

```bash
python -m joblens ingest --limit 200   # fetch every source into the database
python -m joblens transform            # re-parse bronze after a parser fix
python -m joblens stats                # what is in the database right now
python -m joblens embed                # build the vector index
python -m joblens search "remote LLM role" --rerank
python -m joblens dedup                # embedding dedup vs the hash baseline
```

## Repo structure

```
src/joblens/         ingestion pipeline, sources, database
src/joblens/ml/      Phase 2: skills, salary model, clustering, trends
src/joblens/search/  Phase 3: chunking, embeddings, retrieval, reranking
src/joblens/eval/    Phase 3: golden set, retrieval metrics
data/golden/         the judged queries everything is scored against
tests/               pytest suite
migrations/          schema, applied by `joblens migrate`
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
location, with seniority words removed. It finds 26 duplicates. Phase 3 adds a
second, different signal rather than replacing it.

**Reliability.** Retries with backoff on 429 and 5xx, fails fast on other 4xx.
One source failing does not stop the others. Every run writes to
`ingestion_runs` with counts and any error. 34 of 400 Hacker News comments are
replies rather than postings and are counted as skipped, not dropped silently.

**Schedule.** `.github/workflows/ingest.yml` runs daily at 06:00 UTC and is
idempotent, so a missed slot costs that day's postings and nothing else.

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
looks like. Seven more configurations were tried; all converge towards the
baseline rather than past it. **There is no salary prediction endpoint** and
there will not be one until there are roughly 500 salary-disclosing postings.

**Clustering.** TF-IDF and k-means, k chosen by silhouette, clusters labelled
from their own centroid terms. k=10 gives nameable families: `founding
engineer`, `forward deployed engineer`, `open source / developer`, `ml / ai`,
`san francisco onsite`, `rails / toronto`.

The first version produced a 99-posting cluster defined by the terms
`applicants`, `rmjcuni4xmjgumja5`, `read`, `word`, `human`. RemoteOK appends
an anti-bot canary to every posting it serves, containing a token that is
base64 of the scraper's own IP, which made it the most discriminative term in
the corpus. `cleaning.strip_noise()` removes it and the URLs that had built a
second cluster meaning "uses Ashby". Stripping it made the clusters obviously
better and the silhouette score slightly **worse**, which is the argument for
building the golden dataset before trusting any unsupervised metric.

**Trends.** Top skills over time, demand by region, remote share, median
advertised salary by skill. Every share is reported against the postings that
could have answered the question: 65% are remote, but only 23% state a salary.

### Phase 3: Embeddings and semantic search

pgvector, a local MiniLM, hybrid retrieval, a cross-encoder reranker, and the
golden set that makes all of it measurable.

**The golden set** is 60 real queries in `data/golden/retrieval.yaml`, of
which 15 are judged, graded 0 (no) / 1 (acceptable) / 2 (exactly what was
asked). Judgements key on `(source, source_id)` and never on the primary key,
because a re-ingest into an empty database renumbers every posting and would
silently re-point every judgement at a different job. Candidates come from a
pool of all three retrievers, not from the system under test, so a retriever
cannot score well against its own blind spots.

**Retrieval comparison**, 15 judged queries, 465 postings, models warmed
before timing:

| configuration | recall@5 | recall@10 | mrr | ndcg@10 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.299 | 0.358 | 0.406 | 0.328 | 25 |
| vector (whole) | 0.569 | 0.692 | 0.773 | 0.598 | 34 |
| vector (section) | 0.491 | 0.813 | 0.734 | 0.684 | 35 |
| hybrid (whole) | 0.438 | 0.666 | 0.761 | 0.620 | 94 |
| hybrid (section) | 0.430 | 0.646 | 0.736 | 0.600 | 93 |
| hybrid + rerank | 0.497 | 0.725 | **0.900** | **0.692** | 3145 |

**Hybrid lost to plain vector search**, which was not the expected result.
Keyword retrieval has to OR its terms, because ANDing them (what
`websearch_to_tsquery` does) returns nothing at all for "remote machine
learning engineer working on LLMs"; the first version of this shipped that
way and the keyword arm contributed zero candidates. With OR it contributes,
but for a six-word query it ranks a posting matching only "engineer" first,
and reciprocal rank fusion hands that rank-1 noise the same weight as the
vector arm's best result.

Hybrid stays the default anyway, for the case the averages cannot show. On the
query `pgvector`, vector search returns Marketing Manager and ON SITE TORONTO,
because an embedding of "pgvector" is mostly "database"; keyword search
returns the one posting that names it, at rank 1. Hybrid is insurance against
the query type embeddings cannot handle, bought at a measurable cost on
ordinary queries. That is the honest summary and it is in the limitations.

**Reranking is the real improvement.** MRR 0.773 to 0.900, at 3.1 seconds a
query on CPU, so it is off by default and on behind `--rerank`.

**Chunking.** Whole-posting vectors win at the top of the ranking (recall@5,
MRR); section vectors win on depth (recall@10 0.813 vs 0.692). `whole` is the
default because the first three results are what people read.

**Dedup.** Embedding similarity at 0.93 finds 26 pairs, of which 19 overlap
the Phase 1 hash. Each method finds 7 the other misses, because the hash reads
title/company/location and the embedding reads the description. Both are kept.

**Local embeddings** cost nothing: all-MiniLM-L6-v2, 384 dimensions, 1.67 ms
per text, 25 seconds to index the corpus. The plan's local-vs-API comparison
is **not done**: `ApiEmbedder` is implemented and the harness can score it,
but no key is configured, and quoting a comparison from a pricing page would
be worse than leaving it blank.

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
- **The golden set is 15 judged queries out of 60, and I judged them from
  titles and snippets rather than full postings.** Every retrieval number on
  this page rests on that. The remaining 45 are written but unjudged.
- **Recall is recall over the pool**, not over the corpus. A posting no
  retriever surfaces is never judged and never counted as missed. At 465
  postings the gap is small; it would not be at 50,000.
- **Hybrid retrieval is worse than vector alone** on these queries. It is kept
  for rare-token queries, which the current golden set under-represents.
- **No salary prediction.** The models do not beat predicting the median.
- **Salary figures cover 23% of postings** and those are not a random sample.
- **Currency conversion uses rates frozen on 2025-09-01.**
- **Locations are a lookup table, not a geocoder.** 37% of postings have no
  usable location.
- **Skill extraction knows 80 skills** and nothing else.
- **The test suite truncates the development database.** Point `DATABASE_URL`
  at anything you care about and `pytest` will empty it.
