# Devlog

A short note at the end of each work session: what broke, what I decided,
why. These entries become the blog posts and the interview stories later.
Newest entry at the top.

---

## Phase 1 - data ingestion pipeline

**Did:** three source adapters (RemoteOK, Hacker News "Who is hiring" via
Algolia, Adzuna), a bronze/silver schema, the salary parser, content-hash
dedup, a run log, and `python -m joblens` as the one entry point that the
cron job, the GitHub Action and a human debugging all share. Daily schedule
runs as a GitHub Actions workflow.

**Current state:** 465 postings from 2 sources. Adzuna is registered but
skipped because no API key is configured, which is deliberate: a missing
optional key must not fail the run for someone cloning the repo.

**What broke:** Hacker News comments have no schema at all. 34 of 400 fetched
comments are replies or questions rather than postings, so the parser returns
None for them and the run counts them as skipped instead of dying. Counting
the skips is the point: if that number jumps, the parser has started eating
real postings.

**Decided:** fetch and parse stay separate functions. Every source module is
just `fetch()` and `to_posting()`, so `transform` can re-parse everything in
bronze after a parser fix without making a single network request. Already
earned its keep once.

**Known issue:** the test suite truncates the same database the dev data lives
in. Convenient, and one `DATABASE_URL` typo away from being a bad afternoon.
The guard test only checks for hosted-provider hostnames. Tests should get
their own database.

**Next:** Phase 2, classic ML on what has been collected.

---

## Phase 0 - setup and habits

**Did:** repo structure (src, tests, scripts, docs), pyproject.toml,
pre-commit with ruff and black, project board with phases 1-8 as
milestones.

**Decided:** the rule for the whole project. Every phase ends with a
merged PR, an updated README section, and a devlog note here.

**Next:** Phase 1, pick 2-3 job sources and check their terms.