# Runbook

What to check, in order, when JobLens misbehaves. Every command here is one
that exists in the repo today; nothing is aspirational. The three scenarios
are the ones the Phase 7 plan named, plus the one that actually happened
during Phase 6.

Start with health and metrics before anything else:

```bash
curl -s http://localhost:8000/health | python -m json.tool
curl -s http://localhost:8000/metrics | grep -E "^joblens_(db_up|postings|ingest_last|llm_cost_usd_today)"
```

`/health` answers "is the API up, can it reach the database, how much data
is there, when did the last ingest finish, and which LLM is configured".
`/metrics` gives the same facts as numbers Grafana can alert on. If `/health`
itself returns 500, the database is the problem: go to the third section.

Dashboard: `make monitoring-up`, then http://localhost:3000/d/joblens.
Alerts are provisioned from `monitoring/grafana/provisioning/alerting/rules.yml`.

---

## Ingestion failed

**How you find out.** One of:

- An open issue labelled `ingest-failure` on the repository. The daily
  workflow opens it on failure and comments on it if it fails again.
- The Grafana alert "Ingestion stale": no source has completed a successful
  run in 36 hours.
- `/health` shows `last_ingest.status` as `failed`, or a `finished_at` more
  than a day old.

**Triage.**

1. Open the failed run from the issue link, or:

   ```bash
   gh run list --workflow ingest --limit 5
   gh run view --log-failed
   ```

2. Read the exit reason. The pipeline exits non-zero only when every source
   failed; one board being down is logged and tolerated. So a red run means
   something shared broke: the database URL, the network, or the code.

3. Check what the database saw. The run log is the source of truth, not the
   Actions output:

   ```sql
   select source, status, started_at, finished_at, fetched, inserted, error
     from ingestion_runs
    order by started_at desc
    limit 10;
   ```

   Same thing without SQL:

   ```bash
   python -m joblens stats
   ```

**Likely causes, most common first.**

| Symptom in the log | Cause | Fix |
| --- | --- | --- |
| `connection refused` or `could not translate host name` | `DATABASE_URL` secret is wrong, or the hosted database is paused | Check the secret in repo settings. Neon and Supabase free tiers suspend idle databases; the first connection wakes them and can time out. Re-run the workflow. |
| HTTP 429 or 403 from one source | Rate limited or blocked | Nothing to fix tonight. The other sources ran. If it persists for three days, lower `--limit` for that source or check whether the board changed its terms. |
| `KeyError` or `ValidationError` inside `to_posting` | A source changed its payload shape | The raw payload is already in `raw_postings`. Fix the parser, add a fixture from the real payload to `tests/fixtures/`, then `python -m joblens transform --source <name>` to re-parse without re-fetching. |
| Adzuna skipped | `ADZUNA_APP_ID` or `ADZUNA_APP_KEY` missing | Expected when the secrets are not set. Not a failure. |
| Run is green but `inserted` is 0 for every source for days | The boards have nothing new, or the dedup is eating everything | Compare `fetched` to `inserted`. Fetched > 0 with inserted 0 for a week is the second case; check `content_hash` collisions with `select content_hash, count(*) from postings group by 1 having count(*) > 1`. |

**Recovery.** Re-run the workflow by hand:

```bash
gh workflow run ingest -f limit=200
```

The pipeline is idempotent on `(source, source_id)`. Running it twice inserts
nothing twice. A day of missed postings is lost for good only for boards that
do not keep history; Hacker News threads and Adzuna do.

**Close the loop.** Close the `ingest-failure` issue once a green run has
landed, and note the cause in `docs/devlog.md` if it was new.

---

## LLM provider down or expensive

**How you find out.**

- `/chat` and `/match` return 503 `no LLM backend configured`, or hang and
  return 500.
- The Grafana alert "LLM spend today over budget" fires (more than 1 USD
  since midnight UTC).
- `joblens_llm_calls_total{ok="false"}` is climbing on the dashboard.

**Triage.**

1. Which backend is live, and does the service think it is usable?

   ```bash
   curl -s http://localhost:8000/health | grep -E "llm_backend|llm_enabled"
   ```

   `llm_enabled: false` with `llm_backend: anthropic` means the key is not
   set in the environment the API is running in. That is configuration, not
   an outage.

2. Is the provider reachable from where the API runs?

   ```bash
   # Ollama
   curl -s -m 5 "$OLLAMA_BASE_URL/api/tags"
   # Anthropic, without spending anything: an auth check
   curl -s -m 10 https://api.anthropic.com/v1/models \
     -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01" | head -c 200
   ```

3. What have the calls been doing? Every call is logged with its prompt,
   response, latency, cost and error:

   ```bash
   python -m joblens spend --days 1
   ```

   ```sql
   select created_at, feature, ok, attempts, latency_ms, left(error, 120)
     from llm_calls
    order by created_at desc
    limit 20;
   ```

   A rising `attempts` average means the model has started returning JSON
   the schema rejects: that is a prompt regression or a model change, not
   an outage. Compare `prompt_version` between the good and bad rows.

**What the service does on its own.** `/chat` and `/match` fail closed. There
is no fallback model: an answer from a different model than the one the eval
suite scored is an unscored answer. Search and trends keep working because
they never touch the LLM. This is deliberate; see Phase 4 in the README.

**Cost control.** The rate limit is `RATE_LIMIT_PER_MINUTE` (default 10) per
client IP per process. `joblens_llm_cost_usd_today` is the number to watch.
If it is climbing from one IP, lower the limit and restart; if it is climbing
from many, take `/chat` and `/match` offline by unsetting
`ANTHROPIC_API_KEY` (the endpoints then return 503 and cost nothing) until
you know why.

**Switching backends.** `LLM_BACKEND=ollama` or `LLM_BACKEND=anthropic` plus
the matching model variable, then restart. Same code path either way. Run
`python -m joblens eval --suite chat` before calling the switch done: the
Phase 5 floors were set on llama3.1 and an API model will move every number.

---

## Database slow or down

**How you find out.**

- `/health` returns 500.
- `joblens_db_up` is 0; the Grafana alert "Database unreachable from the
  API" fires after two minutes.
- `/search` p95 on the dashboard climbs above a second. Vector search on
  465 postings is normally under 100 ms end to end.

**Triage.**

1. Can you reach it at all?

   ```bash
   docker compose exec -T db pg_isready -U joblens      # local
   psql "$DATABASE_URL" -c "select 1"                    # anywhere
   ```

2. Is it slow or is it stuck? Look for long-running or blocked queries:

   ```sql
   select pid, now() - query_start as age, state, wait_event_type, left(query, 80)
     from pg_stat_activity
    where state <> 'idle' and query_start < now() - interval '5 seconds'
    order by query_start;
   ```

3. Are the indexes there? Vector search without the HNSW index is a
   sequential scan over every chunk, which is fast at 465 postings and a
   cliff at 50,000:

   ```sql
   select indexname from pg_indexes where tablename = 'posting_chunks';
   ```

   The migrations create them; if one is missing, `python -m joblens
   migrate` is idempotent and safe to re-run.

**Likely causes.**

| Symptom | Cause | Fix |
| --- | --- | --- |
| Refused connections, `/health` 500, nothing in the logs | Postgres is not running | Local: `make services-up`. Hosted: the provider's status page, then the console. Free tiers pause idle databases. |
| Connections accepted, queries hang | Lock held by a long transaction, usually an interrupted ingest or a test run | Find it with the `pg_stat_activity` query and `select pg_terminate_backend(<pid>)`. The pipeline commits per source, so nothing is lost. |
| Slow only under load | The API opens one connection per request with no pool | Known limitation, written down in the README. At this size it is fine. If it stops being fine, `psycopg_pool` is the change. |
| Slow after a big ingest | Planner statistics are stale | `analyze postings; analyze posting_chunks;` |
| Disk full | `raw_postings` keeps every fetch forever | It is the bronze layer and is meant to. Free tier limits are the constraint; `delete from raw_postings where fetched_at < now() - interval '90 days'` is safe because `postings` is what the app reads. Confirm before running it. |

**Do not** point `DATABASE_URL` at production and run `pytest`. The test
suite truncates the database it is given. This is the single most dangerous
thing in the repo and it is written on the README too.

---

## Deploying and rolling back

The pipeline on `main` is lint, tests, retrieval eval, then image build. The
image is tagged with the short commit SHA and pushed to
`ghcr.io/raviteja311/joblens`. Rolling back is deploying the previous tag;
there is no separate rollback path because that would be a second, less
tested path.

```bash
gh run list --workflow deploy --limit 5
docker pull ghcr.io/raviteja311/joblens:<sha>
```

Schema changes are additive (`create table if not exists`, `alter table add
column if not exists`) so an older image runs against a newer schema. Run
`python -m joblens migrate` before the new image serves traffic, never after.

---

## Things that broke once and what was learned

- **Services started from a shell die with the shell.** Ollama and Postgres
  were started by a Claude Code session that ended, and a two hour training
  run died with them. `make services-up` starts Ollama detached and
  `make services-check` gives a one-word answer per service.
- **`SSL_CERT_FILE` in the conda base environment points at a file that does
  not exist.** Every HTTPS call from Python fails with a certificate error
  and six tests go red for no visible reason. Fix for the session:
  `export SSL_CERT_FILE="$(python -c 'import certifi;print(certifi.where())')"`.
- **Memory pressure looks like a model bug.** With 0.2GB free, Ollama
  returned HTTP 500 on every request and the teacher extractor scored 0.000
  with 60 unparseable outputs. Nothing was wrong with the prompt. Check
  free memory before debugging a model.
- **The test suite renumbers `postings.id`.** Anything that stores a posting
  id across runs is broken after the next `pytest`. Key on
  `(source, source_id)` instead, as the golden set and the finetune labels
  now do.
