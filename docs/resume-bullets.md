# Resume bullets, with the real numbers

The project plan listed five bullets to fill in "with your real numbers".
Here they are, and where the real number does not exist yet the bullet says
so rather than borrowing one. Every figure points to the README section or
experiment that produced it. Dates are as of 2026-10-27.

The corpus changed on 2026-10-08, from 466 Hacker News and RemoteOK postings
(v1, archived in `docs/archive/v1-global-corpus/`) to Indian company boards.
Bullets marked *(v1)* describe work measured on the old corpus; that is
still true and still claimable, just say which corpus if asked.

## Bullets that are true today

- **Built an end-to-end job intelligence platform** ingesting AI, ML and
  data postings daily from the Greenhouse, Lever and Ashby boards of 291
  Indian companies into a bronze/silver Postgres schema, with India and
  target-role filters, content-hash and embedding dedup, and idempotent
  re-processing from raw payloads; 697 postings in the current corpus, and a
  spot check of 20 found 20 in India and 17 in a target role. *(README
  Phase 1, experiments 2026-10-08)* Say, if asked: the 17 counts
  AI-adjacent roles; under the labelling guide's stricter role families it
  is about 13, and a bare "AI" in a title is the filter's weak spot.

- **Built a human-graded retrieval golden set and eval harness** for a
  pgvector search system: 25 queries graded by hand over 918 candidates
  pooled from every retriever, scoring six configurations; vector search
  leads on nDCG@10 (0.700 against hybrid's 0.671) while hybrid leads on
  recall@10 (0.322 against 0.287), and I measured human-versus-LLM grading
  agreement (weighted kappa 0.24) rather than assuming it. *(README
  Results, experiments 2026-10-09)* Say, if asked: one grader, 24 queries
  scored because one has no relevant posting; about 28 relevant postings
  per query puts MRR near its ceiling and caps recall@10 at 0.421.

- **Ran a pre-registered reranker experiment**: fixed the decision rule
  (+0.05 nDCG@10 at p95 under 500 ms) before any result existed and tested
  14 cross-encoder configurations on the human-graded set. None passed: the
  best gained +0.056 at p95 2.9 s and the fastest +0.026 at p95 434 ms, so
  reranking stayed opt-in. A configuration that looked like a winner on LLM
  grades (+0.129 MRR) gained +0.030 on human grades. *(experiments
  2026-10-09)* Say, if asked: the metric was changed from MRR to nDCG@10
  before the decisive run, because MRR was at its ceiling, and the change is
  recorded as an amendment; under the original MRR rule nothing passed
  either.

- **Improved city-specific search precision from 0.28 to 0.68** (share of
  the top 10 located in the city a query names, over the 10 golden queries
  that name one) by indexing location and adding a city vote to rank
  fusion, choosing the fusion window from a sweep that traded city
  precision against role relevance. *(experiments, city precision)*

- **Built an LLM evaluation suite** (retrieval metrics, LLM-as-judge with a
  Cohen's kappa calibration check, structural refusal detection) whose
  retrieval suite runs in GitHub Actions and fails the build below an
  nDCG@10 floor of 0.45 or on a drop of more than 0.05 against a committed
  baseline. Found and fixed two ways the gate could not fail: it once
  compared each run with itself, and a piped CI step without `pipefail`
  swallowed its exit code. *(README Phase 5, experiments 2026-10-09)* Say,
  if asked: the chat suite is not in CI; the judge is uncalibrated (kappa
  0.00 on 20 hand-scored answers that were all 2s); and a versioned prompt
  fix raised judge faithfulness from 0.69 to 0.75 on an A/B (v1).

- **Ran a LoRA distillation experiment** (Qwen2.5-0.5B student, llama3.1 8B
  teacher, 242 labelled postings): diagnosed a silent loss-masking failure
  that collapsed the student to 0.000 F1, rebuilt a circular benchmark
  blind, and recovered to 0.485 F1; the 80-rule regex baseline still wins
  (0.667 F1, precision 1.000, 5 ms) and stays in production. The blind
  benchmark also exposed a Unicode word-boundary bug in the regex, which I
  fixed and re-measured on the same gold set. *(Phase 6, v1; experiments
  2026-10-27)*

- **Containerised and instrumented the service**: multi-stage Docker build
  (API image 577MB compressed, torch-free UI image 159MB), structured JSON
  logs, Prometheus request histograms and LLM cost counters, a provisioned
  Grafana dashboard with three alert rules, a CI/CD pipeline that gates the
  image build on lint, tests and the retrieval eval, a runbook, and an
  idempotent Azure Container Apps deployment script. *(Phase 7)*

## Bullets from the plan that cannot be claimed yet, and why

- *"Improved retrieval quality from X% to Y% recall@10 by tuning chunking,
  hybrid search, and reranking."* There is no single overall X to Y: on the
  human-graded set no configuration dominates every metric, and reranking
  did not earn the default. The claimable before-and-after is narrower and
  real: city precision 0.28 to 0.68 for queries that name a city (above).
  The v1 figures (58 LLM-graded queries, recall@10 0.653 for vector) belong
  to the archived corpus; do not quote them as current.

- *"Fine-tuned a small open model with LoRA, matching GPT/Claude API accuracy
  at a fraction of the per-request cost."* No API model was in the
  comparison (no key configured) and the student did not match the teacher.
  The claimable version is the Phase 6 bullet above.

- *"Deployed on cloud with Docker, CI/CD, and monitoring; served real users
  with p95 latency under Z ms."* Not deployed; no real users. An Azure
  deployment script is ready but has not been run. Request latency is
  measured (there is a histogram and a dashboard for it) but only on one
  laptop under synthetic traffic, which is not a service-level number.
  Claim the containerisation, pipeline and deployment script, not a
  deployment, until the public URL exists.

## Interview stories these bullets open

Each bullet has a devlog or experiments entry behind it with a specific
thing that broke. The ones that land best in an interview:

1. The base64 of my own IP address was the most discriminative token in the
   corpus. *(Phase 2 devlog)*
2. The hybrid search that had been vector-only for weeks because Postgres
   ANDs search terms. *(Phase 3 devlog)*
3. The refusal metric I rewrote three times before it stopped agreeing with
   the model. *(Phase 5 devlog)*
4. Zero of 316 positions masked: the flag that was silently ignored. *(Phase
   6 experiments)*
5. Precision of exactly 1.000 was an identity, not a result, and the blind
   rebuild found a real bug: `excel` inside "Excelência". *(Phase 6 devlog,
   experiments 2026-10-27)*
6. `docker build .` builds the last stage, and my API image was the UI.
   *(Phase 7 devlog)*
7. A test run wiped the dev corpus fifteen minutes after the golden set was
   scored, and the rebuild became a robustness test. *(2026-09-26 devlog)*
8. The eval gate that could never fail: a green check on a crashed eval,
   because a piped step had no `pipefail`. *(experiments 2026-10-09)*
9. The reranker that won on LLM grades and lost on human ones, and why the
   decision rule was written down before the run. *(experiments 2026-10-09)*
