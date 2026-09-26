# Resume bullets, with the real numbers

The project plan listed five bullets to fill in "with your real numbers".
Here they are filled in, and where the real number does not exist yet the
bullet says so rather than borrowing one. Every figure links back to the
README section or experiment that produced it. Dates are as of 2026-09-26.

## Bullets that are true today

- **Built an end-to-end job intelligence platform** ingesting AI/ML postings
  daily from 2 sources (Hacker News "Who is hiring", RemoteOK) into a
  bronze/silver Postgres schema with content-hash and embedding dedup,
  salary parsing across six currencies, and idempotent re-processing from
  raw payloads; 466 postings collected, 34 of 400 malformed comments per
  fetch detected and counted rather than dropped. *(Phase 1, Phase 3)*

- **Built a retrieval golden set and eval harness** for a pgvector RAG
  system: 58 judged queries, 1,379 candidates pooled from every retriever and
  graded 0-2 from the full posting text; measured that hybrid search
  underperformed plain vector retrieval (recall@10 0.548 vs 0.653, nDCG@10
  0.555 vs 0.612) and that cross-encoder reranking bought MRR 0.794 at 130x
  the latency, and showed the ranking survives a full corpus rebuild.
  *(Phase 3)* Say, if asked: the grades were made by an LLM reading each
  posting, and a human pass over them is still to do.

- **Built an LLM evaluation suite** (retrieval metrics, LLM-as-judge with a
  Cohen's kappa calibration check, structural refusal detection) that runs in
  GitHub Actions and fails the build on regression; caught a prompt scoping
  defect and fixed it with a versioned prompt, faithfulness 0.69 to 0.75 on
  A/B. *(Phase 5)*

- **Ran a LoRA distillation experiment** (Qwen2.5-0.5B student, llama3.1 8B
  teacher, 242 labelled postings): diagnosed a silent loss-masking failure
  that collapsed the student to 0.000 F1, rebuilt a circular benchmark
  blind, and recovered to 0.485 F1; documented that the 80-rule regex
  baseline (0.663 F1, 0.986 precision, 5 ms) still wins and kept it in
  production. *(Phase 6)*

- **Containerised and instrumented the service**: multi-stage Docker build
  (API image 577MB compressed, torch-free UI image 159MB), structured JSON
  logs, Prometheus request histograms and LLM cost counters, a provisioned
  Grafana dashboard with three alert rules, a CI/CD pipeline that gates the
  image build on lint, tests and the retrieval eval, and a runbook.
  *(Phase 7)*

## Bullets from the plan that cannot be claimed yet, and why

- *"Improved retrieval quality from X% to Y% recall@10 by tuning chunking,
  hybrid search, and reranking."* The honest version is above: the best
  configuration is the plain one (whole-posting vector search, recall@10
  0.653 against keyword's 0.402), hybrid and section chunking both lowered
  recall, and reranking only wins MRR. The earlier 15-query figures (section
  recall@10 0.813, reranker MRR 0.900) were graded from snippets and fell on
  the 58-query set; do not quote them. There is no single X to Y.

- *"Fine-tuned a small open model with LoRA, matching GPT/Claude API accuracy
  at a fraction of the per-request cost."* No API model was in the
  comparison (no key configured) and the student did not match the teacher.
  The claimable version is the Phase 6 bullet above.

- *"Deployed on cloud with Docker, CI/CD, and monitoring; served real users
  with p95 latency under Z ms."* Not deployed; no real users. Request
  latency is measured (there is a histogram and a dashboard for it) but
  only on one laptop under synthetic traffic, which is not a service-level
  number. Claim the containerisation and pipeline, not the deployment,
  until the public URL exists.

## Interview stories these bullets open

Each bullet has a devlog entry behind it with a specific thing that broke.
The ones that land best in an interview:

1. The base64 of my own IP address was the most discriminative token in the
   corpus. *(Phase 2 devlog)*
2. The hybrid search that had been vector-only for weeks because Postgres
   ANDs search terms. *(Phase 3 devlog)*
3. The refusal metric I rewrote three times before it stopped agreeing with
   the model. *(Phase 5 devlog)*
4. Zero of 316 positions masked: the flag that was silently ignored. *(Phase
   6 experiments)*
5. Precision of exactly 1.000 was an identity, not a result. *(Phase 6
   devlog)*
6. `docker build .` builds the last stage, and my API image was the UI.
   *(Phase 7 devlog)*
7. A test run wiped the dev corpus fifteen minutes after the golden set was
   scored, and the rebuild became a robustness test: retrieval survived,
   the salary importance ranking did not. *(2026-09-26 devlog)*
8. Four times the queries, graded from full text, moved every retrieval
   number down: the 15-query table had been graded from the retriever's own
   snippets. *(2026-09-26 experiments)*
