# Resume bullets, with the real numbers

The project plan listed five bullets to fill in "with your real numbers".
Here they are filled in, and where the real number does not exist yet the
bullet says so rather than borrowing one. Every figure links back to the
README section or experiment that produced it. Dates are as of 2026-09-22.

## Bullets that are true today

- **Built an end-to-end job intelligence platform** ingesting AI/ML postings
  daily from 2 sources (Hacker News "Who is hiring", RemoteOK) into a
  bronze/silver Postgres schema with content-hash and embedding dedup,
  salary parsing across six currencies, and idempotent re-processing from
  raw payloads; 468 postings collected, 34 of 400 malformed comments per
  fetch detected and counted rather than dropped. *(Phase 1, Phase 3)*

- **Built a retrieval golden set and eval harness** for a pgvector RAG
  system, 60 queries with 15 hand-judged on a 0-2 scale; measured that hybrid
  search underperformed plain vector retrieval (recall@10 0.666 vs 0.692)
  and that cross-encoder reranking raised MRR from 0.773 to 0.900, and
  shipped the configuration the numbers supported rather than the one the
  literature recommended. *(Phase 3)*

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
  hybrid search, and reranking."* The honest version is above: section
  chunking raised recall@10 from 0.692 to 0.813 and reranking raised MRR to
  0.900, but hybrid search lowered recall. Do not claim a single X to Y.

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
