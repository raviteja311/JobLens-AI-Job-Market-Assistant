# JobLens-AI-Job-Market-Assistant


Collects real AI/ML job postings daily, then offers semantic search,
resume matching, grounded chat with citations, and a live analytics
dashboard.

Status: Phase 0 complete.

## Development process

Every phase ends with three things: a merged PR, an updated section in
this README, and a note in `docs/devlog.md`.

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Repo structure

```
src/        application code
tests/      pytest suite
scripts/    entry points and one-off jobs
docs/       devlog, experiments
```

## Phases

### Phase 0: Setup and habits

Repo structure, pyproject.toml, pre-commit with ruff and black, project
board with phases as milestones.

### Phase 1: Data ingestion pipeline

<!-- Sources used, schema, cleaning and dedup approach, postings/day. -->

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

<!-- Kept current and honest: what the system cannot do, where data is thin. -->