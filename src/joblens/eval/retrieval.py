"""Scoring the retrievers against the golden set.

Every configuration runs the same queries through the same code path that
`/search` uses. If the eval called the retrievers directly instead of going
through `retrieval.search`, it would be scoring a system that no user ever
touches, which is the most common way an eval harness ends up lying.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from joblens import db
from joblens.eval import golden, metrics
from joblens.search import retrieval
from joblens.search.embeddings import get_embedder

# The configurations the README table compares.
CONFIGURATIONS = (
    {"name": "keyword", "mode": "keyword", "strategy": "whole", "rerank": False},
    {"name": "vector (whole)", "mode": "vector", "strategy": "whole", "rerank": False},
    {
        "name": "vector (section)",
        "mode": "vector",
        "strategy": "section",
        "rerank": False,
    },
    {"name": "hybrid (whole)", "mode": "hybrid", "strategy": "whole", "rerank": False},
    {
        "name": "hybrid (section)",
        "mode": "hybrid",
        "strategy": "section",
        "rerank": False,
    },
    {"name": "hybrid + rerank", "mode": "hybrid", "strategy": "whole", "rerank": True},
)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: an observed latency, never an interpolated
    one, so p95 over 25 queries is the 24th slowest and not a blend."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class ConfigScore:
    name: str
    scores: dict[str, float]
    queries: int
    seconds: float
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    # Wall-clock per query. The mean hides the tail, and the tail is what the
    # Phase 3 decision rule is written against (p95 under 500 ms).
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def ms_per_query(self) -> float:
        return self.seconds * 1000 / self.queries if self.queries else 0.0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 95)


@dataclass
class RetrievalReport:
    configs: list[ConfigScore]
    queries: int
    corpus: int
    # Where the judgements came from, printed above the table. Anything but
    # the human-verified golden set is a provisional number.
    judged_by: str = "human-verified golden set"

    @property
    def best(self) -> ConfigScore:
        return max(self.configs, key=lambda c: c.scores.get("ndcg@10", 0.0))

    def as_table(self, columns=("recall@5", "recall@10", "mrr", "ndcg@10")) -> str:
        lines = [
            f"{self.queries} judged queries over {self.corpus} postings,"
            f" judged by: {self.judged_by}",
            "",
            "| configuration | " + " | ".join(columns) + " | p50 ms | p95 ms |",
            "| --- | " + " | ".join("---:" for _ in columns) + " | ---: | ---: |",
        ]
        for config in self.configs:
            cells = " | ".join(f"{config.scores.get(c, 0.0):.3f}" for c in columns)
            lines.append(
                f"| {config.name} | {cells} | {config.p50_ms:.0f}"
                f" | {config.p95_ms:.0f} |"
            )
        return "\n".join(lines)


def run(
    queries: list[golden.GoldenQuery] | None = None,
    configurations=CONFIGURATIONS,
    limit: int = 10,
    judged_by: str = "human-verified golden set",
) -> RetrievalReport:
    """Score every configuration on every judged query."""
    queries = queries if queries is not None else golden.verified_only(golden.load())
    if not queries:
        raise ValueError(
            "no verified golden queries. Judge a candidate pool with "
            "`python scripts/golden_pool.py` and merge the grades with "
            "`python scripts/golden_apply.py` first: an eval with nothing to "
            "compare against is worse than no eval."
        )

    embedder = get_embedder()
    # Warm both models before the clock starts. Without this the first
    # configuration that needs a model pays two seconds of load time and the
    # ms/query column says the bi-encoder is 80x slower than keyword search,
    # which is an artefact of measurement order and not true.
    embedder.encode(["warm up"])
    for config in configurations:
        if config["rerank"]:
            from joblens.search.rerank import DEFAULT, _load

            settings = config.get("rerank_config") or DEFAULT
            _load(settings.model, settings.max_length)

    configs: list[ConfigScore] = []
    with db.connect() as conn:
        corpus = conn.execute("select count(*) as n from postings").fetchone()["n"]
        judged = golden.resolve(conn, queries)

        for config in configurations:
            per_query: dict[str, dict[str, float]] = {}
            latencies: list[float] = []
            began = time.perf_counter()
            for query in queries:
                relevant = judged.get(query.id, {})
                if not relevant:
                    continue
                query_began = time.perf_counter()
                hits = retrieval.search(
                    conn,
                    query.query,
                    mode=config["mode"],
                    embedder=embedder,
                    strategy=config["strategy"],
                    limit=limit,
                    rerank=config["rerank"],
                    rerank_config=config.get("rerank_config"),
                )
                latencies.append((time.perf_counter() - query_began) * 1000)
                per_query[query.id] = metrics.score_run(
                    [h.posting_id for h in hits], relevant, ks=(5, limit)
                )
            elapsed = time.perf_counter() - began
            configs.append(
                ConfigScore(
                    name=config["name"],
                    scores=metrics.mean_scores(list(per_query.values())),
                    queries=len(per_query),
                    seconds=elapsed,
                    per_query=per_query,
                    latencies_ms=latencies,
                )
            )
    return RetrievalReport(
        configs=configs, queries=len(queries), corpus=corpus, judged_by=judged_by
    )
