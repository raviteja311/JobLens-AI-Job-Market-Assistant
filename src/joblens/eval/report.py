"""The scorecard, the history, and the gate that fails CI.

The gate is the part that matters. An eval that prints numbers is a report;
an eval that fails the build is a test. Thresholds live here in code rather
than in the workflow file, so changing one shows up in a pull request diff
next to the change that needed it.

History goes to a CSV as well as the database. The CSV is what the README
chart reads and what survives someone dropping the database; the table is
what the dashboard queries.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from joblens import db

HISTORY_CSV = Path(__file__).resolve().parents[3] / "data" / "eval_history.csv"
HISTORY_FIELDS = (
    "timestamp",
    "suite",
    "config",
    "git_sha",
    "queries",
    "metric",
    "value",
)

# A drop bigger than this fails the build. Not zero: these numbers move by a
# point or two between runs on a corpus that grows daily, and a gate that
# fires on noise gets disabled within a week.
REGRESSION_TOLERANCE = 0.05

# Absolute floors. Below these the feature is broken regardless of trend.
#
# faithfulness is deliberately NOT here. It comes from the LLM judge, and the
# judge is uncalibrated (see docs/experiments.md), so gating the build on it
# would mean blocking merges on a number the repo itself says not to trust.
# It is still recorded on every run, and it becomes a floor the day
# `joblens calibrate` reports a usable kappa.
#
# refusal_accuracy is gated, because it is a deterministic string check over
# questions the corpus cannot answer, and a system that invents answers to
# those is broken in the way that matters most.
FLOORS = {
    "retrieval": {"ndcg@10": 0.45, "recall@10": 0.55},
    "chat": {"refusal_accuracy": 0.75},
}


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - running outside a checkout is fine
        return "unknown"


def record(
    suite: str, config: str, queries: int, metrics: dict, note: str = ""
) -> None:
    """Append one run to both the table and the CSV."""
    sha = git_sha()
    try:
        with db.connect() as conn:
            conn.execute(
                """
                insert into eval_runs (suite, config, git_sha, queries, metrics, note)
                values (%s, %s, %s, %s, %s, %s)
                """,
                (suite, config, sha, queries, json.dumps(metrics), note),
            )
            conn.commit()
    except Exception:  # noqa: BLE001 - the CSV is the copy that must not fail
        pass

    # One row per metric, not one row per run. The first version used a wide
    # layout, which works exactly until the second suite writes to it: the
    # header was fixed by whichever run created the file, so chat's
    # faithfulness landed in retrieval's mrr column and the file read as if
    # retrieval had regressed. A long layout has no shared header to disagree
    # about and charts just as well.
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_CSV.exists()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        for metric, value in sorted(metrics.items()):
            writer.writerow(
                {
                    "timestamp": stamp,
                    "suite": suite,
                    "config": config,
                    "git_sha": sha,
                    "queries": queries,
                    "metric": metric,
                    "value": round(float(value), 4),
                }
            )


def previous(suite: str, config: str) -> dict | None:
    """The last recorded run for this suite and config, if there is one."""
    try:
        with db.connect() as conn:
            row = conn.execute(
                """
                select metrics from eval_runs
                 where suite = %s and config = %s
                 order by created_at desc limit 1
                """,
                (suite, config),
            ).fetchone()
        return row["metrics"] if row else None
    except Exception:  # noqa: BLE001
        return None


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    checked: list[str]

    def summary(self) -> str:
        if self.passed:
            return f"eval gate passed ({len(self.checked)} checks)"
        return "eval gate FAILED\n  " + "\n  ".join(self.failures)


def gate(
    suite: str, config: str, metrics: dict, baseline: dict | None = None
) -> GateResult:
    """Fail on an absolute floor, or on a regression against the last run."""
    baseline = baseline if baseline is not None else previous(suite, config)
    failures: list[str] = []
    checked: list[str] = []

    for metric, floor in FLOORS.get(suite, {}).items():
        if metric not in metrics:
            continue
        checked.append(f"{metric} >= {floor}")
        if metrics[metric] < floor:
            failures.append(
                f"{metric} is {metrics[metric]:.3f}, below the floor of {floor}"
            )

    if baseline:
        for metric, value in metrics.items():
            if metric not in baseline or metric == "cost_usd":
                continue
            checked.append(f"{metric} vs last run")
            drop = baseline[metric] - value
            if drop > REGRESSION_TOLERANCE:
                failures.append(
                    f"{metric} dropped {drop:.3f} "
                    f"({baseline[metric]:.3f} -> {value:.3f}), "
                    f"tolerance is {REGRESSION_TOLERANCE}"
                )

    return GateResult(passed=not failures, failures=failures, checked=checked)


def history(
    suite: str | None = None, metric: str | None = None, limit: int = 100
) -> list[dict]:
    if not HISTORY_CSV.exists():
        return []
    with HISTORY_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if suite:
        rows = [r for r in rows if r["suite"] == suite]
    if metric:
        rows = [r for r in rows if r["metric"] == metric]
    return rows[-limit:]


def trend_table(suite: str, metric: str, limit: int = 10) -> str:
    rows = history(suite, metric, limit)
    if not rows:
        return f"no recorded runs for {suite}"
    lines = [
        f"| run | {metric} | config | sha |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        value = row["value"]
        lines.append(
            f"| {row['timestamp'][:16]} | {value} "
            f"| {row['config']} | {row['git_sha']} |"
        )
    return "\n".join(lines)
