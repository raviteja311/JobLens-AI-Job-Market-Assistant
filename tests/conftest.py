"""Shared fixtures.

The ML tests need a corpus, and the real one lives in Postgres and changes
every day, which makes it useless for assertions. So they get a synthetic one
built here: four job families with genuinely different pay, so a model that
has learned nothing scores visibly worse than one that has. The numbers are
made up, the shape of the data is not.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

# family -> (skill sentence, base annual USD)
FAMILIES = {
    "Machine Learning Engineer": (
        "Work with PyTorch and TensorFlow on deep learning models. "
        "Deploy with Docker and Kubernetes on AWS. Python and SQL required.",
        165_000,
    ),
    "LLM Engineer": (
        "Build RAG systems over a vector database using LangChain and "
        "Hugging Face transformers. Prompt engineering and fine-tuning with "
        "LoRA. Python, FastAPI, embeddings.",
        190_000,
    ),
    "Data Engineer": (
        "Own pipelines in Spark, Airflow and Kafka landing into Snowflake. "
        "Python, SQL, dbt, Terraform and GCP.",
        145_000,
    ),
    "Data Analyst": (
        "Build dashboards in Tableau and Power BI. Strong SQL and Excel, "
        "some Python. Statistics and a/b testing.",
        85_000,
    ),
}

SENIORITY = {"Junior": 0.7, "": 1.0, "Senior": 1.35}
REGIONS = ["London, United Kingdom", "New York, United States", "Berlin, Germany", None]
SOURCES = ["remoteok", "hackernews", "adzuna"]


def build_corpus(n: int = 160, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        family = rng.choice(list(FAMILIES))
        description, base = FAMILIES[family]
        prefix = rng.choice(list(SENIORITY))
        salary = base * SENIORITY[prefix] * rng.uniform(0.9, 1.1)
        location = rng.choice(REGIONS)
        rows.append(
            {
                "id": i,
                "source": rng.choice(SOURCES),
                "source_id": str(i),
                "title": f"{prefix} {family}".strip(),
                "company": f"Company {i % 37}",
                "location": location,
                "is_remote": location is None or rng.random() < 0.3,
                "salary_raw": f"${salary:,.0f} per year",
                # A third of postings never say, which is roughly what the real
                # boards do and keeps the coverage assertions honest.
                "salary_min_year": None if i % 3 == 0 else salary * 0.92,
                "salary_max_year": None if i % 3 == 0 else salary * 1.08,
                "salary_currency": "USD",
                "description": description,
                "url": f"https://example.com/{i}",
                "posted_at": now - timedelta(days=rng.randint(0, 120)),
                "content_hash": f"hash{i}",
                "first_seen_at": now,
            }
        )
    frame = pd.DataFrame(rows)
    frame["posted_at"] = pd.to_datetime(frame["posted_at"], utc=True)
    frame["first_seen_at"] = pd.to_datetime(frame["first_seen_at"], utc=True)
    return frame


@pytest.fixture(scope="session")
def corpus() -> pd.DataFrame:
    return build_corpus()
