import pandas as pd
import pytest

from joblens.ml import dataset
from joblens.models import Posting


def test_region_takes_the_last_comma_separated_part():
    assert dataset.region("London, United Kingdom") == "United Kingdom"
    assert dataset.region("Berlin") == "Berlin"


def test_short_codes_keep_their_case():
    assert dataset.region("Austin, TX") == "TX"
    assert dataset.region("London, UK") == "UK"
    assert dataset.region("san francisco, ca") == "CA"


def test_region_of_nothing_is_unknown():
    assert dataset.region(None) == "unknown"
    assert dataset.region("   ") == "unknown"
    # A missing location arrives from pandas as NaN, which is truthy.
    assert dataset.region(float("nan")) == "unknown"


def test_seniority_buckets():
    assert dataset.seniority("Senior ML Engineer") == "senior"
    assert dataset.seniority("Staff Research Scientist") == "senior"
    assert dataset.seniority("ML Intern") == "junior"
    assert dataset.seniority("Graduate Data Scientist") == "junior"
    assert dataset.seniority("Machine Learning Engineer") == "mid"


def test_frame_from_postings_has_every_column():
    posting = Posting.build(
        source="remoteok",
        source_id="1",
        title="ML Engineer",
        company="Acme",
        url="https://example.com/1",
        location="London, UK",
        salary_raw="$120,000 - $150,000 per year",
        description_html="<p>PyTorch</p>",
    )
    frame = dataset.frame_from_postings([posting])
    assert list(frame.columns) == list(dataset.COLUMNS)
    assert frame.loc[0, "salary_min_year"] == 120_000
    assert frame.loc[0, "description"] == "PyTorch"


def test_normalise_coerces_string_salaries_and_dates():
    frame = dataset.frame_from_postings([])
    assert frame.empty


def test_with_region_adds_one_column(corpus):
    assert "region" in dataset.with_region(corpus).columns
    assert len(dataset.with_region(corpus)) == len(corpus)


def test_require_rows_says_what_is_missing():
    with pytest.raises(ValueError, match="needs at least 30 rows, got 1"):
        dataset.require_rows(pd.DataFrame({"a": [1]}), 30, "salary regression")
