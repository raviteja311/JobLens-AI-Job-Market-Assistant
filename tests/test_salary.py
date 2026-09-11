import pytest

from joblens.salary import Salary, parse_salary


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$120,000 - $150,000", Salary(120000, 150000, "year", "USD")),
        ("£90,000 to £110,000 per annum", Salary(90000, 110000, "year", "GBP")),
        ("120k-150k", Salary(120000, 150000, "year", None)),
        ("$60/hr", Salary(60, 60, "hour", "USD")),
        ("$45 per hour", Salary(45, 45, "hour", "USD")),
        ("€75,000 per year", Salary(75000, 75000, "year", "EUR")),
        ("Up to £90,000", Salary(None, 90000, "year", "GBP")),
        ("From $130,000", Salary(130000, None, "year", "USD")),
        ("₹18,00,000 per annum", Salary(1800000, 1800000, "year", "INR")),
        ("USD 140000", Salary(140000, 140000, "year", "USD")),
        ("$8,000 per month", Salary(8000, 8000, "month", "USD")),
    ],
)
def test_parses_common_shapes(text, expected):
    assert parse_salary(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "Competitve salary",
        "Salary negotiable, DOE",
        "Depending on experience",
        "Great benefits and unlimited PTO",
    ],
)
def test_returns_nothing_when_there_is_no_salary(text):
    result = parse_salary(text)
    assert not result.parsed
    assert result.min is None and result.max is None


def test_equity_tail_is_ignored():
    result = parse_salary("$150,000 + 0.5%  equity")
    assert result.min == 150000
    assert result.max == 150000


def test_bare_small_number_is_treated_as_hourly():
    assert parse_salary("65").period == "hour"
    assert parse_salary("130000").period == "year"


def test_reversed_range_is_corrected():
    assert parse_salary("$150,000 - $120,000") == Salary(120000, 150000, "year", "USD")


def test_annulises_hourly_rate():
    low, high = parse_salary("$60 - $80 per hour").annualised()
    assert low == 60 * 2080
    assert high == 80 * 2080


def test_does_not_annualise_when_period_is_unknown():
    salary = Salary(100, 200, None, "USD")
    assert salary.annualised() == (None, None)


def test_canadian_dollars_are_not_us_dollars():
    assert parse_salary("C$110,000 per year").currency == "CAD"
