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
        # One suffix covers both ends of the range. Real salary_raw strings
        # that were stored with a floor of 100 and 150.
        ("100-200k CHF", Salary(100000, 200000, "year", "CHF")),
        ("$150 - 210K", Salary(150000, 210000, "year", "USD")),
        # An ISO code after the amount beats the bare "$". Both were stored
        # as USD.
        ("$130,000 – $210,000 CAD", Salary(130000, 210000, "year", "CAD")),
        ("$109K–$136K CAD", Salary(109000, 136000, "year", "CAD")),
        # "AU$" contains neither "A$" nor "C$"; it was read as a bare "$".
        ("AU$120–160k", Salary(120000, 160000, "year", "AUD")),
        ("CA$90,000", Salary(90000, 90000, "year", "CAD")),
        # Real salary_raw strings whose "/hour" the pattern did not read; they
        # were only stored as hourly through the guess that no longer exists.
        ("$90–$110/hour USD", Salary(90, 110, "hour", "USD")),
        ("$23-$34 USD/hour", Salary(23, 34, "hour", "USD")),
        ("50 / hour", Salary(50, 50, "hour", None)),
        ("€6,000 / month", Salary(6000, 6000, "month", "EUR")),
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


def test_a_bare_small_number_is_not_annualised_on_a_guess():
    # It used to be called hourly and annualised: "65" became 135,200 a year.
    salary = parse_salary("65")
    assert salary == Salary(65, 65, None, None)
    assert salary.annualised() == (None, None)
    assert parse_salary("$60 - $80").annualised() == (None, None)
    # A stated period still converts.
    assert parse_salary("65 per hour").annualised() == (65 * 2080, 65 * 2080)


def test_a_bare_figure_in_the_thousands_is_annual():
    # "year" multiplies by 1, so reading it as annual invents nothing.
    assert parse_salary("130000") == Salary(130000, 130000, "year", None)
    assert parse_salary("130000").annualised() == (130000, 130000)


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
