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


# Indian salary formats. A lakh is 100,000 and a crore 10,000,000, and LPA
# ("lakhs per annum") states the period and the currency in three letters.
@pytest.mark.parametrize(
    "text, expected",
    [
        # The plan's table.
        ("₹6-10 LPA", Salary(600000, 1000000, "year", "INR")),
        ("6 - 10 LPA", Salary(600000, 1000000, "year", "INR")),
        ("8 lakhs per annum", Salary(800000, 800000, "year", "INR")),
        ("8 L p.a.", Salary(800000, 800000, "year", "INR")),
        ("₹1.2 Cr", Salary(12000000, 12000000, "year", "INR")),
        ("₹50,000 per month", Salary(50000, 50000, "month", "INR")),
        # Spellings and shapes Indian postings use.
        ("12 LPA", Salary(1200000, 1200000, "year", "INR")),
        ("12.5 LPA", Salary(1250000, 1250000, "year", "INR")),
        ("10LPA", Salary(1000000, 1000000, "year", "INR")),
        ("INR 12-18 LPA", Salary(1200000, 1800000, "year", "INR")),
        ("₹12 LPA - ₹18 LPA", Salary(1200000, 1800000, "year", "INR")),
        ("CTC: 12 to 18 LPA", Salary(1200000, 1800000, "year", "INR")),
        ("5 lacs", Salary(500000, 500000, "year", "INR")),
        ("4 lac per annum", Salary(400000, 400000, "year", "INR")),
        ("5-8 Lakhs", Salary(500000, 800000, "year", "INR")),
        ("8 lakh", Salary(800000, 800000, "year", "INR")),
        ("Rs. 8 lakh", Salary(800000, 800000, "year", "INR")),
        ("Rs 50,000 per month", Salary(50000, 50000, "month", "INR")),
        ("₹10 L - ₹15 L p.a.", Salary(1000000, 1500000, "year", "INR")),
        ("1.5 lakh per month", Salary(150000, 150000, "month", "INR")),
        ("₹1.2 Cr - 1.5 Cr", Salary(12000000, 15000000, "year", "INR")),
        ("1 crore", Salary(10000000, 10000000, "year", "INR")),
        ("Up to 20 LPA", Salary(None, 2000000, "year", "INR")),
        ("upto 12 LPA", Salary(None, 1200000, "year", "INR")),
        ("12 LPA + ESOPs", Salary(1200000, 1200000, "year", "INR")),
        ("Competitive, 10-15 LPA", Salary(1000000, 1500000, "year", "INR")),
    ],
)
def test_parses_indian_formats(text, expected):
    assert parse_salary(text) == expected


def test_a_monthly_rupee_salary_annualises():
    assert parse_salary("₹50,000 per month").annualised() == (600000, 600000)
    assert parse_salary("1.5 lakh per month").annualised() == (1800000, 1800000)


def test_a_bare_l_is_not_a_lakh():
    # Only "L p.a." is read as lakhs. A lone L is too ambiguous to multiply.
    assert parse_salary("10L") == Salary(10, 10, None, None)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Experience: 3-5 years, 10-15 LPA", Salary(1000000, 1500000, "year", "INR")),
        ("0-1 years, 3.5 LPA", Salary(350000, 350000, "year", "INR")),
        ("5+ yrs, $150,000", Salary(150000, 150000, "year", "USD")),
        ("2 to 4 years experience", Salary()),
    ],
)
def test_experience_is_not_read_as_pay(text, expected):
    assert parse_salary(text) == expected
