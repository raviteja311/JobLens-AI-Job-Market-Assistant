"""The India and target-role filters the board sources apply in to_posting()."""

import pytest

from joblens.sources.filters import is_india, is_target_role


@pytest.mark.parametrize(
    "location, expected",
    [
        ("Bengaluru, Karnataka, India", True),
        ("Remote - India", True),
        ("Remote (India)", True),
        ("Hyderabad", True),
        ("Bangalore, India", True),
        ("Pune, Maharashtra", True),
        ("Gurgaon", True),
        ("New Delhi", True),
        ("Trivandrum", True),
        # Seen on Razorpay's board with no state or country attached.
        ("Lucknow", True),
        ("London; Bengaluru", True),
        ("BENGALURU", True),
        # The cases the word boundaries exist for.
        ("Indiana", False),
        ("Indianapolis, IN", False),
        ("Fort Wayne, Indiana, United States", False),
        # Plain remote usually means US-remote.
        ("Remote", False),
        ("Remote - US", False),
        ("London", False),
        ("Malaysia", False),
        # Hyderabad is also a city in Pakistan.
        ("Hyderabad, Pakistan", False),
        ("", False),
        (None, False),
    ],
)
def test_is_india(location, expected):
    assert is_india(location) is expected


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Data Scientist - Evaluations, Chanakya", True),
        ("Data Analyst", True),
        ("Analytics Manager", True),
        ("Machine Learning Engineer", True),
        ("ML Ops Engineer, Chanakya", True),
        ("MLOps Engineer", True),
        ("AI Solution Engineer", True),
        ("Senior AI/ML Engineer", True),
        ("NLP Researcher", True),
        ("LLM Engineer", True),
        ("GenAI Developer", True),
        ("Gen AI Engineer", True),
        ("Python Developer", True),
        ("Business Intelligence Analyst", True),
        ("BI Developer", True),
        ("Data Engineer II", True),
        ("Research Scientist, Foundational Models", True),
        ("Computer Vision Engineer", True),
        ("Quantitative Analyst", True),
        # A bare "analyst" is usually a business role, not a data one.
        ("Analyst, Banking Alliances", False),
        ("Analyst, Risk Management", False),
        # Data in the title, but not the work this corpus is about.
        ("Data Entry Operator", False),
        ("Data Center Technician", False),
        ("Account Executive, AI Products", False),
        ("Sales Engineer - Data Platform", False),
        # Word boundaries again: these contain "ai" or "ml" as letters only.
        ("Email Marketing Specialist", False),
        ("HTML Developer", False),
        ("Product Designer", False),
        ("DevOps Enginer", False),
        ("", False),
        (None, False),
    ],
)
def test_is_target_role(title, expected):
    assert is_target_role(title) is expected
