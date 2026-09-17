from joblens import cleaning
from joblens.cleaning import content_hash, is_remote, normalise_location, strip_html


def test_strips_tags_and_keeps_paragraph_breaks():
    html = "<p>We build things.</p><ul><li>Python</li><li>SQL</li></ul>"
    text = strip_html(html)
    assert "<p>" not in text
    assert "We build things." in text
    assert "Python" in text and "SQL" in text


def test_drops_script_contents():
    assert "trackPixel" not in strip_html("<p>Hi</p><script>trackPixel();</script>")


def test_unescapes_entities():
    assert strip_html("<p>R&amp;D team, &pound;90k</p>").startswith("R&D team")


def test_handles_empty_and_none():
    assert strip_html(None) == ""
    assert strip_html("") == ""


def test_remote_detection():
    assert is_remote("Remote (Europe)")
    assert is_remote(None, "Work from home")
    assert not is_remote("London, UK")


def test_location_normalisation():
    assert normalise_location("Remote (US)") == "United States"
    assert normalise_location("NYC") == "New York, United States"
    assert normalise_location("London, UK") == "London, UK"
    assert normalise_location("Remote") is None
    assert normalise_location(None) is None


def test_unknown_locations_pass_through_unchanged():
    # Better a messy string than a confidently wrong one.
    assert normalise_location("Kraków, Poland") == "Kraków, Poland"


def test_same_job_on_two_boards_hashes_the_same():
    a = content_hash("Senior ML Engineer", "Acme Corp", "London, UK")
    b = content_hash("ML Engineer", "acme corp.", "London, UK")
    assert a == b


def test_different_companies_do_not_collide():
    a = content_hash("ML Engineer", "Acme Corp", "London, UK")
    b = content_hash("ML Engineer", "Globex", "London, UK")
    assert a != b


REMOTEOK_CANARY = (
    "We need a Rust engineer. Apply at https://remoteok.com/l/123456 today.\n\n"
    "Please mention the word **FUTURESTIC** and tag RMjcuNi4xMjguMjA5 when "
    "applying to show you read the job post completely (#RMjcuNi4xMjguMjA5). "
    "This is a beta feature to avoid spam applicants. Companies can search "
    "these words to find applicants that read this and see they're human."
)


def test_strip_noise_removes_the_remoteok_canary():
    cleaned = cleaning.strip_noise(REMOTEOK_CANARY)
    assert "RMjcuNi4xMjguMjA5" not in cleaned
    assert "FUTURESTIC" not in cleaned
    assert "beta feature" not in cleaned
    assert "spam applicants" not in cleaned


def test_strip_noise_keeps_the_actual_job():
    assert "Rust engineer" in cleaning.strip_noise(REMOTEOK_CANARY)


def test_strip_noise_removes_urls():
    cleaned = cleaning.strip_noise("Apply at https://jobs.ashbyhq.com/acme/123 now")
    assert "ashbyhq" not in cleaned
    assert cleaned == "Apply at now"


def test_strip_noise_handles_nothing():
    assert cleaning.strip_noise(None) == ""
    assert cleaning.strip_noise("") == ""
