# What scraping 468 job postings taught me about messy data

*Draft for publication. The plan for this project called the post "What
scraping 10k job postings taught me about messy data". The corpus is 468
postings from two sources as of 2026-09-22, because that is what two free
boards yield in a few weeks and I would rather quote the real number.
Everything below is from the devlog and `docs/experiments.md` in
[JobLens](https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant).*

---

Kaggle CSVs are clean. That is what makes them useless for learning data
engineering. Every one of the lessons below came from a posting that a
human would have read without noticing anything wrong, and a program did
something quietly stupid with.

## 1. The most discriminative token in the corpus was my own IP address

The classic ML phase clustered postings with TF-IDF and k-means. The first
run produced a beautiful 99-posting cluster whose defining terms were
`applicants`, `rmjcuni4xmjgumja5`, `read`, `word`, `human`.

That second token is base64. Decoded, it is the IP address of my scraper.
RemoteOK appends an anti-bot canary to every posting it serves, something
like "mention the word FUTURESTIC and tag <token> when applying", and the
token is derived from whoever fetched the page. So it was identical across
every RemoteOK posting and absent from every Hacker News one, which makes it
the single most discriminative term a clustering algorithm could hope for.
k-means found it in seconds and handed back a cluster meaning "came from
RemoteOK".

A second cluster had learned which applicant tracking system employers use,
because 329 of 465 postings contain a URL and the URLs cluster by vendor.

Nothing errored. The output looked plausible. I only caught it because one
top term was unpronounceable, and "a human noticed a weird word" is not a
QA strategy. After stripping the canary and the URLs, the clusters became
`founding engineer`, `forward deployed engineer`, `ml / ai`, `rails /
toronto`. The silhouette score got slightly worse. Optimising the metric
would have reverted the fix.

**Lesson:** any token that is constant within a source and absent outside
it will dominate any unsupervised method. Check your top terms for things
you cannot pronounce.

## 2. Clean at feature time, not at ingestion

The fix for the canary could have gone into the scraper. It went into a
`strip_noise()` function that runs when features are built, and the raw
description stays exactly as the board published it.

The storage is two layers. `raw_postings` keeps the payload the source sent,
one row per fetch, history retained. `postings` holds my interpretation of
it. When the next piece of boilerplate turns up, or the salary parser turns
out to be wrong, `transform` re-runs over the raw layer instead of waiting
two weeks to re-collect. This has already paid for itself once: I found a
Microsoft Word CSS class, `msonormal`, surviving the HTML stripping in 36
postings. It is a parser gap, it is fixable from data I already have, and
nothing was lost in the meantime.

**Lesson:** you do not know yet what is noise. Keep the bytes.

## 3. Hacker News comments have no schema, and 34 in 400 are not postings

The "Who is hiring" thread is the best source of AI startup jobs on the
internet and it is free text in a comment box. Company, role, location,
salary and remote policy appear in any order, separated by pipes, commas or
nothing. Some comments are replies. Some are questions. Some are blog posts
someone pasted.

The parser returns None for anything it cannot recognise as a posting, and
the pipeline counts those as skipped rather than dropping them silently. On
a 400-comment fetch that number is 34. Counting it is the point: if it jumps
to 100 next month, the parser has started eating real postings and I find
out from a number rather than from an empty dashboard.

One of the postings that made it through was a blog post about hiring, not
a job. It reached the fine-tuning test set before anyone noticed.

**Lesson:** a parser's skip count is a metric. Log it, graph it, alert on
its derivative.

## 4. Salary is mostly missing, and guessing is worse than missing

Only 23% of postings state a salary at all. Of those, the text says things
like `$120k-$150k`, `up to 90,000 GBP`, `from €60k`, `$60/hr`, `competitive`,
and `DOE`. The parser handles ranges, `up to`, `from`, hourly and monthly
rates and six currencies, and it recognises the strings that mean "we are
not telling you".

The rule that mattered: hourly and monthly figures are annualised only when
the period is actually stated. A guessed period is how a $60/hr contract ends
up in the data as a $60 salary, or a monthly stipend as an annual one. The
absurd values are dropped with tests that name them: a $1,200 salary that is
a monthly stipend misread, and a $9,000,000 salary that is a phone number.

Then I tried to predict salary from the text. Five models, five-fold cross
validation, 106 postings with a parseable salary. None beat predicting the
median by more than noise. The ridge regression's top features were `money`,
`worth` and `meaningful`, which is recruiter prose, not signal. There is no
salary prediction endpoint, and the README says there will not be one until
there are roughly 500 salary-disclosing postings.

Nine days later, at 109 rows, the ridge beat the median by 10.5%, and it
is still wrong by about $60k on an average posting. The more useful lesson
came from asking which input mattered. Permutation importance on one 25%
test split said the text, by a wide margin. The corpus was then rebuilt from
the boards, one priced row fewer and the rest mostly the same, and the same
command said the source column.
A 25% split of 110 rows is 28 postings; the ranking had never been measured.
Scored across all five folds, the text is first again, with an error bar
from about 4k to 25k and every other column indistinguishable from zero.

**Lesson:** the honest output of a salary model on 106 rows is "not enough
data". Ship that. And any ranking you pull out of that little data needs its
spread across folds printed next to it, or one re-ingest will reverse it.

## 5. Locations are a lookup table, and 37% are unusable

"London, UK", "Remote (US)", "SF/NYC", "anywhere", "EMEA", "Berlin or
remote". A geocoder would happily resolve most of these to a point on a map
and be wrong about what the posting meant. Instead, region is the last
comma-separated part of the string, normalised against a small table, and
37% of postings end up as "unknown". Every share on the trends page is
reported against the postings that could answer the question: 65% are
remote, but only 23% state a salary, and those are not the same postings.

**Lesson:** a coverage figure next to every aggregate is the difference
between a dashboard and a misleading one.

## 6. Two dedup methods that disagree in both directions

Exact duplicates were caught with a hash of normalised title, company and
location, with seniority words removed. Twenty-six pairs. Later, embedding
similarity on the description at a 0.93 threshold also found 26 pairs. Only
19 overlap. Each method finds 7 the other misses, because the hash reads the
header and the embedding reads the body, and reposts change one or the
other. Both are kept.

**Lesson:** when two reasonable methods disagree, the disagreement is
information about your data, not a bug to resolve by picking one.

## 7. Never key anything on an auto-increment id

The test suite truncates the development database. Every re-ingest into an
empty table renumbers every posting. Anything that stored a `posting.id`
across runs, the retrieval golden set, the fine-tuning labels, a resume
pointer for a labelling job, was silently wrong after the next `pytest`.
The labelling job once reported "-242 remaining".

Everything now keys on `(source, source_id)`, the identifier the board
itself assigns, and the pipeline upserts on that pair so a re-run inserts
nothing twice.

**Lesson:** your primary key is an implementation detail. The source's key
is the identity.

## 8. Word boundaries are not the same in every language

The skill extractor is a regex over 80 canonical skills with word
boundaries. On a Portuguese posting it matched `excel` inside `Excelência`,
because `\b` is ASCII-aware and an accented letter reads as a boundary. The
match was invisible for weeks because the benchmark had been built in a way
that could not fail the regex. When the benchmark was rebuilt blind, the
regex's precision went from a perfect 1.000 to 0.986, and that 0.014 was
the one real false positive.

**Lesson:** a perfect score is a reason to check the scorer, not to
celebrate.

## What the corpus is, honestly

468 postings from two boards, 79% of them Hacker News comments, which
over-represents startups and US remote work. Any "the market wants X" claim
from this data is really "these two boards wanted X this month". That
sentence is in the README's limitations section and stays there until the
corpus earns its removal.
