# Experiments

One entry per thing tried: hypothesis, setup, result, decision. Negative
results stay in. Most of the value of this file is the experiments that did
not work, because those are the ones that change what gets built next.

Numbers come from the corpus as it stood on the date of the entry. The corpus
grows daily, so re-running an old experiment will not reproduce its exact
numbers; the decision it led to is the part that has to survive.

---

## 2026-09-17 - Salary regression: nothing beats the median

**Hypothesis.** Job title and description text predict advertised salary well
enough to be worth serving. Baseline is predicting the median of the training
fold; anything shipped has to beat that.

**Setup.** 106 postings with a salary that parsed, annualised and converted to
USD at frozen rates. 5-fold CV, shuffled, seed 42. MAE as the headline metric
because a handful of 750k postings dominate RMSE.

**Result.**

| model | MAE (USD) | RMSE (USD) | R2 | vs median |
| --- | ---: | ---: | ---: | ---: |
| ridge | 62,904 | 88,374 | -0.098 | +1.8% |
| ridge_log | 63,119 | 88,790 | -0.064 | +1.4% |
| median | 64,042 | 86,859 | -0.036 | +0.0% |
| random_forest | 68,563 | 97,385 | -0.340 | -7.1% |
| gradient_boosting | 78,292 | 113,514 | -1.113 | -22.3% |

Every R2 is negative, including the baseline's. That is not a bug: R2 is
computed per fold against that fold's own mean, and with 106 rows drawn from a
20k-750k range the folds disagree about where the centre is.

The ridge's 1.8% edge over the median is inside the noise. Its top features
say so plainly: `money`, `worth`, `kind`, `meaningful`, `world`. Those are
words from recruiter prose, not salary signal. 106 rows against roughly 14,000
TF-IDF columns overfits exactly this way.

**Follow-up sweep.** If the problem is dimensionality, more regularisation or
fewer features should help. It does not help, it converges:

| configuration | MAE (USD) | R2 |
| --- | ---: | ---: |
| median baseline | 64,042 | -0.036 |
| ridge, skills + categorical, alpha=1 | 77,700 | -0.705 |
| ridge, skills + categorical, alpha=10 | 65,942 | -0.138 |
| ridge, skills + categorical, alpha=100 | 64,276 | -0.066 |
| ridge, TF-IDF min_df=5, alpha=10 | 64,640 | -0.096 |
| ridge, TF-IDF min_df=5, alpha=100 | 64,417 | -0.069 |
| categorical features only | 65,352 | -0.147 |

Every configuration walks towards the baseline as the penalty rises and none
of them passes it. A model that can only match the baseline by being
regularised into predicting a constant has not learned anything.

**Decision.** Ship the comparison, not the model. `/trends` will report the
median advertised salary and its spread; there will be no salary prediction
endpoint until there is enough data to earn one. The threshold to revisit is
roughly 500 salary-disclosing postings, which at the current rate of about 25
a day is three weeks of ingestion, or sooner if the Adzuna key gets configured
since Adzuna is the only source with structured salary fields.

**Kept anyway.** The whole comparison stays in the repo and in CI. When the
corpus is four times the size this file gets a second entry, and the
interesting part will be the delta between the two.

---

## 2026-09-17 - Clustering was grouping by scraper, not by job

**Hypothesis.** TF-IDF plus k-means over title and description finds job
families that the job titles themselves do not, because "Member of Technical
Staff" and "ML Engineer" are the same job.

**Setup.** 465 postings, TF-IDF (1-2 grams, min_df=2, max_df=0.6, sublinear),
k-means, k chosen by silhouette over 3..10.

**First result, and the bug.** k=5, silhouette 0.011. Two of the five clusters
were nonsense:

- Cluster of 99 postings whose defining terms were `applicants`,
  `rmjcuni4xmjgumja5`, `read`, `word`, `human`.
- Cluster of 54 whose defining terms were `https jobs`, `jobs ashbyhq`,
  `ashbyhq com`.

The second is URLs: 329 of 465 postings contain at least one link, so the
model learned which applicant tracking system the employer uses. The first
took longer to work out. Every RemoteOK posting ends with:

> Please mention the word \*\*FUTURESTIC\*\* and tag RMjcuNi4xMjguMjA5 when
> applying to show you read the job post completely (#RMjcuNi4xMjguMjA5).
> This is a beta feature to avoid spam applicants.

It is an anti-bot canary, and the token is base64 of the IP address that
fetched the page. Identical across all 99 RemoteOK postings, absent from every
Hacker News one, so it is the single most discriminative term in the corpus.
k-means found it immediately and produced a cluster meaning "came from
RemoteOK". Labelled by top terms, it looked like a real job family.

**Fix.** `cleaning.strip_noise()` removes URLs and known per-board furniture.
It runs at feature time, not at ingestion: `postings.description` keeps what
the board actually published, so the next piece of furniture we find can be
removed from data already collected without re-scraping.

**Second result.** Same corpus, same seed, noise stripped. k moved to 10 and
the clusters became nameable: `founding engineer`, `forward deployed
engineer`, `open source / developer`, `ml / ai`, `san francisco onsite`,
`rails / toronto`.

**The uncomfortable part.** Silhouette did not reward the fix:

| corpus | vocabulary | k=5 | k=10 |
| --- | ---: | ---: | ---: |
| raw | 14,049 | 0.0110 | 0.0065 |
| noise stripped | 13,599 | 0.0065 | 0.0092 |

The clusters got obviously better and the metric moved by 0.005 in the wrong
direction at k=5. Silhouette on sparse high-dimensional text is close to
meaningless at these values, and a change that is unmistakable to a human
reading the labels is invisible to it.

**Decision.** Keep the fix, keep k=10, and stop treating silhouette as
anything more than a tie-breaker for choosing k. This is the concrete argument
for the Phase 3 golden dataset: until there is a hand-labelled set, "did that
change help" is being answered by a number that demonstrably cannot tell.

---

## 2026-09-17 - Skill extraction v1: a dictionary, and what it misses

**Hypothesis.** A hand-written alias table over roughly 80 canonical skills is
enough to power the trends dashboard, and is the baseline Phase 6's
fine-tuned extractor has to beat.

**Setup.** `SKILL_ALIASES` in `src/joblens/ml/skills.py`, case-insensitive
regex with custom word boundaries. Corpus of 465 postings.

**Result.** Coverage, meaning the share of postings where at least one known
skill is found, is **58.9%**. Top of the table:

| skill | postings | share |
| --- | ---: | ---: |
| python | 112 | 24.1% |
| typescript | 83 | 17.8% |
| postgresql | 71 | 15.3% |
| llm | 59 | 12.7% |
| aws | 51 | 11.0% |
| kubernetes | 48 | 10.3% |
| git | 44 | 9.5% |
| terraform | 31 | 6.7% |

**Two bugs found by writing the tests.**

The word boundary. `\b` does not work next to `+` or `#`, so `c++` needed a
custom boundary of "any character that could continue a technology name". The
first version included `.` in that set, which was correct for `node.js` and
wrong for every skill at the end of a sentence: `...and AWS.` stopped matching
`aws`. Silent, and it would have understated every skill count by whatever
fraction of mentions happen to end a sentence. The dot now only blocks when a
letter or digit follows it.

Missing locations. A posting with no location arrives from pandas as NaN, NaN
is truthy, and `str(NaN)` is `"nan"`, so the dashboard grew a region called
`Nan` sitting above most real countries.

**What the 41% gap is.** Spot-checking postings with zero hits: they are
non-technical roles in the Hacker News threads (recruiting, sales, ops), and
postings that describe the work without naming tools. Some of both is correct
behaviour, so coverage should not be read as 41% missed.

**Decision.** Ship it. Do not grow the table by guessing; grow it from
`tfidf_keywords()`, which surfaces terms that are frequent in a posting and
absent from the taxonomy. Coverage is the number Phase 6 reports against.

---

## 2026-09-17 - Retrieval: hybrid lost to plain vector search

**Hypothesis.** Hybrid retrieval beats either half. Keyword search knows
exact terms, vector search knows vague ones, and reciprocal rank fusion keeps
both strengths. This is the received wisdom and the reason the plan called
for it.

**Setup.** 15 hand-judged queries over 465 postings, graded 0/1/2. Judgements
key on (source, source_id) rather than the primary key, so they survive a
re-ingest into an empty database. Every configuration goes through
`retrieval.search`, the same function `/search` calls. Both models warmed
before timing.

**Result.**

| configuration | recall@5 | recall@10 | mrr | ndcg@10 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.299 | 0.358 | 0.406 | 0.328 | 25 |
| vector (whole) | 0.569 | 0.692 | 0.773 | 0.598 | 34 |
| vector (section) | 0.491 | 0.813 | 0.734 | 0.684 | 35 |
| hybrid (whole) | 0.438 | 0.666 | 0.761 | 0.620 | 94 |
| hybrid (section) | 0.430 | 0.646 | 0.736 | 0.600 | 93 |
| hybrid + rerank | 0.497 | 0.725 | 0.900 | 0.692 | 3145 |

Hybrid is worse than vector alone on recall@5, recall@10 and MRR. It is
better on nDCG@10, and only because vector (whole) is unusually weak there.

**Why.** Keyword retrieval here ORs its terms, because ANDing them (which is
what `websearch_to_tsquery` and `plainto_tsquery` both do) returns literally
nothing for a query like "remote machine learning engineer working on LLMs".
OR plus `ts_rank_cd` is the standard fix and it works, but for a six-word
natural-language query it still ranks a posting matching only "engineer"
first. RRF does not care that the keyword arm has no idea what it is doing:
it gives that posting rank 1 and therefore the same 1/(60+1) that the vector
arm's genuinely best result gets.

So fusion is not adding a second opinion, it is adding noise with a vote.

**Where hybrid does win**, and the reason it stays the default: the rare
token. On the query "pgvector", vector search returns Marketing Manager,
Chicago or Washington DC, and ON SITE TORONTO, because an embedding of
"pgvector" is mostly "database". Keyword search returns the one posting that
names it, at rank 1. Hybrid returns that posting first and the vector results
below. A 15-query set with two such queries in it cannot show that as an
average, which is an argument about the golden set and not about the method.

**Decision.** Keep hybrid as the default and stop claiming it is better.
The honest summary is that hybrid is insurance against the query type that
embeddings cannot handle at all, bought at a measurable cost on ordinary
queries. Revisit with a weighted fusion that down-weights the keyword arm
when its top score is low, and measure that instead of assuming it.

**Reranking is the real win.** MRR 0.773 to 0.900 means the first relevant
result moved from about rank 1.3 to about rank 1.1. It costs 3.1 seconds a
query on CPU, which is 90x the first stage, so it is off by default and on
behind a flag.

---

## 2026-09-17 - Chunking: sections win on depth, whole wins on the top

**Hypothesis.** Job postings are short enough to embed whole. Splitting them
by section should not help.

**Setup.** Same 15 queries. 465 postings became 465 whole chunks (1,223 chars
average) or 1,091 section chunks (805 chars average). A posting's score is
its best chunk, not its average, so a long posting is not penalised for the
paragraphs that do not match.

**Result.** Section chunking is better at recall@10 (0.813 vs 0.692) and
nDCG@10 (0.684 vs 0.598), and worse at recall@5 (0.491 vs 0.569) and MRR
(0.734 vs 0.773).

That split is the whole finding. One vector per posting averages the four
things a posting is about, which makes the strongest match slightly blunter
but keeps the document coherent, so the very top of the ranking is good.
Section vectors match sharply on one paragraph, which surfaces postings the
whole-document vector missed entirely, but a posting can win on a paragraph
about the company's funding round rather than the job.

**Decision.** `whole` stays the default because the top three results are
what a user looks at. `section` is one config flag away and is the better
choice for /chat, where the retriever feeds eight sources to a model and
breadth matters more than the exact order. Both stay indexed; the storage
cost is 1,091 rows.

---

## 2026-09-17 - Dedup: embeddings and the hash disagree in both directions

**Hypothesis.** Embedding similarity is a superset of the Phase 1 content
hash. Identical text has cosine similarity 1.0, so anything the hash finds
the embeddings must also find.

**Result.** Wrong, and instructively so.

| threshold | hash pairs | embedding pairs | both | embedding only | hash only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.90 | 26 | 31 | 19 | 12 | 7 |
| 0.93 | 26 | 26 | 19 | 7 | 7 |
| 0.95 | 26 | 26 | 19 | 7 | 7 |
| 0.97 | 26 | 19 | 16 | 3 | 10 |

Seven pairs the hash finds survive even at a 0.90 threshold. The two methods
read different fields: the hash is title, company and location with seniority
words stripped, so it pairs "Senior ML Engineer, London" with "ML Engineer,
London" at the same company even when the two descriptions share nothing. The
embedding reads the description, so it pairs postings describing the same work
under different titles.

**Decision.** Threshold 0.93, and keep both. Verdicts go to
`posting_duplicates` with the method recorded, so a results page can suppress
either or both and the counts stay comparable. The Phase 1 hash was never a
baseline to be replaced; it is a second signal that happens to be free.

---

## 2026-09-17 - Local embeddings, and the comparison that did not run

**Result.** all-MiniLM-L6-v2 on CPU: 384 dimensions, 1.67 ms per text at
batch size 64, $0.00. Indexing the whole corpus is 25 seconds for the whole
strategy and 55 seconds for sections.

**Not done.** The plan calls for comparing this against an API embedding
model on cost, latency and quality. `ApiEmbedder` is implemented behind the
same interface and the eval harness can score either, but no
`EMBEDDING_API_KEY` is configured, so there is nothing to report. Quoting a
comparison from the provider's marketing page would be worse than leaving
this blank.

What can be said without the key: a 384-dimension local model retrieves
well enough that MRR is 0.77 before reranking, on hardware that costs nothing
per query, which is the number an API model would have to beat rather than
merely match.
