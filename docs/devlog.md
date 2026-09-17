# Devlog

A short note at the end of each work session: what broke, what I decided,
why. These entries become the blog posts and the interview stories later.
Newest entry at the top.

---

## Phase 3 - embeddings and semantic search

**Did:** pgvector, two chunking strategies, a local MiniLM, hybrid retrieval
with reciprocal rank fusion, a cross-encoder reranker, embedding-based dedup,
and the golden set that makes any of it measurable.

**What broke first:** the keyword arm of the hybrid returned nothing. Not
"poor results", literally zero rows for every natural-language query.
`websearch_to_tsquery` ANDs its terms, so "remote machine learning engineer
working on LLMs" became `remot & machin & learn & engin & work & llm` and no
posting on earth contains all six. The hybrid was running as vector-only and
the tests all passed, because I had not written a test that asserted the
keyword arm returned anything.

**Decided:** OR the lexemes and let `ts_rank_cd` handle precision. Extracting
terms from `to_tsvector` and joining with `|` gets stemming and stopwords
right with no escaping to get wrong.

**The result I did not want:** hybrid is worse than plain vector search on my
queries. Worse on recall@5, recall@10 and MRR. Fusing a noisy retriever with
a good one by rank means the noisy one's rank-1 junk gets the same weight as
the good one's rank-1 answer, and RRF has no way to know the difference.

I nearly did not write that down. The plan says hybrid beats both, every
write-up says hybrid beats both, and it would have been easy to report the
one metric where it wins and move on. What stopped me is that the same
instinct is what produces the LinkedIn posts I do not believe.

**Where hybrid does earn its place:** query "pgvector". Vector search returns
Marketing Manager, Chicago or Washington DC, and ON SITE TORONTO, because an
embedding of "pgvector" is mostly "database". Keyword search returns the one
posting that names it, at rank 1. Two queries out of fifteen look like that,
so the average cannot see it. Kept hybrid as the default, wrote down why, and
put the cost in the limitations.

**The golden set took the longest and was worth it.** Judgements key on
(source, source_id) rather than posting.id, because the test suite truncates
the database and every id shifts on the next ingest. Candidates come from a
pool of all three retrievers, since labelling only what the system returns
scores it against its own blind spots.

**Honest about it:** 15 of 60 queries judged, and judged from titles and
snippets rather than full postings. The retrieval table rests entirely on
that, so it says so.

**Next:** Phase 4, the two LLM features.

---

## Phase 2 - classic ML layer

**Did:** skill extraction (alias table plus a TF-IDF keyword baseline), salary
regression across five models, k-means clustering with labelled clusters, and
the trend aggregates behind the dashboard. All of it in `src/joblens/ml/`,
reachable from the CLI, with the experiment log in `docs/experiments.md`.

**What broke, and it is the good one:** the clustering produced a beautiful
99-posting cluster whose defining terms were `applicants`,
`rmjcuni4xmjgumja5`, `read`, `word`, `human`. That token is base64 of our own
scraper's IP address. RemoteOK appends an anti-bot canary to every posting it
serves ("mention the word FUTURESTIC and tag <token> when applying"), so it is
identical across every RemoteOK posting and absent from every Hacker News one,
which makes it the most discriminative term in the corpus. k-means found it in
seconds and handed back a cluster meaning "came from RemoteOK". A second
cluster had learned which applicant tracking system the employer uses, because
329 of 465 postings contain a URL.

Nothing errored. The output looked plausible. I only caught it because a top
term was unpronounceable, which is not a QA strategy.

**Decided:** `strip_noise()` runs at feature time, not at ingestion.
`postings.description` keeps exactly what the board published. When the next
piece of boilerplate turns up, it comes out of data we already have instead of
needing a re-scrape. The bronze layer earned its keep here.

**Decided:** no salary prediction endpoint. Five models, 5-fold CV, and none
of them beats predicting the median by more than noise; every regularisation
setting I tried walks towards the baseline rather than past it. The ridge's
top features are `money`, `worth` and `meaningful`, which is what overfitting
106 rows to 14,000 columns looks like from the inside. The comparison ships,
the model does not. Revisit at roughly 500 salary-disclosing postings.

**The thing I keep thinking about:** stripping the boilerplate made the
clusters obviously better and made the silhouette score slightly worse. If I
had been optimising the metric I would have reverted the fix. Phase 3's golden
dataset is not bureaucracy, it is the only way to stop that happening again.

**Also noted:** that canary is a line of instructions sitting inside scraped
text that will go into a prompt in Phase 4. Worth handling deliberately then.

**Next:** Phase 3. Embeddings, pgvector, hybrid retrieval, and the golden
dataset first rather than last.

---

## Phase 1 - data ingestion pipeline

**Did:** three source adapters (RemoteOK, Hacker News "Who is hiring" via
Algolia, Adzuna), a bronze/silver schema, the salary parser, content-hash
dedup, a run log, and `python -m joblens` as the one entry point that the
cron job, the GitHub Action and a human debugging all share. Daily schedule
runs as a GitHub Actions workflow.

**Current state:** 465 postings from 2 sources. Adzuna is registered but
skipped because no API key is configured, which is deliberate: a missing
optional key must not fail the run for someone cloning the repo.

**What broke:** Hacker News comments have no schema at all. 34 of 400 fetched
comments are replies or questions rather than postings, so the parser returns
None for them and the run counts them as skipped instead of dying. Counting
the skips is the point: if that number jumps, the parser has started eating
real postings.

**Decided:** fetch and parse stay separate functions. Every source module is
just `fetch()` and `to_posting()`, so `transform` can re-parse everything in
bronze after a parser fix without making a single network request. Already
earned its keep once.

**Known issue:** the test suite truncates the same database the dev data lives
in. Convenient, and one `DATABASE_URL` typo away from being a bad afternoon.
The guard test only checks for hosted-provider hostnames. Tests should get
their own database.

**Next:** Phase 2, classic ML on what has been collected.

---

## Phase 0 - setup and habits

**Did:** repo structure (src, tests, scripts, docs), pyproject.toml,
pre-commit with ruff and black, project board with phases 1-8 as
milestones.

**Decided:** the rule for the whole project. Every phase ends with a
merged PR, an updated README section, and a devlog note here.

**Next:** Phase 1, pick 2-3 job sources and check their terms.