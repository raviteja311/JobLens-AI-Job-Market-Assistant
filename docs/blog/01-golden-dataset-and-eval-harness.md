# Building a golden dataset and an eval harness for a RAG app

*Draft for publication. Every number is from `docs/experiments.md` and the
README of [JobLens](https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant),
measured on a 465-posting corpus on 2026-09-17. The corpus grows daily, so
re-running today gives slightly different figures and the same decisions.*

---

Most RAG demos end with "and it works". Mine ended with a table where the
configuration every blog post recommends came third, and a metric I had to
rewrite three times before it stopped agreeing with whatever the model said.
This is the story of how a small job-search project got a real evaluation
harness, what it caught, and what it still cannot see.

## Why a golden set before anything clever

JobLens collects AI and ML job postings from Hacker News "Who is hiring"
threads and RemoteOK, stores them in Postgres with pgvector, and serves
semantic search, a grounded chat endpoint and resume matching. By the time
search worked I had three retrievers (keyword, vector, hybrid), two chunking
strategies and a cross-encoder reranker, and no way to say which was better
except by typing queries and squinting.

So the first deliverable of the search phase was not a retriever. It was 60
real queries in a YAML file, of which I judged 15 by hand on a 0, 1, 2 scale:
no, acceptable, exactly what was asked. Fifteen is small and the README says
so. It was still enough to overturn my assumptions twice.

Two design choices mattered more than the size.

**Candidates come from a pool of all three retrievers**, not from the system
I was testing. If you only label what your retriever returns, you score it
against its own blind spots and it can never look bad.

**Judgements key on the source's own identifier**, never on the database
primary key. The test suite truncates the development database, and a
re-ingest into an empty table renumbers every posting. A golden set keyed on
`posting.id` would have silently pointed every judgement at a different job
the first time someone ran `pytest`. That bug would have produced numbers
that looked fine.

## The first thing the harness caught was that hybrid search did not exist

The hybrid retriever fused keyword and vector results with reciprocal rank
fusion. Tests passed. Results looked reasonable. Then the harness reported
that the keyword arm contributed zero candidates on every natural-language
query.

Postgres's `websearch_to_tsquery` ANDs its terms. "Remote machine learning
engineer working on LLMs" became six stemmed terms that all had to match,
and no posting on earth contains all six. The hybrid had been running as
vector-only the whole time, and I had never written a test asserting that
the keyword arm returned anything at all.

The fix was to OR the lexemes and let `ts_rank_cd` handle precision. That
is a one-line change. Finding it without a harness would have taken a user
noticing that searching for "pgvector" returned marketing jobs.

## The second thing it caught was that I was wrong about hybrid

| configuration | recall@5 | recall@10 | MRR | nDCG@10 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.299 | 0.358 | 0.406 | 0.328 | 25 |
| vector (whole posting) | 0.569 | 0.692 | 0.773 | 0.598 | 34 |
| vector (sections) | 0.491 | 0.813 | 0.734 | 0.684 | 35 |
| hybrid (whole) | 0.438 | 0.666 | 0.761 | 0.620 | 94 |
| hybrid (sections) | 0.430 | 0.646 | 0.736 | 0.600 | 93 |
| hybrid + rerank | 0.497 | 0.725 | **0.900** | **0.692** | 3145 |

Hybrid lost to plain vector search on recall@5, recall@10 and MRR. The plan
said hybrid would win. Every write-up I had read said hybrid would win. The
mechanism is not mysterious once you see it: fusing a noisy retriever with a
good one by rank means the noisy one's rank-1 junk gets the same weight as
the good one's rank-1 answer, and RRF has no way to tell them apart. For a
six-word query the keyword arm ranks a posting that matches only "engineer"
first.

I nearly reported the one metric where hybrid edged ahead and moved on. What
stopped me was noticing that the same instinct is what produces the LinkedIn
posts I do not believe.

Hybrid stayed the default anyway, and the reason is a single query. Search
for "pgvector" and vector retrieval returns Marketing Manager and ON SITE
TORONTO, because the embedding of "pgvector" is mostly "database". Keyword
retrieval returns the one posting that names it, at rank 1. Two of fifteen
queries look like that, so the average cannot see them. Hybrid is insurance
for the query type embeddings cannot handle, paid for with a measurable cost
on ordinary queries. That trade-off is written in the limitations section,
which is where it belongs.

The reranker was the real improvement: MRR from 0.773 to 0.900. It costs
3.1 seconds a query on CPU, so it is off by default and on behind a flag.

## Evaluating the chat endpoint: the metric was wrong before the system was

The chat endpoint retrieves, builds a prompt with numbered sources, and
answers with citations. When retrieval finds nothing above a score
threshold, the model is never called. A RAG system that always answers is
easy to build and useless, because its answer to a question the corpus
cannot address is indistinguishable from a real one.

So the chat golden set is 8 questions, half of which the corpus genuinely
cannot answer. "What is the capital of Peru?" is in there on purpose. The
headline metric is refusal accuracy: did the system decline exactly the
questions it should have declined.

My first refusal detector was a list of phrases. It scored three correct
refusals as failures, because the model worded them differently each time:
"None of the postings mention Elon Musk as a founder." "I'm not able to
answer that." Each miss was fixable by adding another phrase. I did that
twice before recognising what I was doing: tuning the measurement until it
agreed with the output in front of it.

The version that shipped is structural. A refusal has no citations, because
there is nothing to cite. A real answer has at least one, because the prompt
requires one per claim. That definition agrees with a human reading on all 8
questions across both prompt versions; the phrase list got 3 of 16 wrong.

The harness then caught a real defect. Asked "which posting pays the highest
salary across the whole database?", the first prompt answered "this is the
highest salary mentioned in the database" having seen six postings. Prompt
v2 adds a rule about scope. The A/B is the reason to believe the fix:

| metric | v1 | v2 |
| --- | ---: | ---: |
| faithfulness (judge) | 0.69 | 0.75 |
| completeness (judge) | 0.94 | 0.94 |
| citation rate | 0.75 | 0.75 |

## The judge, and why its numbers are labelled untrustworthy

Faithfulness and completeness above come from an LLM judge. The judge is
llama3.1 8B running locally, which is also the model that wrote the answers.
That is self-preference bias at its maximum, and the code cannot fix it.

What the code can do is refuse to pretend. The harness has a calibration
command that scores the judge against hand-graded answers and reports
Cohen's kappa, not raw agreement, because the grades skew high and a judge
that says "2" to everything would score around 70% agreement with a kappa
of zero. Until 20 hand-graded answers exist, `Calibration.trustworthy`
returns False and the README says the judge scores are a smoke test.

I have not graded those 20 answers. Writing the calibration code and then
quoting uncalibrated scores as if they meant something would have been the
most dishonest thing in the repository, and it would have been invisible.

## Making it a gate, not a report

An eval that prints numbers is a report. This one fails the build.

Retrieval runs in GitHub Actions on every pull request that touches
retrieval code, prompts or the golden set. Floors live next to the metric
code, not in the workflow file, so a threshold change shows up in a pull
request beside whatever needed it. Refusal accuracy has a hard floor of
0.75. Everything else is compared to the previous run with a tolerance of
0.05, because the corpus grows daily and a gate that fires on noise gets
commented out within a week, after which there is no gate.

Chat is not in the pull-request gate. A local Llama on a GitHub runner takes
about half an hour for eight questions. It runs on demand and before a
release. A green tick that means "we skipped it" is worse than no tick.

## What the harness cannot see

- Recall is recall over the candidate pool. A posting no retriever surfaces
  is never judged and never counted as missed. At 465 postings the gap is
  small; it would not be at 50,000.
- Fifteen judged queries, graded from titles and snippets rather than full
  postings. Differences of a few points in the table above are not
  meaningful.
- The judge is uncalibrated and shares a model with the system under test.

## The one paragraph I would keep

In the classic ML phase I stripped scraper boilerplate out of the clustering
input, because k-means had found a token that turned out to be the base64
of my own IP address and built a 99-posting cluster around it. The clusters
went from "came from RemoteOK" to "founding engineer". The silhouette score
got slightly worse. If I had been optimising the metric I would have
reverted a correct fix. That is the argument for building the golden set
before trusting any number, and it is the whole harness in one story.
