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

---

## 2026-09-17 - The refusal metric was wrong three times before the system was

**Hypothesis.** Half the chat golden set is questions the corpus cannot
answer. Detecting whether the system declined is a cheap string check, so the
metric can be deterministic and gate CI without involving the judge.

**What happened.** The string check was wrong three times running, and every
time it was wrong about an answer that was correct.

| run | answer | phrase check | truth |
| --- | --- | --- | --- |
| v1, first | "There is no information about the capital of Peru..." | refusal | refusal |
| v1, first | "None of the postings mention Elon Musk as a founder." | not a refusal | refusal |
| v2 | "I'm not able to answer that. The job postings don't mention it." | not a refusal | refusal |

The first run scored refusal accuracy at 0.33 and the gate failed the build.
One of those three was a real defect; two were the metric.

Each miss was fixable by adding a phrase, and that is exactly the problem.
Adding "none of the postings" after seeing the model say it is tuning the
measurement until it agrees with the output in front of you. Do that twice
and the metric reports whatever it was most recently taught to report.

**The real defect it did find,** and the reason the gate stays: asked "which
posting pays the highest salary across the whole database?", v1 answered
"According to posting [3] and [5]... This is the highest salary mentioned in
the database." It had seen six postings. That is the overreach the question
was written to catch, and no amount of phrase-list tuning would have
invented it.

**Fix, structural rather than lexical.** A refusal has no citations, because
there is nothing to cite. A real answer has at least one, because the prompt
requires one per claim. `is_refusal(cited, sources)` is that one line. It
agrees with a human reading on all 8 questions across both prompt versions,
where the phrase list got 3 of 16 wrong.

The failure it cannot see is a real answer that forgot to cite, which would
read as a refusal. That is what `citation_rate` measures, and the two are
reported side by side so a swap between them shows up.

**Second correction: the test case was wrong.** `c08` was marked
`must_refuse: true` while its own `expected_facts` said the answer should
"say it can only speak to the postings retrieved". Those contradict: the
right behaviour is to answer and scope the claim, not to decline. v2 does
exactly that ("I can only see the salaries in [1], [3] and [5]") and the
binary check called it a failure. The case was mis-specified and is now
`must_refuse: false` with the reason written into the file.

Changing a test because it fails is usually the worst thing you can do to an
eval. The distinction here is that the fields inside the case disagreed with
each other before any answer was produced, and the expected_facts were the
correct description.

---

## 2026-09-17 - Prompt v1 vs v2, and a judge that cannot be trusted yet

**Setup.** Same 8 questions, same judge, two prompt versions.
`joblens ab v1 v2`. v2 adds one rule: never answer as though you had seen
the whole database, say what is true of the postings in front of you.

**Result.**

| metric | v1 | v2 |
| --- | ---: | ---: |
| faithfulness (judge) | 0.69 | 0.75 |
| completeness (judge) | 0.94 | 0.94 |
| citation rate | 0.75 | 0.75 |

v2 is better on the one dimension the rule targets and unchanged elsewhere,
which is what a well-scoped prompt change should look like. It is 8
questions, so this is a direction rather than a result.

**The judge is not calibrated and its numbers are not gated.** Calibration
needs roughly 20 answers scored by hand, compared against the judge, and
reported as agreement plus Cohen's kappa. Those hand scores do not exist, so
`Calibration.trustworthy` returns False and faithfulness was removed from the
CI floors. Gating merges on a number this repo describes as untrustworthy
would be worse than having no gate.

Kappa rather than raw agreement, because the grades skew hard towards 2: a
judge that answers "2" to everything scores about 70% agreement and a kappa
of zero. `test_kappa_is_zero_for_a_judge_that_always_says_two` is that case
written down.

The worst bias here is self-preference. The judge is the same llama3.1 that
wrote the answers it is grading. The code can report that; only human labels
can correct it.

**What is gated:** refusal accuracy, because it is now a structural check
rather than a model's opinion, and retrieval nDCG and recall, because they
are scored against human judgements. Everything else is recorded and
charted, not enforced.

---

## 2026-09-18 - Distillation: half the teacher's labels did not survive review

**Hypothesis.** Skill extraction is the right task to distil. The reasoning is
shallow, the output format is rigid and the vocabulary is nearly closed, so a
0.5B student should be able to match an 8B teacher after a few hundred
examples.

**Setup.** 242 postings labelled by llama3.1 8B over Ollama. The plan calls
for a frontier API model as the teacher; no API key is configured, so the
teacher is the same local model that serves `/chat`. That is a materially
weaker teacher and every number below inherits it. 182 train, 60 test, split
before any label correction so nothing could leak.

**The teacher is loose.** 242 postings produced 251 distinct skills the
Phase 2 taxonomy has never heard of, and the mean label count was 2.99 with
140 of 242 postings getting an empty list.

**How the test labels were corrected.** Scoring a distilled student against
its own teacher's output measures imitation, and the teacher wins 1.0 by
construction, so the test split had to be corrected first. Two rules did the
work.

The first is the useful one. **The Phase 2 dictionary is exhaustive over its
own vocabulary**: it is a regex across 80 canonical skills and their aliases,
so if a posting names PyTorch anywhere in its title or description, the
dictionary found it. Therefore any taxonomy skill returned by the teacher
alone is a skill the posting does not contain. That is a proof, not a
heuristic, and it needs no human.

It caught a lot. Teacher-only mentions that appear nowhere in the posting
text were led by:

| hallucinated skill | postings |
| --- | ---: |
| pytorch | 12 |
| python | 10 |
| aws | 5 |
| javascript | 4 |
| postgresql | 3 |

The model is pattern-matching "this is an engineering job" onto the skills
such jobs usually want. It is exactly the failure mode a dictionary cannot
have, and it is invisible unless something independent checks the text.

The second rule needed reading. For the 80 distinct terms outside the
taxonomy, a substring check cannot separate a technology the dictionary is
missing from prose the teacher lifted out of the posting, because both are in
the text. So they were read and split by hand into
`data/finetune/vocabulary.yaml`:

- **41 accepted**, and these are real gaps in the Phase 2 taxonomy: ansible,
  react, vuejs, sveltekit, prometheus, grafana, playwright, pulumi,
  cloudformation, duckdb, opensearch, ebpf, webassembly, quic, tls.
- **39 rejected**, almost all of it the teacher answering a different
  question than it was asked: `production ai infrastructure`,
  `energy-aware scheduling and control`, `homeowner tradeoffs`,
  `pc hardware configuration`, `neuralwatt cloud`. These describe what the
  job does. They are not tools anyone puts on a CV.

**Result of the correction.** On the 60-posting test split, 47 teacher labels
were rejected as hallucinations, 43 more as unreviewed or non-skills, and 46
off-taxonomy mentions were accepted. Mean labels per posting fell from 2.99
to 1.93.

Applying the same rules to the training split dropped **300 of 610 teacher
labels, 49%**.

**Decision.** Train on the corrected labels rather than the raw teacher
output. Distilling a teacher that invents PyTorch on one posting in five
would teach the student to do the same, and the corrected set is free: the
correction is two rules and a vocabulary file, not an afternoon of reading.

**What this does to the headline.** The plan's punchline is "the fine-tuned
3B matched the API model at 10x lower cost". There is no API model here, and
the local teacher has a measured hallucination rate on the most common skills
in the corpus. So the interesting comparison is no longer student against
teacher. It is student against the 80-line regex, which costs nothing, runs
in microseconds and cannot hallucinate.

**Blind spot, stated because it is real.** A skill that both the dictionary
and the teacher missed never entered the labels. Recall on this test set is
recall against the union of what those two saw, not against the posting.

---

## 2026-09-19 - The fine-tune collapsed, and the regex won

**Setup.** Qwen2.5-0.5B-Instruct, LoRA rank 16, alpha 32, attention
projections only, 2,162,688 trainable parameters out of 496,195,456 (0.44%).
163 training examples after the internal eval split, 3 epochs, effective
batch 8, cosine schedule, lr 2e-4. CPU only: this machine has a 4GB GTX 1650
but torch 2.14 publishes no matching CUDA wheel, and running the student on
the same CPU as the teacher and the embedder keeps the latency column
comparable. **115.6 minutes.** Final train loss 1.615, eval loss 1.565.

The loss curve looked fine. Eval loss fell every epoch, 1.60 to 1.573 to
1.565, and token accuracy sat near 0.70. Nothing in the training output
suggested a problem.

**Result on 60 hand-corrected postings.**

| extractor | micro F1 | macro F1 | precision | recall | exact | ms/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher (llama3.1 8B) | 0.675 | 0.756 | 0.548 | 0.879 | 0.567 | 10,897 |
| rules (Phase 2 regex) | 0.654 | 0.742 | 0.714 | 0.603 | 0.567 | 14 |
| base (Qwen 0.5B) | 0.112 | 0.062 | 0.094 | 0.138 | 0.017 | 7,024 |
| tuned (Qwen 0.5B + LoRA) | **0.000** | 0.617 | 0.000 | 0.000 | 0.617 | 3,471 |

**The student learned to say nothing.** 60 empty predictions out of 60. Micro
F1 is zero because it never returned a single correct skill. Its macro F1 of
0.617 and its exact-match of 0.617 come entirely from the 37 postings whose
correct answer is an empty list: it scores full marks on those by refusing to
answer anything, ever.

This is the exact case the two averages were reported for. A single headline
number would have read 0.617 exact-match, close to the teacher's 0.567, and
the model is worthless.

**Why.** 56% of the training labels were empty, because a third of this
corpus is non-technical and the label correction stripped the teacher's
inventions from many of the rest. For a 0.5B model with 163 examples,
"always answer `{"skills": []}`" is a strong local optimum: it is right 56%
of the time on the training distribution and costs no capacity.

**The base model is worse and fails differently.** It returned
`["python", "pytorch", "aws"]` for a Marketing Student Assistant, for a BMS
Service Technician, and for everything else. That triple is the example in
the prompt's schema block. It is not extracting, it is copying the
instructions back.

**The comparison the plan wanted does not exist.** There is no API model
here, so the intended punchline of "matched the API model at 10x lower cost"
has nothing to attach to. What the table shows instead is more useful:

**An 80-line regex scores within 0.021 micro F1 of an 8B language model,
at 780x the speed and no failure modes.** The dictionary wins precision
0.714 to 0.548, because it cannot hallucinate. The model wins recall 0.879
to 0.603, because it is not limited to 80 skills. Both land at 0.567
exact-match.

That is the honest cost/quality story for this task, and the fine-tuned
model is not part of it.

**Decision.** `SKILL_EXTRACTOR` stays `rules`. The adapter ships in the repo
with its training log because the phase is the experiment, not the artifact,
and a negative result with a diagnosis is worth more than a missing one.

---

## 2026-09-19 - Why the student collapsed: six checks, one cause

The rebalanced retrain was nearly launched straight away. It would have
addressed the wrong thing. Six checks first, cheapest to most expensive.

**2c. Loss was computed on the prompt. This is the cause.**

`SFTConfig(completion_only_loss=True)` was set and had no effect whatsoever.
Printing the labels tensor for one training example:

```
sequence length      : 316
positions masked -100: 0
positions with loss  : 316
```

Zero masked. The trl docstring says why: completion_only_loss is *"supported
only for prompt-completion datasets"*. The dataset here is conversational,
a list of `messages`, so the flag was ignored. Silently. No warning, no
error, and `training.json` recorded `completion_only_loss: true` the whole
time.

The completion is `{"skills": []}`, about 8 tokens of 316. So roughly 97% of
every gradient step was teaching the model to recite the system prompt, the
five rules, the schema block and the job posting back. Only 2.5% was about
extracting skills.

It gets worse. The masked-in text includes the prompt's own schema line:

```
SCHEMA

{"skills": ["python", "pytorch", "aws"]}
```

The model was being explicitly trained to reproduce that literal. It also
explains the *base* model's behaviour, which returned exactly
`["python", "pytorch", "aws"]` for a Marketing Student Assistant and a BMS
Service Technician: it was never extracting, it was completing the example.

The correct flag for a messages dataset is `assistant_only_loss`, which
exists in trl 1.13, defaults to False, and needs `{% generation %}` markers
in the chat template. Qwen2.5's template has them.

**2a. The 56% empty rate is real, not a labelling failure.** Of 102
empty-labelled training postings, zero have skills the regex found and the
label dropped, which the adjudication rule guarantees by construction, and
only 4 had a teacher proposal rejected. Scanning the empty postings for
unknown capitalised tokens returns `strong`, `support`, `lead`, `ability`,
`responsibilities`: prose, not technology. Those postings are genuinely
non-technical, which is what a Hacker News hiring thread contains. Class
imbalance is a real property of the corpus and a contributing cause, but it
is not a bug and rebalancing alone would not have fixed anything.

**2b. Prompt parity is exact.** The training text starts with the inference
prompt character for character, then continues with the assistant turn.
Ruled out.

**2d. Label vocabulary parity is exact.** All 310 training label mentions
pass the same vocabulary filter the scorer applies. Nothing is being trained
in that the scorer then rejects. Ruled out.

**2e. Generation config is fine.** 160 max new tokens for an answer that
needs about 20, greedy decoding, and the parser counts unparseable responses
separately rather than coercing them to an empty list. Ruled out.

**2f. The collapse is genuine, not a harness artifact.** Raw untruncated
generations from the shipped adapter, on the eight test postings with the
richest gold labels:

```
gold=['ansible','aws','cloudformation','kubernetes','pulumi','saltstack','terraform']
  RAW: '{"skills": []}'
gold=['docker','elasticsearch','fastapi','flask','lmdb','mongodb','opensearch',...]
  RAW: '{"skills": []}'
```

Valid JSON, correct schema, nothing inside, every time.

**Conclusion.** One primary mechanical cause (2c), one real but secondary
data property (2a), four ruled out. The fix order follows: mask the prompt
first, and only then decide whether the class balance still needs touching.

**Noted in passing, not fixed here:** `msonormal` appears 36 times across
the empty-labelled postings. That is a Microsoft Word CSS class surviving
Phase 1's HTML stripping. It is a cleaning gap, it is not what broke this
phase, and it belongs in a Phase 1 fix rather than here.

---

## 2026-09-19 - Rung 2: masking the loss, 0.000 to 0.383

**Hypothesis.** The collapse is caused by computing loss on the prompt. Mask
it and the model starts extracting.

**Setup.** Exactly one variable changed from the collapsed run:
`assistant_only_loss=True` in place of the inert `completion_only_loss=True`.
Same 182 examples, split hash `4d3e33f8f7dd`, rank 16, alpha 32, 3 epochs,
lr 2e-4, CPU. `verify_masking()` confirmed the mask at the start of the run
rather than after it: 318 of 326 positions masked, 98%, and the 8 that carry
loss decode to the assistant answer.

Command: `python -m joblens distil-train`, log in `artifacts/rung2.log`.

**Validation F1 by epoch,** the curve the first run had no way to produce:

| epoch | micro F1 | empty | eval loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.000 | 100% | 0.1828 |
| 2 | 0.357 | 94% | 0.1351 |
| 3 | 0.588 | 88% | 0.1275 |

Note epoch 1. With the loss correctly masked the model still answered empty
on everything after a full epoch, and only started extracting in epoch 2.
Had this run been stopped at one epoch it would have looked identical to the
failure it was fixing.

**Result on the frozen 60-posting test set:**

| extractor | micro F1 | macro F1 | precision | recall | empty | ms/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher (llama3.1 8B) | 0.671 | 0.754 | 0.543 | 0.879 | 58% | 10,249 |
| rules (Phase 2 regex) | 0.654 | 0.742 | 0.714 | 0.603 | 52% | 11 |
| tuned-v2 (masked loss) | 0.383 | 0.692 | 0.674 | 0.267 | 85% | 3,319 |
| base (Qwen 0.5B) | 0.112 | 0.062 | 0.094 | 0.138 | 2% | 5,881 |
| tuned-v1 (unmasked loss) | 0.000 | 0.617 | 0.000 | 0.000 | 100% | 3,359 |

**Decision: confirmed, and not sufficient.** One flag moved micro F1 from
0.000 to 0.383 and cleared the collapse. The shape of what is left is
specific rather than mysterious. Precision is 0.674 against the regex
baseline's 0.714, so what the model says is mostly right. Recall is 0.267
because it answers empty on 85% of postings when the true rate is about 55%.
It under-answers; it does not hallucinate.

Two things say what to try next, and neither is a guess. The 85% empty rate
sits against a training set that is 56% empty. And the validation curve was
still climbing steeply at the last epoch, 0.000 to 0.357 to 0.588, so three
epochs under-trains.

**A note on eval loss.** It fell from 0.1828 to 0.1275 across a run where F1
went from 0.000 to 0.588, so here loss and quality happened to agree. In the
collapsed run loss fell just as neatly, 1.600 to 1.565, while F1 stayed at
zero. The lesson is not that loss is useless, it is that loss cannot
distinguish those two runs and F1 can. That is why the checkpoint is
selected on F1.

---

## 2026-09-19 - Gold v1 was circular, and the regex was the beneficiary

**How it surfaced.** Someone asked why the empty-prediction rate was so
high. Answering it meant looking at how gold was built, and gold was built
wrong.

**The defect.** Test gold was `rules(posting) | adjudicated(teacher(posting))`,
where adjudication filtered teacher terms through a hand-reviewed vocabulary
that is, in substance, the regex's own capability spec. Truth was therefore
defined as things the regex could have found. Gold was a superset of the
regex output on all 60 of 60 postings, so the regex could not emit a false
positive, and its precision came back as exactly 1.000.

That number was published as a finding. It was an identity.

Gold was also a subset of what the two labellers found between them, so both
recall figures were really recall-at-union and flattered both systems.

**The rebuild.** Three changes.

*Blind.* Candidates from both labellers pooled and shuffled, with provenance
written to a separate file and joined back only after every decision was
recorded. 206 candidate occurrences across the 60 postings: 136 teacher-only,
56 proposed by both, 14 rules-only.

*Per occurrence, not per term.* The unit is (posting, term). A global accept
list is wrong in both directions: "go" is a language in one posting and a
verb in the next.

*Additive.* A read of all 60 postings added 22 skills neither labeller
proposed, including figma, canva, opentofu, terragrunt, aeron, artio,
agrona, railway, vcl, xdp, ruby, hpc, .net, photoshop, intercom and discord.

**Counts.** 116 kept, 90 dropped, 22 added. Gold went from 116 mentions to
138, and from 62% empty to 57%.

**One rules-origin drop, and it is the whole point.** On a Portuguese
posting, the regex matched `excel` inside `Excelência`. The word boundary
characters in `skills.py` are ASCII, so an accented letter reads as a
boundary. Under gold v1 that error was structurally unobservable.

**Result, same 60 postings, gold hash 256c78609120d71f.**

| extractor | micro F1 | macro F1 | precision | recall | empty | ms/call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rules | 0.663 | 0.792 | 0.986 (69/70) | 0.500 (69/138) | 65% | 5 |
| teacher (llama3.1 8B) | 0.634 | 0.682 | 0.551 (103/187) | 0.746 (103/138) | 55% | 9,512 |
| tuned-balanced | 0.369 | 0.653 | 0.559 (38/68) | 0.275 (38/138) | 82% | 3,612 |
| tuned-masked | 0.337 | 0.639 | 0.674 (31/46) | 0.225 (31/138) | 85% | 3,348 |
| base (Qwen 0.5B) | 0.104 | 0.075 | 0.094 (16/170) | 0.116 (16/138) | 2% | 6,046 |
| tuned-v1 (collapsed) | 0.000 | 0.567 | 0.000 (0/0) | 0.000 (0/138) | 100% | 3,105 |

**What the circularity was worth.** Against gold v1 the regex led the teacher
0.753 to 0.671. Against v2 the gap narrows to 0.663 against 0.634, because
the additive pass put 22 terms into gold that an 80 word vocabulary cannot
reach. The regex did not get worse; the measurement got honest.

Recall is the cleanest line in the table. The teacher finds 103 of 138 gold
mentions, the regex 69. The trade is precision against coverage, at 5ms
against 9.5 seconds.

**Superseded.** The v1 table is kept above for history. Every v1 number was
computed against a gold set that could not falsify the baseline.

**No agreement figure.** Gold v2 has one adjudicator, who judged all 60
postings in one sitting. A second pass by the same adjudicator would agree
with itself, and reporting that kappa as inter-annotator reliability would be
a fabricated quality number. `adjudicate_cli --pass two --sample 20` reads
only the candidate pool and never the provenance or the first-pass file, so
an independent pass can produce a real figure later. Until then gold v2 is
better built than v1 and has no measured reliability, and both halves of that
sentence matter.

**Two guards, both confirmed failing first.** The superset guard asserts gold
is not a superset of the regex output. The parity guard asserts the Phase 6
baseline reads byte-identical text to what gold was built from; checked by
pointing `RuleExtractor` back at `posting_text` and watching it fail on a
posting where `rust` appears past the 1800 character cutoff.

The parity guard is deliberately narrower than "make the two text builders
identical". `posting_text` repeats the title as Phase 2 TF-IDF weighting and
feeds clustering, the salary model and skill_matrix; forcing byte-identity
would move Phase 2's published coverage figure to fix a Phase 6 bug. A third
test pins the divergence as intentional.

**Operational note.** The first rescore returned the teacher at 0.000 with 60
of 60 unparseable. Not a result: four Qwen models were resident in one
process, free memory reached 0.2GB, and Ollama answered HTTP 500 on every
call. Re-run in a clean process it scores 0.634 with zero parse failures. The
collapse guard caught it, the second time that guard has paid for itself.

---

## 2026-09-19 - Training data for the gold v2 retrain

**Counts before anything was applied.** 404 training examples, zero overlap
with the frozen test set checked by (source, source_id). 164 empty, 41%.

**Empty cap.** 30%, giving 342 examples with 102 empty. Not zero: 57% of the
test set is genuinely empty, and a model that never answers empty invents
skills for the non-technical third of the corpus.

**Agreement on the 240 non-empty examples.**

| | count | share |
| --- | ---: | ---: |
| exact rules/teacher agreement | 9 | 4% |
| partial overlap | 177 | 74% |
| no overlap at all | 54 | 22% |

**Decision: no disagreement filter.** Dropping the 54 no-overlap examples
would remove 13% of the set, within the permitted range, and it was still
the wrong call. Those are largely postings where the regex is silent because
the technology sits outside its 80 words and the teacher found it correctly.
They are the only examples that could teach the student to beat the regex on
recall, which is the one dimension where it loses badly, 0.275 against 0.500.
The filter would have removed signal and called it noise.

Filtering to exact agreement was never an option: it leaves 9 examples, and
it would teach the student the regex's own coverage, which is the gold
superset problem one layer down.

---

## 2026-09-19 - The retrain on gold v2 data: 0.000 to 0.485

**Hypothesis.** The remaining gap is training data, not adapter capacity.
Validation F1 had plateaued flat at 0.364 across epochs 4 and 5 on 103
examples, which points at data rather than under-training.

**Setup.** 342 examples at 30% empty, split hash `6eaa8009eb61`, no
disagreement filter. Rank 16, alpha 32, 2 epochs, lr 2e-4, 1800 character
input, CPU. `verify_masking()` confirmed 577 of 607 positions masked before
the run started. 104.2 minutes.

Command: the balanced split through `training.train`, log in
`artifacts/rung_gold_v2.log`.

**Validation F1 by epoch.**

| epoch | micro F1 | empty | unparseable | eval loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.573 | 38% | 1 | 0.3919 |
| 2 | 0.602 | 31% | 0 | 0.3602 |

Compare epoch 1 across the three runs. Unmasked loss: 0.000 at 100% empty.
Masked loss on 182 examples: 0.000 at 100% empty. Masked loss on 342 better
labelled examples: **0.573 at 38% empty after a single epoch**. The model no
longer has to escape the always-empty attractor, because the attractor is
much weaker.

**Result on the frozen 60 postings, gold v2.**

| | tuned-v1 | tuned-masked | tuned-balanced | tuned-v2 |
| --- | ---: | ---: | ---: | ---: |
| micro F1 | 0.000 | 0.337 | 0.369 | **0.485** |
| macro F1 | 0.567 | 0.639 | 0.653 | **0.710** |
| precision | 0.000 | 0.674 | 0.559 | 0.516 |
| recall | 0.000 | 0.225 | 0.275 | **0.457** |
| empty | 100% | 85% | 82% | 77% |

**Decision: the regex still ships.** `SKILL_EXTRACTOR` stays `rules`. At
0.485 against 0.663 the student is not close enough to justify 4.2 seconds a
posting against 5 milliseconds.

**What is interesting anyway.** Its macro F1 of 0.710 beats the teacher's
0.682, so on a per-posting basis it handles the sparse majority better than
the model that taught it. Its recall of 0.457 is within striking distance of
the regex's 0.500, which is the dimension a vocabulary-bound regex can never
improve on without someone hand-writing more aliases. What it lacks is
precision, 0.516 against 0.986.

That shape is the opposite of the teacher's, which is a distillation working
as intended in one direction: the student learned to be quieter than its
teacher (77% empty against 55%) without learning to be as accurate.

**Still climbing.** Validation F1 rose 0.573 to 0.602 between the two epochs
and the run was stopped at two by the training budget. A third epoch was not
run. Whether it would have closed any of the remaining 0.18 gap is unknown,
and saying so is cheaper than guessing.

**What did not get tried.** Rung 5 of the earlier plan, escalating to
Qwen2.5-1.5B or the 3B the project plan specifies. This machine has 7.3GB of
RAM; a 1.5B model in fp32 is about 6GB of weights before optimiser state and
activations. Not reachable here, and recorded rather than quietly skipped.

---

## 2026-09-26 - Clustering on embeddings vs TF-IDF: they disagree, and both are right

**Hypothesis.** k-means on the Phase 3 embeddings groups postings by what
the job is rather than by which words the recruiter used, so it should
find the same families as TF-IDF with less noise and higher agreement
between the two.

**Setup.** 468 postings, every one with a whole-posting all-MiniLM-L6-v2
vector from the Phase 3 index. Same k-means, same seed, k chosen by
silhouette over 3..10 in each space. Clusters found in embedding space are
still labelled from TF-IDF, by averaging the TF-IDF rows of their members,
so both runs are named by the same vocabulary. Agreement is the adjusted
Rand index between the two partitions: 1.0 is identical, 0.0 is chance.
`python -m joblens cluster --compare`.

**Result.**

| representation | k | silhouette | clusters (postings) |
| --- | ---: | ---: | --- |
| tfidf | 8 | 0.008 | ai / infrastructure (95), role / experience (92), software engineer / senior (86), product / remote (80), ai / founding (65), francisco / san (19), https / www (18), healthcare / data platform (13) |
| embedding | 4 | 0.024 | software / senior (164), ai / software (146), role / team (101), health / clinical (57) |

Adjusted Rand index **0.104**. The two partitions barely agree, and reading
them says why. TF-IDF splits on surface tokens: a `francisco / san` cluster
that is a location, not a job, and an `https / www` cluster of 18 postings
whose title or company field is a bare URL, which `strip_noise` never sees
because it only runs on descriptions. Embeddings ignore both and settle on
four broad families, one of which, `health / clinical` with 57 postings, is
the same healthcare group TF-IDF found with 13. The embedding version is the
one a person would draw; the TF-IDF version is the one with more names.

The silhouettes are not comparable. Each is measured in its own geometry,
and 0.024 in 384 dense dimensions says nothing about 0.008 in 13,000 sparse
ones. The Rand index is the only number in the table that compares them.

**Decision.** Keep TF-IDF for the dashboard: eight nameable groups beat four
vague ones when the label is the product. Use the embedding partition as
the sanity check it turned out to be, and fix the thing it exposed: run
`strip_noise` over the title as well as the description in `posting_text`,
so URL fragments stop qualifying as vocabulary. Applied at feature time,
like the description, so the stored rows stay as the boards wrote them.

**After the fix.** Same run with titles stripped:

| representation | k | silhouette | clusters (postings) |
| --- | ---: | ---: | --- |
| tfidf | 9 | 0.007 | ai / software (117), team / product (99), experience / role (76), software engineer (55), ai / agents (51), ml / models (29), uk / latam (18), remote europe / senior (13), net / healthcare (10) |
| embedding | 4 | 0.024 | unchanged |

The `https / www` cluster is gone and `ai / agents` and `ml / models`
appeared in its place, which is the first time k-means has separated the
agent-building roles from the rest. The Rand index fell to 0.064: the
embedding partition did not move, and the TF-IDF one moved further from
it, which is what happens when the surface-token clusters get better
rather than fewer. The silhouette did not notice, as before.

---

## 2026-09-26 - Permutation importance: the text is the only column that matters

**Hypothesis.** The earlier salary write-up read the ridge's coefficients
and found recruiter prose. Coefficients say which term the model leaned on,
not whether the text mattered at all. Shuffling one input column at a time
on a held-out split answers the second question for every column, and does
it for the trees too.

**Setup.** 110 postings with a usable salary (the corpus grew by four since
the first run). 75/25 split, seed 42, the ridge that won 5-fold CV, ten
shuffles per column. The number is how much MAE rises when that column is
scrambled. `python -m joblens train-salary --importance`.

**Result.**

| column | MAE increase (USD) | sd |
| --- | ---: | ---: |
| text | +11,762 | 2,101 |
| region | +2,699 | 2,024 |
| seniority | +1,695 | 2,063 |
| source | +244 | 515 |
| is_remote | -287 | 434 |

Only the text moves the needle, and it moves it by 11.8k on a model whose
MAE is 59k. Region and seniority are inside one standard deviation of zero.
`is_remote` going slightly negative means shuffling it made the held-out
score fractionally better, which is what a column the model should not
have been given looks like.

The CV table itself shifted with the four new rows: ridge now beats the
median by 8.2% (59,148 vs 64,426), up from 1.8%. That is the same lesson as
the first run from the other side: with this few rows, four postings move
the headline by six points.

**Decision.** Unchanged. The text carries what signal there is, the signal is
not enough to serve, and the categorical features can be dropped from the
ridge without loss. No salary endpoint until roughly 500 disclosing
postings.

**Correction, same day.** The dev database was emptied by a test run and
rebuilt from the boards (see the rebuild entry below): 109 priced postings,
one fewer, mostly the same rows. The same command then ranked `source` first
at +8,552 and text second at +2,767. One ingest reversed the conclusion,
which means the conclusion was never measured: a 25% split of 110 rows is 28
postings, and the "sd" column above was the spread across shuffles of those
28, not across samples of the data. It could not have shown this.

`column_importance` now scores every fold of the same 5-fold split the model
comparison uses, and the sd is taken over folds and shuffles together. On
the rebuilt corpus:

| column | MAE increase (USD) | sd |
| --- | ---: | ---: |
| text | +14,350 | 10,659 |
| source | +3,834 | 4,774 |
| region | +1,425 | 5,467 |
| seniority | +974 | 2,142 |
| is_remote | -1,303 | 2,822 |

Text is still first and is the only column more than one sd above zero; the
headline survives. What changed is the honest size of the error bar: the
text effect is somewhere between about 4k and 25k, and nothing else is
distinguishable from zero. The decision stands, now for a reason that holds
up under a re-ingest.

---

## 2026-09-26 - The golden set at 58 queries: the 15-query numbers were optimistic

**Hypothesis.** The retrieval leaderboard measured on 15 queries judged from
titles and snippets would hold on the full 60-query set judged from full
posting text.

**Setup.** Every candidate any retriever surfaced in its top 10 for each of
the 60 queries, 1,274 candidates in all, dumped with the full cleaned
description by `scripts/golden_pool.py` and graded 0/1/2 by Claude Fable 5.1
reading the posting, six batches of ten queries. Merged with
`scripts/golden_apply.py`, which records that provenance in every query's
note. Two queries ended with no relevant posting (`pgvector`, whose one
posting has expired, and `jobs paying over 200k`, which the note says
retrieval should fail honestly) and sit out of the average. Same corpus of
468 postings for every row.

**Result.**

| configuration | recall@5 | recall@10 | mrr | ndcg@10 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.209 | 0.372 | 0.471 | 0.353 | 12 |
| vector (whole) | 0.361 | 0.658 | 0.781 | 0.621 | 16 |
| vector (section) | 0.339 | 0.590 | 0.753 | 0.589 | 19 |
| hybrid (whole) | 0.361 | 0.541 | 0.770 | 0.547 | 51 |
| hybrid (section) | 0.313 | 0.521 | 0.719 | 0.525 | 53 |
| hybrid + rerank | 0.373 | 0.561 | 0.818 | 0.586 | 2030 |

Against the 15-query table every number fell and the top reordered. The
reranker had MRR 0.900 and nDCG 0.692; it now has 0.818 and 0.586, still the
best MRR but no longer the best nDCG, which goes to plain whole-posting
vector search. Section chunking lost the depth advantage it had shown.
Hybrid still trails vector on everything.

Two reasons, both about the old set rather than the retrievers. Fifteen
queries is a sample where one query is 7% of the score. And grading from a
snippet the retriever chose is grading the retriever's argument for itself:
a posting whose snippet mentions the query term looks relevant even when the
full text says it is a marketing role that happens to name the tool.

**Decision.** The 58-query numbers replace the 15-query ones everywhere they
were quoted. The floors in `eval/report.py` (nDCG 0.45, recall@10 0.55)
still hold with room, so the gate is unchanged. Whole-posting vector search
is confirmed as the default representation; hybrid stays the default mode
for the rare-token case, with the cost now measured on four times the data.
The gap that remains is that the grades are a model's reading, not a
person's: the file says so on every query, and a human pass over the 227
grade-2 judgements is the next step that would make these numbers quotable
without a caveat.

---

## 2026-09-26 - The corpus was wiped and rebuilt: which numbers survived

**What happened.** Fifteen minutes after the 58-query eval above was
recorded, a test run pointed at the dev database emptied it: postings,
chunks and the bronze history. The suite truncates whatever `DATABASE_URL`
names, and `.env` names the dev corpus. The rebuild (`migrate`, `ingest
--limit 400`, `embed`) came back with 466 postings instead of 468, and 88 of
the golden set's 1,274 judgements now point at postings the boards have
expired. Eleven queries lost a relevant posting.

That is an accidental robustness test, so it was run as one. Hypothesis:
every conclusion above survives a re-ingest. Anything that does not was
never measured.

**Retrieval, before the top-up.** Same 58 queries, same grades:

| configuration | recall@10 | mrr | ndcg@10 |
| --- | ---: | ---: | ---: |
| vector (whole) | 0.665 | 0.761 | 0.618 |
| hybrid (whole) | 0.544 | 0.781 | 0.548 |
| hybrid + rerank | 0.551 | 0.772 | 0.564 |

Vector search reproduced to within 0.01. The reranker's MRR fell from 0.818
to 0.772 and lost the column to plain hybrid, which is what an ungraded
posting does: it scores as irrelevant, and the reranker is the configuration
most likely to pull a new posting into the top ranks.

**Top-up.** `scripts/golden_pool.py --all --unjudged` (new flag) listed 105
candidates across 46 queries that no one had graded. They were graded 0/1/2
by Claude Opus 5.5 from the same pool dump the Fable 5.1 pass read: 88 zeros,
11 ones, 6 twos, mostly keyword hits on company boilerplate. Merged with
`golden_apply.py`, which now appends provenance on a merge instead of
replacing it, so every topped-up query names both graders.

**Retrieval, after the top-up.**

| configuration | recall@5 | recall@10 | mrr | ndcg@10 | ms/query |
| --- | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.224 | 0.402 | 0.537 | 0.390 | 10 |
| vector (whole) | 0.374 | 0.653 | 0.761 | 0.612 | 15 |
| vector (section) | 0.347 | 0.590 | 0.734 | 0.581 | 17 |
| hybrid (whole) | 0.366 | 0.548 | 0.781 | 0.555 | 46 |
| hybrid (section) | 0.319 | 0.527 | 0.729 | 0.533 | 47 |
| hybrid + rerank | 0.355 | 0.553 | 0.794 | 0.577 | 1956 |

Every column has the same winner as the 468-posting run except recall@5,
which moved from the reranker (0.373) to vector search (0.374), a tie in
anything but name. The retrieval conclusions survived.

**Clustering.** `cluster --compare` now picks k=10 for TF-IDF and k=6 for
embeddings, with an adjusted Rand index of 0.066 (0.064 before). The
embedding run again finds a health / clinical family (45 postings); TF-IDF
again finds the forward-deployed and founding-engineer groups. The
specific k moved, the finding that the two spaces barely agree did not.

**Salary importance.** Did not survive, and is corrected in its own entry
above: a single 28-row test split ranked a different column first after the
rebuild. It now scores every CV fold, and the corrected ranking is the one
that is quoted.

**Decision.** Three changes so this cannot recur or go unnoticed:

1. `tests/conftest.py` renames the database in `DATABASE_URL` to
   `<name>_test` before any test runs, and CI's Postgres now creates
   `joblens_test`. The suite can no longer reach the dev corpus at all.
2. After any rebuild, run `golden_pool.py --all --unjudged` and grade what
   it lists before quoting retrieval numbers. 105 candidates took one pass;
   re-grading all 1,379 would not have been needed.
3. Any importance or ranking claim on the salary data is scored across
   folds with the spread reported, because 110 rows cannot support a single
   split.

---

## 2026-09-26 - Judge calibration: 20 hand scores, and why they cannot calibrate it

**Hypothesis.** The LLM judge (llama3.1 8B, `prompts/judge_answer`) agrees
with a person often enough to trust its faithfulness and completeness
scores, measured as Cohen's kappa of at least 0.4 on 20 hand-scored answers.

**Setup.** 20 real `/chat` answers drafted by `scripts/calibration_draft.py`,
scored 0-2 on both scales by the project owner, then re-scored by the judge
with `python -m joblens calibrate`.

**Result.** The owner scored all 20 answers 2 for faithfulness and 2 for
completeness.

| dimension | agreement | kappa |
| --- | ---: | ---: |
| faithfulness | 65.0% | 0.00 |
| completeness | 95.0% | 0.00 |

The judge scored 7 answers 1 for faithfulness where the owner gave 2, with
reasons that are checkable against the sources: "[1] does not explicitly
mention Kubernetes" (remote Kubernetes jobs), "a claim that is not supported
by the sources" (jobs in Germany), and a partial answer to the Canadian visa
question from a support-role posting.

Kappa is 0.00 by construction, not by measurement. When every human score is
the same value, the agreement expected by chance equals the observed
agreement, whatever the judge does, so the statistic cannot tell a good judge
from a bad one. `Calibration.trustworthy` stays False and the judge's scores
remain labelled uncalibrated.

**Decision.** The judge is not calibrated. The seven disagreements are the
useful output: each is an answer to re-read against its sources. A second
scoring pass that marks unsupported claims as 0 or 1 would give the sample
the variation kappa needs; until then no judge score is quoted as more than
a smoke test.

The same session also merged the owner's review of the 232 grade-2
retrieval judgements into `data/golden/retrieval.yaml`: all 232 were kept,
no grade changed, and each affected query's note now records the review. The
retrieval numbers are unchanged because no grade moved.
