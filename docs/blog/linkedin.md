# LinkedIn drafts

One post per blog post, plus a headline. Drafts, not published: posting is a
human's decision. Each post ends with the repo link so the numbers can be
checked. No claims here that the README does not also make.

Repo: https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant

---

## Headline

> AI/ML engineer in training. Built JobLens: a job-market assistant with
> hybrid search over pgvector, grounded RAG chat, an eval harness that
> fails CI on regressions, and a fine-tuning experiment an 80-line regex won.

Shorter variant for the 220-character limit:

> Building JobLens, an AI job-market assistant: pgvector search, RAG with
> citations, eval gates in CI, LoRA fine-tuning. Numbers in the README.

---

## Post 1: the golden dataset and the eval harness

I built an evaluation harness for a RAG app and the first thing it told me
was that my hybrid search did not exist.

The keyword arm had returned zero candidates on every natural-language
query for weeks. Postgres ANDs search terms by default, and no job posting
contains all six words of "remote machine learning engineer working on
LLMs". Tests passed. Results looked fine. The harness said 0.

The second thing it told me was that hybrid search, once it worked, lost
to plain vector search: recall@10 0.55 against 0.65 across 58 judged
queries. Every write-up says hybrid wins. On mine it came third. Reranking
won exactly one column, MRR, at 130 times the latency. And my first 15-query
table, graded from snippets, had been flattering everything.

The third thing was about me. My refusal detector for the chat endpoint was
a list of phrases, and it had scored three correct refusals as failures. I
added a phrase each time. That is tuning the measurement until it agrees
with the output. The version that shipped is structural: a refusal cites no
sources, an answer cites at least one.

What I learned:
- Build the golden set before the clever retriever, and make it bigger than
  feels necessary. Fifteen queries overturned two assumptions; 58 overturned
  some of the fifteen's conclusions.
- Key judgements on the source's id, never the database's. A re-ingest
  renumbers everything and the numbers still look fine.
- If a metric is easy to fix by adding a case, the metric is wrong.
- Label the judge untrustworthy until it is calibrated. Mine still is.

Full write-up, with the table where hybrid came third:
[link to post 1]

Repo: https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant

---

## Post 2: the fine-tune that lost to a regex

I fine-tuned a 0.5B model to extract skills from job postings, and an
80-line regular expression is what shipped.

The first run scored 0.000. The model returned an empty list for all 60
test postings. Validation loss had fallen smoothly the whole time.

Six checks later: the loss-masking flag I set was silently ignored because
my dataset was in the wrong format for it. Zero of 316 positions masked.
97% of every gradient step was teaching the model to recite the prompt back.
The library did not warn. The training log said the flag was on.

Fixing that took it to 0.38. Then I found my benchmark could not fail the
baseline: the gold labels were built from the regex's own output, so its
precision was exactly 1.000 by construction. I had published that as a
result. Rebuilt the gold set blind and the regex dropped to 0.986, because it
had matched "excel" inside a Portuguese "Excelência". That 0.014 was real.

Final table on 60 postings: regex 0.66 F1 at 5 ms, 8B teacher 0.63 at
9.5 seconds, fine-tuned student 0.49 at 4.2 seconds. The regex cannot
hallucinate. The teacher invented PyTorch on 12 postings that never
mentioned it.

Five things I would tell anyone starting a distillation project:
1. Correct the teacher before you copy it.
2. Print the labels tensor once.
3. Select checkpoints on generated output, not loss.
4. A perfect score is a reason to check the scorer.
5. Keep the failed row in the table.

Full write-up, including the six checks:
[link to post 2]

Repo: https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant

---

## Post 3: messy data

The most discriminative token in my job-postings corpus was the base64 of
my own IP address.

RemoteOK appends an anti-bot canary to every posting it serves. It was
identical across one source and absent from the other, so k-means built a
99-posting cluster around it in seconds. Nothing errored. I caught it only
because one of the top terms was unpronounceable.

Eight things 468 real postings taught me that a Kaggle CSV never would:

1. A token constant within a source will dominate any unsupervised method.
2. Clean at feature time, keep the raw bytes. I have re-parsed the whole
   corpus twice without a single new network request.
3. A parser's skip count is a metric. 34 of 400 Hacker News comments are
   not postings; if that becomes 100, something broke.
4. Only 23% of postings state a salary. The best of five models is still
   off by about $49k. There is no salary endpoint, on purpose.
5. 37% of locations are unusable. Every aggregate on the dashboard carries
   its coverage next to it.
6. Two dedup methods found 26 pairs each with only 19 in common. That is
   information, not a bug.
7. Never key anything on an auto-increment id. A re-ingest renumbers it.
8. Word boundaries are ASCII. "excel" matched inside "Excelência".

Full write-up: [link to post 3]

Repo: https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant
