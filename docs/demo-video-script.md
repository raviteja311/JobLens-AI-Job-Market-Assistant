# Demo video script

About 2.5 minutes, screen recording with voice. The plan asks for 2 to 3
minutes and a pinned post. Every number said out loud is in the README, so
nothing in the video needs a caveat the page does not already have.

**Before recording**

- `docker compose up --build`, wait for the API health check, open
  http://localhost:8501. Run one search first so the models are warm; a cold
  first query takes 20 seconds and looks like a hang.
- Have the README open in a second tab, scrolled to the retrieval table.
- Chat and resume match take about 50 seconds an answer on the local model.
  Record one answer ahead of time and cut to it, and say that it was cut
  (the script does).
- Keep a PDF resume ready for the match tab. Use your own; it is the story.

---

## 0:00 - 0:15 The hook

*Screen: the JobLens header with the four metrics (postings, chunks,
embedder, LLM).*

> I built JobLens to run my own AI job search. Every day it collects AI and
> data postings from the job boards of 291 companies that hire in India, about 625 right
> now, and lets me search them by meaning, ask questions with citations,
> and match them against my resume.

## 0:15 - 0:50 Search, and the query embeddings get wrong

*Screen: Search tab. Type* `remote machine learning engineer for someone
with two years experience`*, mode* `vector`*.*

> Semantic search: the results are about the job, not the exact words.

*Change the query to* `vLLM`*, keep* `vector`*.*

> Now a rare term. Two postings name vLLM, and vector search finds neither
> in its top five: to an embedding, "vLLM" is just "something about AI
> infrastructure".

*Switch mode to* `hybrid`*.*

> Hybrid adds keyword search back in, and both postings come first and
> second.

*(Checked on 2026-09-26 against the v1 corpus, not yet on the Indian one.
Before recording, find a rare term that shows the effect today: run
`python -m joblens search "<term>" --mode vector --limit 5` and the same
with `--mode hybrid`, and use a term where hybrid puts the matching
postings first and vector does not. Try `vLLM`, `JAX`, `CUDA` and
`pgvector`. If none shows the effect, cut this beat rather than fake it.)*

*Optional city beat: type* `data engineer Pune`*, mode* `hybrid`*.*

> Name a city and the in-city postings move up. City precision on those
> queries went from 0.28 to 0.68.

> That is why hybrid is the default, even though plain vector search
> scores as well on ordinary queries. Which brings me to the part I am
> proudest of.

## 0:50 - 1:25 The numbers

*Screen: README retrieval table.*

> Every configuration is scored against a golden set I graded by hand: 25
> queries, more than 900 candidate postings. No single configuration wins
> everything. For the reranker I wrote the decision rule down before
> running it: it had to gain five points of nDCG at under half a second.
> Nothing passed, so it stays optional, and a version that looked like a
> winner on model-made grades did not hold up on mine. This eval runs in CI
> and fails the build if retrieval gets worse.

*Optional, 10 seconds: the Actions tab with a green deploy run.*

## 1:25 - 1:55 Chat that knows when to say no

*Screen: Chat tab. Ask a question the current postings can answer, such as*
`which companies are hiring for LLM roles in Bengaluru?`*, after checking
it on recording day. Cut to the pre-recorded answer.*

> Chat answers from the postings and cites every claim. (That took about 50
> seconds on a local model, so I cut the wait.)

*Ask* `what is the capital of Peru?`*, show the refusal.*

> And when the postings cannot answer, it says so instead of guessing. A
> quarter of the chat test set is questions like this, and refusal accuracy
> has a hard floor in the evaluation.

## 1:55 - 2:20 Resume match

*Screen: Resume match tab, upload the PDF, cut to the result.*

> Upload a resume and it ranks postings with a fit score, the skills I
> match, and the ones I am missing. This is the tab I actually use.

## 2:20 - 2:35 Trends, and the close

*Screen: Trends tab, top skills chart.*

> Plus a dashboard of which skills are in demand. Everything is in the repo:
> the eval, a fine-tuning experiment where an 80-line regex beat my model,
> and the write-ups of what went wrong. Link below.

---

**Things not to say:** "production", "users", or a latency figure. There is
no deployment yet, and the README says so. Do not say chat is tested in
CI; only retrieval is. "About 625 postings" is safe as the corpus grows;
check the header metric on recording day and say that number instead if
it has moved a lot.
