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

> I built JobLens to run my own AI job search. It collects real postings
> from Hacker News and RemoteOK every day, about 470 right now, and lets me
> search them by meaning, ask questions with citations, and match them
> against my resume.

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

*(Checked on 2026-09-26 against 466 postings. Postings expire, so run both
searches once before recording; `Elixir`, `JAX` and `CUDA` showed the same
effect as backups. The README's original example, `pgvector`, no longer has
a posting.)*

> That is why hybrid is the default, even though on ordinary queries
> it scores worse. Which brings me to the part I am proudest of.

## 0:50 - 1:25 The numbers

*Screen: README retrieval table.*

> Every configuration is scored against a golden set: 58 real queries,
> nearly 1,400 candidate postings, each graded for relevance from the full
> text. Plain vector search wins recall and nDCG. The reranker wins exactly
> one column, at 130 times the latency. Hybrid came third, which is not what
> any blog post told me. This eval runs in CI and fails the build if
> retrieval gets worse.

*Optional, 10 seconds: the Actions tab with a green deploy run.*

## 1:25 - 1:55 Chat that knows when to say no

*Screen: Chat tab. Ask* `which companies are hiring Rust engineers?`*.
Cut to the pre-recorded answer.*

> Chat answers from the postings and cites every claim. (That took about 50
> seconds on a local model, so I cut the wait.)

*Ask* `what is the capital of Peru?`*, show the refusal.*

> And when the postings cannot answer, it says so instead of guessing. A
> quarter of the chat test set is questions like this, and refusal accuracy
> is gated in CI.

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
no deployment yet, and the README says so. "About 470 postings" is safe as
the corpus grows; check the header metric on recording day and say that
number instead if it has moved a lot.
