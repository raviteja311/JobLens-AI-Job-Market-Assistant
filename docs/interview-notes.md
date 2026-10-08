# Interview notes

The questions an interviewer is most likely to ask about JobLens, each with
the facts, the evidence from this repo, and the follow-ups that usually come
next.

**How to use this file.** The facts and evidence below were gathered and
measured by Claude Opus 5.5 on 2026-10-08. The "My answer" sections are
deliberately empty: write each one yourself, in three to five sentences, in
your own words. Then say it out loud without the page. If you cannot, you do
not know it yet, and the evidence section is where to go back to.

Numbers marked **provisional** were scored against LLM grades, before the
human golden set existed. Re-run the scripts named next to them once the
human grades are merged, and quote those instead.

---

## 1. "Why RRF, and why k = 60?"

**Facts.**

- Reciprocal rank fusion scores a document as the sum of `1 / (k + rank)`
  over the ranked lists it appears in (`reciprocal_rank_fusion` in
  `src/joblens/search/retrieval.py`). It uses ranks, not scores, so it needs
  no calibration between `ts_rank_cd` (around 0.1) and cosine similarity
  (around 0.6).
- k = 60 is the constant from Cormack, Clarke and Buettcher (2009), where it
  worked well across TREC collections. Nothing about this corpus chose it.
- Small k trusts each list's top ranks: rank 1 is worth far more than rank 5.
  Large k flattens that, so a posting both retrievers rank moderately well
  beats one that only a single retriever ranks first.

**Evidence** (provisional, 23 queries, `scripts/rrf_sweep.py`):

| k | recall@10 | MRR | nDCG@10 |
| ---: | ---: | ---: | ---: |
| 1 | 0.426 | 0.604 | 0.437 |
| 10 | 0.432 | 0.625 | 0.448 |
| 30 | 0.374 | 0.592 | 0.413 |
| 60 (shipped) | 0.374 | 0.596 | 0.412 |
| 100 | 0.374 | 0.596 | 0.411 |
| 300 | 0.371 | 0.586 | 0.407 |

The curve is flat from 30 upwards and k = 10 is a little better, by 0.03 MRR,
which is within noise at 23 queries. A more important number sits next to
it: on the same queries **vector search alone beats hybrid** (MRR 0.696
against 0.596; keyword alone is 0.431). See question 6.

**Likely follow-ups.** Why not a weighted sum of scores? What would you
change if one retriever were much better than the other? Did you tune k on
the same queries you report on?

**My answer.**

>

---

## 2. "What HNSW parameters did you use, and do they matter?"

**Facts.**

- `migrations/002_search.sql` creates the index with pgvector's defaults:
  `m = 16` (links per node; more is more accurate and uses more memory) and
  `ef_construction = 64` (the build-time candidate list; larger builds a
  better graph, more slowly). At query time `hnsw.ef_search` defaults to 40
  (the search-time candidate list; larger is better recall, slower).
- pgvector 0.8.6 on this database. One HNSW index over all of
  `posting_chunks` (697 `whole` chunks plus 4,907 `section` chunks), because
  pgvector cannot index a subset; the `strategy` and `model` columns are
  filtered separately.

**Evidence** (measured, no labels needed):

- **For the `whole` strategy, Postgres does not use the HNSW index at all.**
  `EXPLAIN` shows a btree filter to the 697 `whole` rows and an exact sort.
  At this size an exact search is cheap, so the default retrieval path is
  exact and `ef_search` cannot affect it.
- **For the `section` strategy it does use HNSW, and filters after the index
  scan.** The index returns at most about `ef_search` nearest chunks, the
  `strategy = 'section'` filter then drops some, and `vector_search` asks for
  `limit * 4` chunks. With the default `ef_search = 40`, a request for 80
  chunks returns 37 on average (29 at worst). At the posting level, over the
  58 golden queries:

| postings requested | default ef_search = 40 | queries short | overlap with exact |
| ---: | --- | ---: | ---: |
| 10 | 10 | 0 / 58 | 0.960 |
| 20 | 13 to 20 | 11 / 58 | 0.909 |
| 30 (the reranker's pool) | 13 to 30, mean 24.9 | 38 / 58 | 0.798 |
| 50 (hybrid's candidates per arm) | 13 to 50, mean 25.8 | 58 / 58 | 0.516 |

- `ef_search = 100` returns the full count, identical to exact search, at
  about 27 ms a query. pgvector 0.8's `hnsw.iterative_scan = relaxed_order`
  at `ef_search = 40` also returns the full count, at 0.99 overlap with exact
  and about 5 ms. Exact search on this table takes about 28 to 38 ms.
- **Fixed on 2026-10-08** with `set local hnsw.iterative_scan =
  relaxed_order` inside `vector_search`, chosen over `ef_search = 100`
  because that only matched exact search by costing as much as an exact scan
  (about 78 ms against 31). A regression test forces the index plan and
  returns 0 of 50 postings without the fix.
- **The fix did not improve the scores.** On 23 LLM-judged queries vector
  (section) is flat (MRR 0.632 to 0.631) and hybrid (section) slightly lower
  (recall@10 0.439 to 0.399, within noise). It is a correctness fix: the
  search now returns what it is asked for. The Phase 3 reranker experiments
  were never affected; they all use `whole`.
- A measurement trap found on the way: psycopg prepares a statement after 5
  executions and keeps its plan. Measuring "exact" and "HNSW" on one
  connection silently reused the exact plan and showed no loss at all. Each
  setting needs its own connection.

**Likely follow-ups.** Pre-filter vs post-filter in vector search, and how
pgvector handles it. When is HNSW not worth it? Why not a partial index or a
separate table per strategy? How did you notice? Why ship a fix that did not
raise the metric?

**My answer.**

>

---

## 3. "Why did the reranker lower recall@10?" (v1) / "What did reranking do here?"

**Facts.**

- A cross-encoder can only reorder what the first stage hands it
  (`rerank_hits` in `src/joblens/search/rerank.py`). A relevant posting
  outside the pool cannot be promoted back, so reranking can push a different
  set into the top ten and lower recall.
- On the v1 global corpus it did: hybrid + rerank had recall@10 0.553 against
  0.653 for plain vector search, and cost about 1,956 ms a query.
- The shipped reranker read a short snippet (title, company, location and the
  first stage's snippet), not the posting.

**Evidence** (provisional, 23 queries, `scripts/rerank_experiments.py`,
`docs/experiments.md` 2026-10-08):

- On this corpus reranking **raised** recall@10, 0.374 to 0.506, because the
  first stage is weak (question 6).
- What the cross-encoder reads mattered most: the best-matching section beat
  the snippet on every metric (MRR 0.763 against 0.729) and was faster.
- A pool of 30 beat 20 and 50; blending in the first-stage score hurt,
  monotonically in alpha.
- TinyBERT-L-2 matched MiniLM-L6 on MRR (0.728 against 0.729) at about a
  seventh of the latency. With `max_length = 256` it is the only
  single-change configuration that passes the pre-registered rule (MRR at
  least +0.05 over hybrid, p95 under 500 ms): +0.129 at about 320 ms.
- The rule and its tie-break were written down before the results that
  decide them (`docs/experiments.md`).

**Likely follow-ups.** Bi-encoder vs cross-encoder. Why does a bigger pool
make it worse? What is pre-registration and why bother? Is 23 queries enough
to trust +0.05?

**My answer.**

>

---

## 4. "Your relevance labels were LLM-made. Why should anyone trust them?"

**Facts.**

- v1: every judgement was made by an LLM, yet each query was marked
  `verified: true`. Worse, the pool showed graders only the first 1,400
  characters of each posting (the median posting is about 5,200), while the
  script's docstring promised the full text.
- Now: `verified: true` means a human graded every judgement on that query,
  enforced by a required `--grader human|llm` flag in
  `scripts/golden_apply.py`. The pool prints the full text. A labelling guide
  (`docs/labelling-guide.md`) defines the grades, a "one step" rule per
  constraint, and was revised twice from graders' reported ambiguities before
  any human grading.

**Evidence.**

- Effect of the cut-off text, LLM against LLM on the same 607 pairs: exact
  agreement 0.802, linearly weighted kappa 0.680 (confounded: the guide was
  revised between the two passes). The full-text pass found 94 grade-2
  postings where the cut-off pass found 68.
- Effect of the last guide revision: weighted kappa 0.896 between the two
  full-text passes.
- Still to measure: **your self-agreement kappa** (blind re-grade of 10
  queries three days later) and **human against LLM kappa** on the 25-query
  subset, both with `scripts/golden_agreement.py`.

**Likely follow-ups.** What is Cohen's kappa and why weighted? What kappa
would make you trust the LLM? How did you pick the 25 queries? How long did
grading take?

**My answer** (fill in the two kappas first).

>

---

## 5. "Your LoRA student (0.485 F1) lost to a regex (0.663). Why, and why keep it?"

**Facts** (`docs/experiments.md`, 2026-09-18 and 2026-09-19).

- Skill extraction against a fixed 80-term vocabulary is closed-set matching,
  which suits a dictionary: the regex has precision 0.986 and runs in 5 ms.
- The student is Qwen 0.5B with a LoRA adapter (rank 16, alpha 32), distilled
  from llama3.1 8B labels.
- The first runs collapsed to answering "empty" on every posting (micro F1
  0.000). The cause: loss was computed on the prompt as well as the answer.
  `completion_only_loss=True` had no effect; printing the labels tensor showed
  0 of 316 positions masked. **Loss masking** means setting the prompt
  positions' labels to -100 so only the answer tokens carry loss.
  `assistant_only_loss=True` fixed it (0.000 to 0.383), and
  `verify_masking()` now checks it before every run.
- Better, less empty-heavy training data took it to 0.485 micro F1, with
  macro F1 0.710, above the teacher's 0.682.
- Eval loss could not tell the collapsed run from a working one (it fell
  neatly in both); F1 could.
- Decision: `skill_extractor = "rules"` in `config.py`. The experiment's
  value was telling you not to ship the model.

**Likely follow-ups.** What is LoRA? Why distil at all? What would make the
student win? Why did loss look fine while the model was broken?

**My answer.**

>

---

## 6. "Why is your hybrid search worse than vector search alone?"

**Facts and evidence** (provisional, same 23 queries):

| retriever | recall@10 | MRR | nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| keyword | 0.313 | 0.431 | 0.293 | 54 |
| vector (whole) | 0.472 | 0.696 | 0.536 | 30 |
| hybrid (whole) | 0.374 | 0.596 | 0.412 | 169 |

- Keyword search ORs the query terms and ranks with `ts_rank_cd` without
  length normalisation (`keyword_search` in `retrieval.py`). Very long
  postings that say "data", "analyst" and "engineer" many times outrank
  precise matches: in the labelling pool the same 6 or 7 long postings
  appeared for 18 to 24 of the 58 queries, all from the keyword arm.
- RRF gives the keyword arm an equal vote, so its bad top ranks pull good
  vector results down.
- Hybrid is slower than both arms together (169 ms against 54 + 30): it asks
  each arm for 50 candidates, and `ts_headline` over 50 long postings is
  expensive.
- `ts_rank_cd` normalisation flags were tried on 2026-10-08 and none helped
  city queries (question 7). Weighting the arms is not yet tried.

**Likely follow-ups.** When would keyword search beat vectors? Would you
remove the keyword arm? How would you fix the length bias?

**My answer.**

>

---

## 7. "Search for 'data engineer Pune' returned Hyderabad jobs. Why, and what did you do?"

**Facts** (`docs/experiments.md`, 2026-10-08).

- Found by manual testing, not by the eval: the golden set had city queries,
  but relevance metrics on 23 queries hid it.
- Root cause: the `location` field was not indexed. Keyword search read
  title, company and description; embeddings read "Title at Company" and the
  description. Only 28 of 93 Pune postings repeat "Pune" in the text.
- Fix 1: location into the search vector (migration 005, weight B) and into
  every chunk header. City precision@10 for hybrid 0.28 to 0.36.
- Tried and rejected: `ts_rank_cd` length normalisation; no flag helped.
- Fix 2: query understanding. `places.query_city` reads the city out of the
  query and in-city postings in the fused top 30 get a third RRF vote. Hybrid
  city precision 0.36 to 0.68; Pune 0.1 to 0.6 (7 of 10 live).
- The trade-off was measured: a wider window wins more city matches but
  promotes in-city jobs in the wrong role (the Chennai case).
- Why relevance numbers could not judge it: pooling bias, 17 to 31% of new
  results unjudged.

**Likely follow-ups.** Why a boost and not a filter? What is pooling bias and
how do you fix it? How would this scale to "near Pune" or "Maharashtra"? Why
not let the LLM parse the query?

**My answer.**

>

---

## General defence: explain any function

An interviewer may open a file and point at a function. Practise with a
friend picking at random and asking "what does this do, and why is it built
this way?"

| file | functions |
| --- | --- |
| `search/retrieval.py` | `keyword_search`, `vector_search`, `reciprocal_rank_fusion`, `hybrid_search`, `search` |
| `search/rerank.py` | `RerankConfig`, `rerank_hits`, `_document`, `_texts`, `_normalised`, `_load` |
| `search/chunking.py` | `chunk_whole`, `chunk_sections`, `chunk` |
| `search/index.py` | `build_index`, `index_stats` |
| `search/dedup.py` | `candidates`, `record`, `compare_to_hash` |
| `sources/base.py` | `client`, `get_json`, `_retry_delay` |
| `sources/boards.py` | `fetch_boards`, `company_of`, `parse_iso` |
| `sources/filters.py` | `is_india`, `is_target_role` |
| `sources/greenhouse.py`, `lever.py`, `ashby.py` | `fetch`, `to_posting`, and each one's quirks |
| `sources/companies.py` | `for_provider`, `by_slug` |
| `db.py` | `insert_raw` (change-only bronze, hashes before payloads), `upsert_postings` |
| `salary.py` | `parse_salary`, `_expand_indian_units` (dormant, but you wrote it) |

Design choices worth being able to defend without notes: filtering at
`to_posting` rather than `fetch` (bronze keeps everything, so `transform`
re-filters offline); `limit` counting usable jobs in `fetch_boards`; storing
a raw payload only when its hash changes; judging on `(source, source_id)`
rather than `postings.id`.
