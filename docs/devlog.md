# Devlog

A short note at the end of each work session: what broke, what I decided,
why. These entries become the blog posts and the interview stories later.
Newest entry at the top.

---

## Phase 8 - packaging, as far as a repo can package itself

**Did:** rewrote the README top as a landing page (pitch, demo, live URL
status, Mermaid architecture, the two results tables, three-command
quickstart), wrote the Playwright script in `scripts/` that records the demo
GIF from the real UI, wrote three blog post drafts from the devlog and
experiments, drafted three LinkedIn posts and a headline, and filled in the
plan's resume bullets with the real numbers.

**Decided:** retitle two of the three blog posts. The plan's titles were
"Fine-tuning a 3B model to replace an API call" and "What scraping 10k job
postings taught me". The model was 0.5B, the teacher was local, and the
corpus is 468 postings. A title that promises a number the post cannot
deliver is the exact thing the rest of this repo argues against. The posts
say what the plan called them and why they are called something else.

**Decided:** the resume bullets file has a section for the three plan
bullets that cannot be claimed. "Matched API accuracy at a fraction of the
cost" did not happen; "served real users with p95 under Z ms" needs a
deployment and users. Writing down what a bullet is not allowed to say is
cheaper than being asked about it in an interview.

**Decided:** the demo GIF records Search and Trends only. Chat and matching
take about 50 seconds an answer on the local model. A GIF that cuts away and
comes back with an answer would be implying a latency the system does not
have. The README states the 50 seconds instead.

**Decided:** the demo is a script, not a one-off recording. The UI will
change, and a GIF from three phases ago is a small lie on the landing page.
`python scripts/demo_gif.py` against the compose stack regenerates it in
about a minute.

**What broke:** Playwright's Python package pinned a browser build that was
not the one already on the machine, and downloading another 150MB browser
for a GIF felt wrong. Playwright can drive the system Edge through
`channel="msedge"`, so the script does that by default and takes a flag for
Playwright's own Chromium.

**What broke, second:** Docker Desktop on this machine stopped starting
mid-phase. Every service crashes renaming its Unix socket file with Windows
error 1920, and the pending Docker Desktop update or a reboot is the fix.
The demo GIF is therefore not recorded yet; the script is, and the README
says so instead of showing a broken image.

**Not done, recorded:** the demo GIF recording (see above), the demo video
(needs a voice), pinning the repo,
publishing the posts, updating LinkedIn and the resume. Those are actions on
accounts, and the drafts are in the repo ready for them. The public dataset
from the stretch goals needs a permissions check with each board first.

**Next:** the owner's decisions from Phase 7 (hosting, Grafana Cloud), then
the stretch goals if there is appetite. The plan is otherwise complete.

---

## Phase 7 - deployment and MLOps, the local half

**Did:** a multi-stage Dockerfile with two targets (API and UI), a compose
stack for Postgres, API, UI, Prometheus and Grafana, structured JSON logs,
a `/metrics` endpoint with request histograms, LLM cost counters and
database-backed ingest gauges, a provisioned Grafana dashboard with three
alert rules, a `deploy` workflow that runs lint, tests and the retrieval
eval before pushing the image to GitHub's registry, an ingestion failure
alert that opens an issue, and `docs/runbook.md`. 12 new tests, 191 total.

**Current state:** everything runs locally with `docker compose --profile
monitoring up`. No public URL yet: that needs a hosting account and a
hosted pgvector database, which are not mine to create.

**What broke:**

- `docker build .` builds the *last* stage of a multi-stage Dockerfile. I
  added the UI stage at the bottom, rebuilt, and the "API" image was the
  159MB Streamlit image with no `joblens` module in it. The API stage is
  now last and the compose file names its target explicitly.
- A YAML folded scalar (`>`) keeps the newline when the continuation line
  is indented more than the first. My `sh -c "migrate && serve"` command
  became two lines and `sh` choked on a line starting with `&&`. The
  command is a list now.
- A Dockerfile heredoc inside `if ... fi` is an "unterminated heredoc" to
  BuildKit. `python -c` with a multi-statement string instead.
- The first API image was 721MB to pull and about 2.1GB unpacked, with a
  1.8GB virtualenv. Torch is 769MB even from the CPU index, and pip had
  left every `.pyc` behind in the builder. Deleting `torch/include`,
  `torch/test` and `__pycache__` took the pull to 577MB and the unpacked
  size to 1.6GB. Splitting the UI into its own image took torch out of
  that one entirely: 159MB to pull, 501MB unpacked.

**Decided:**

- The database gauges are computed on scrape, not kept in process. A
  counter that resets on every deploy cannot answer "did last night's
  ingest run", and that is the one question the on-call alert exists to
  answer. Four indexed queries every 15 seconds is nothing.
- A scrape must never fail because Postgres did. The collector yields
  `joblens_db_up 0` and stops. The alternative, a 500 from `/metrics`, means
  the dashboard goes blank at exactly the moment it is needed.
- Route labels use the route template, never the path. One series per
  query string is how a Prometheus instance runs out of memory, and a
  scanner probing for `/wp-admin` should not get to create a series.
- The `ci` workflow no longer fires on push to `main`; `deploy` calls it as
  a stage instead. Otherwise every merge runs the suite twice for one
  green tick.
- No fallback LLM when the provider is down. An answer from a model the
  eval suite never scored is an unscored answer. `/chat` and `/match` return
  503; search and trends keep working.
- The ingestion alert is a GitHub issue, not only a Grafana rule. Grafana
  runs locally today and the issue is visible from the repo page, which is
  where the owner already looks.

**Not done, recorded:** public deploy (needs the owner's hosting account and
a hosted pgvector Postgres), Grafana Cloud (same), AWS ECS (optional in the
plan, and pointless before a first deploy anywhere). The deploy job is the
one piece of the pipeline that does not exist yet.

**Next:** the owner picks a host. Then Phase 8, packaging.

---

## Phase 6 - fine-tuning, and a benchmark that could not fail

**Did:** distilled a 0.5B skill extractor from a local llama3.1, watched it
collapse, found out why, fixed it, then found out the benchmark I was fixing
it against was rigged in the baseline's favour and rebuilt that too.

**The collapse.** The student answered `{"skills": []}` on 60 of 60 test
postings. Micro F1 exactly 0.000. Eval loss had fallen every epoch, 1.600 to
1.573 to 1.565, and nothing in the training output hinted at a problem.

The cause was one flag. `SFTConfig(completion_only_loss=True)` is supported
only for prompt-completion datasets, and mine is conversational, so trl
ignored it. Silently. Zero of 316 positions were masked, which means about
97% of every gradient step taught the model to recite the system prompt, the
five rules, the schema block and the job posting back to itself. The
completion is eight tokens.

The masked-in text includes the prompt's own schema line,
`{"skills": ["python", "pytorch", "aws"]}`. That is exactly what the untuned
base model returns for a Marketing Student Assistant and a BMS Service
Technician. Neither model was extracting. Both were completing the example.

`training.json` recorded `completion_only_loss: true` for a run where it did
nothing. The config and the log agreed with each other and both were wrong.

**Decided:** never trust a flag that claims to work. `verify_masking()` reads
the first batch back and refuses to train if the prompt is not masked. It
prints 577 of 607 positions masked before a run starts, rather than after it.

**Decided:** select checkpoints on F1, not loss. The whole lesson of the
collapse is that loss fell smoothly into a model scoring zero. The callback
generates on a held-out slice each epoch and logs the empty-prediction rate
next to the F1, and the eval harness now fails loudly above 90% empty. That
guard has caught two things since: the collapse itself, and a teacher row
that came back 0.000 because four Qwen models in one process drove free
memory to 0.2GB and Ollama returned HTTP 500 on every call.

**What the instrumentation earned immediately:** the fixed run scored F1
0.000 at 100% empty after epoch 1, identical to the failure it was fixing,
and only broke out in epoch 2. Judged early, or on loss, I would have
concluded the fix did not work.

**Then the harder problem.** Someone asked why the empty-prediction rate was
so high. Answering it meant reading how gold was built, and gold was built as
`rules(posting) | adjudicated(teacher(posting))`, where adjudication filtered
teacher terms through a hand-reviewed vocabulary that is, in substance, the
regex's own capability spec.

Truth was defined as things the regex could have found. Gold was a superset
of the regex output on all 60 postings, so the regex could not produce a
false positive, and precision came back as exactly 1.000. I had published
that as a finding. It was an identity. A benchmark that cannot falsify the
baseline is not a benchmark.

**Decided:** rebuild gold blind. Candidates from both labellers pooled and
shuffled, provenance written to a separate file and joined back only after
every decision was recorded. Decisions per (posting, term) rather than per
term, because "go" is a language in one posting and a verb in the next. And
an additive pass over all 60 postings, because without it gold stays a subset
of the union and both recall numbers flatter both systems.

206 occurrences judged, 116 kept, 90 dropped, 22 added that neither labeller
had proposed.

**The single most satisfying line in the phase:** one dropped term came from
the regex. On a Portuguese posting it matched `excel` inside `Excelência`,
because the word boundaries in `skills.py` are ASCII and an accented letter
reads as a boundary. Precision went from 1.000 to 0.986, 69 correct out of 70,
and that 0.014 is a real bug that gold v1 made structurally invisible.

**What the circularity was worth:** the regex led the teacher 0.753 to 0.671
under v1 and 0.663 to 0.634 under v2. The additive pass put 22 terms into
gold that an 80-word vocabulary cannot reach. The regex did not get worse;
the measurement got honest.

**Did not do:** the inter-annotator kappa. I judged all 60 postings myself in
one sitting, so a second pass would agree with itself and measure my memory
rather than label quality. The second-pass tooling reads only the candidate
pool, never the provenance or the first-pass file, so an independent run can
produce a real figure. Publishing a self-agreement number as reliability
would have been the same class of mistake as the precision column.

**Cut:** forcing `posting_text` and `render` to be byte-identical. It would
have moved Phase 2's published coverage figure and cluster labels to fix a
Phase 6 bug. The guard asserts what actually broke instead, that the baseline
reads the same text gold was built from, and a third test pins the
divergence as deliberate.

**Cut:** the disagreement filter on training data, though 13% was inside the
permitted range. The no-overlap examples are mostly postings where the regex
is silent because the technology is outside its 80 words and the teacher got
it right. They are the only examples that could teach the student recall, the
one dimension it loses badly on. Filtering them would have been removing
signal and calling it noise.

**Noted elsewhere, not fixed here:** `msonormal` appears 36 times across the
corpus, a Microsoft Word CSS class surviving Phase 1's HTML stripping. A
Hacker News comment that is a blog post rather than a job ad also made it
into the test set. Both are Phase 1 parser gaps.

**Next:** Phase 7. Deployment and MLOps.

---

## Phase 5 - evaluation harness

**Did:** chat QA set, LLM-as-judge on faithfulness and completeness, a
refusal metric, an A/B harness over prompt versions, run history in CSV and
Postgres, and a CI gate that exits non-zero on a regression.

**Decided:** refusal accuracy is reported separately from the judge and is a
hard floor rather than a trend. Half the QA set is questions the corpus
cannot answer. Getting those right is not a nice-to-have; a system that
invents an answer to them is worse than one that answers nothing.

**Decided:** thresholds live in `report.py`, not in the workflow YAML. A
threshold change should appear in a pull request next to the change that
needed it, not in a file nobody reads.

**Tolerance is 0.05, not zero.** The corpus grows daily and these numbers
move a point or two between runs. A gate that fires on noise gets commented
out within a week, and then there is no gate.

**The uncomfortable part, again:** the judge is not calibrated. It needs
twenty answers scored by hand and I have not scored them, so
`Calibration.trustworthy` returns False and the README says the judge numbers
are a smoke test rather than a measurement. Writing the calibration code and
then quoting the uncalibrated scores as if they meant something would have
been the single most dishonest thing in this repo, and it would have been
invisible.

Worse: the judge is the same llama3.1 that wrote the answers. Self-preference
bias at its maximum. The code can report that; only human labels can correct
it.

**What the eval is for, concretely:** in Phase 2 I stripped scraper
boilerplate out of the clustering input. The clusters went from "came from
RemoteOK" to "founding engineer", and the silhouette score got slightly
worse. Optimising the metric would have reverted a correct fix. That is the
argument for this whole phase in one paragraph.

**Chat eval is not in the PR gate.** A local Llama on a GitHub runner takes
half an hour. Retrieval is gated on every PR touching retrieval, prompts or
the golden set; chat runs on demand. A green tick that means "we skipped it"
is worse than no tick.

**Next:** Phase 6, one honest fine-tuning experiment. The Phase 2 skill
dictionary at 58.9% coverage is the number to beat.

---

## Phase 4 - RAG chat and resume matching

**Did:** /chat with citations, /match for resumes, versioned prompt files,
schema-validated structured output with repair retries, full LLM call
logging, a rate limit, the FastAPI backend and a Streamlit UI.

**Decided first, before any endpoint:** the `llm_calls` table. Prompt,
response, tokens, cost, latency, attempt count, prompt version. Writing it
after the fact is impossible, because the questions it answers are all about
calls that already happened.

**Decided:** when retrieval scores below a threshold, the model is never
called at all. The failure mode I was actually afraid of is not a wrong
answer, it is a fluent one. Ask an always-answering RAG system something the
corpus cannot address and you get a confident paragraph about the job market
that a user has no way to distinguish from a real answer. Cheaper, faster and
safer to refuse in code.

Tested it with "what is the capital of Peru?" Retrieval returned five
postings over the threshold, so the model did run, and it said the postings
contain no information about Peru. The prompt held where the threshold did
not, which is two layers doing their job rather than one.

**Decided:** prompts in `prompts/<name>/<version>.md`, never f-strings. Used
`string.Template` and not `str.format`, because these prompts are full of
JSON schema examples and every brace would need doubling. `safe_substitute`
and not `substitute`, because postings are full of dollar signs and a `$120k`
in a description should not raise KeyError mid-request.

**Retries carry the error.** "That did not parse, try again" gets the same
broken output back. "field fit_score must be an integer between 0 and 100,
you sent 'high'" usually does not. The attempt count goes in the log so a
prompt regression shows up as a rising average before anyone notices.

**What surprised me:** llama3.1 on CPU takes about 50 seconds an answer with
six sources. That number changes the design of Phase 5 rather than being an
inconvenience: a 60-question chat eval would take an hour and would therefore
never be run.

**Next:** Phase 5, and the thing that separates this from every other repo.

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