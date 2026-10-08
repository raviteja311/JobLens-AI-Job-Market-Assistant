# v1: global corpus (archived 2026-10-08)

The evaluation state of JobLens before the switch to Indian postings.

- `retrieval.yaml`: the golden set, keyed by Hacker News and RemoteOK posting
  IDs. Relevance was judged by an LLM; only the grade-2 labels were reviewed
  by hand.
- `eval_baseline.json`: the retrieval baseline the CI eval compared against.
- `eval_history.csv`: every eval run on that corpus.

Corpus at archive time: 367 Hacker News and 99 RemoteOK postings (466), from
502 raw payloads. Those postings have expired upstream and cannot be fetched
again; a full `pg_dump` is kept outside the repository.

None of these numbers apply to the Indian corpus. They are kept as history.
