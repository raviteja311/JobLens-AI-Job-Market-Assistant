"""Interactive blind adjudication of the gold v2 candidates.

    python -m joblens.finetune.adjudicate_cli
    python -m joblens.finetune.adjudicate_cli --pass two --sample 20

Shows one posting at a time: the text exactly as `dataset.render()` produces
it, then the pooled candidate terms in shuffled order. It never prints which
system proposed a term, and it never reads the provenance file.

For each candidate: y keeps it, n drops it. After the candidates, a free
text prompt takes any skill the posting names that nobody proposed. Saves
after every posting, so the pass can be interrupted.
"""

from __future__ import annotations

import argparse
import random

from joblens.finetune import dataset, gold


def _judge_posting(
    key: str, text: str, terms: list[str], index: int, total: int
) -> dict:
    print("\n" + "=" * 78)
    print(f"[{index}/{total}]  {key}")
    print("=" * 78)
    print(text)
    print("-" * 78)

    keep, drop = [], []
    for term in terms:
        while True:
            answer = input(f"  keep '{term}'? [y/n] ").strip().lower()
            if answer in ("y", "yes"):
                keep.append(term)
                break
            if answer in ("n", "no"):
                drop.append(term)
                break
            print("    answer y or n")

    extra = input("  skills nobody proposed (comma separated, blank for none): ")
    added = [s.strip().lower() for s in extra.split(",") if s.strip()]
    return {"keep": keep, "drop": drop, "added": added}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass", dest="which", default="one", choices=["one", "two"])
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="judge a random subset. Used for the second pass agreement check.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    examples = dataset.load(dataset.TEST_FILE)
    candidates, _ = gold.load_pool()  # provenance is loaded and discarded
    path = gold.DECISIONS_FILE if args.which == "one" else gold.SECOND_PASS_FILE
    decisions = gold.load_decisions(path)

    keys = [e.key for e in examples]
    if args.sample:
        rng = random.Random(args.seed)
        keys = rng.sample(keys, min(args.sample, len(keys)))
    todo = [k for k in keys if k not in decisions]
    print(f"pass {args.which}: {len(todo)} postings to judge, {len(decisions)} done")

    by_key = {e.key: e for e in examples}
    for i, key in enumerate(todo, start=1):
        example = by_key[key]
        text = dataset.render(example.title, example.description)
        decisions[key] = _judge_posting(
            key, text, list(candidates.get(key, [])), i, len(todo)
        )
        gold.save_decisions(decisions, path)
        print(f"  saved ({len(decisions)} of {len(keys)})")

    print(f"\ndone. {len(decisions)} postings judged, written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
