"""
fever_sample.py — turn a FEVER split into a gold set evaluation.py can score.

WHY A SAMPLE AND NOT THE WHOLE SPLIT. FEVER's shared_task_dev.jsonl holds 19,998
claims. At roughly 150 seconds of verification per claim on CPU that is 34 days
of compute, so the question is not "how much of FEVER can we use" but "how few
claims still give a number worth reporting". A stratified 30 per label — 90
claims, about 3.75 hours — is the smallest set that puts a usable count in every
cell of a 3x3 confusion matrix.

Start smaller than you think. `--per-label 5` runs in ~35 minutes and will
surface format problems before you commit an afternoon to a real run.

USAGE

    1. Download the labelled dev split (no registration):
         https://fever.ai/download/fever/shared_task_dev.jsonl

    2. Convert a sample:
         python -m benchmarks.fever_sample --input shared_task_dev.jsonl --per-label 30

    3. Run and score it:
         python -m evaluation run   --gold fever_gold.json
         python -m evaluation score --gold fever_gold.json

WHAT THIS EVALUATION DOES AND DOES NOT MEASURE

    DOES     retrieval and verification. FEVER labels are the exact strings this
             pipeline emits — SUPPORTS / REFUTES / NOT ENOUGH INFO — so no
             mapping is needed and nothing is lost in translation.

    DOES NOT extraction. FEVER claims are already single atomic sentences, so
             the extractor mostly passes them through. Do not report a FEVER
             number as an end-to-end score; it is a verifier score.

    CAVEAT   FEVER's evidence was annotated against a June 2017 Wikipedia dump
             and this pipeline queries live Wikipedia. Label scoring is
             unaffected, but a claim whose supporting sentence has since been
             rewritten may be genuinely unverifiable today. Some disagreement
             with gold is the corpus moving, not the system failing.

    CAVEAT   FEVER's NOT ENOUGH INFO means "annotators found no evidence". This
             pipeline's means "retrieved evidence was inconclusive". Related,
             not identical.
"""

import argparse
import json
import random
import re
from pathlib import Path

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Enough to strip the words that identify nothing. Deliberately not spaCy: this
# script should run on a clone with no model downloaded.
_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "by", "with", "from", "as", "and", "or",
    "but", "not", "no", "it", "its", "he", "she", "they", "them", "his", "her",
    "their", "that", "this", "these", "those", "has", "have", "had", "did",
    "does", "do", "only", "also", "than", "then", "there", "which", "who",
    "what", "when", "where", "how", "into", "about", "over", "after", "before",
}


def match_tokens(claim: str, count: int = 3) -> list[str]:
    """
    A few distinctive lowercase tokens that an extracted claim must contain.

    WHY NOT COMPARE THE WHOLE STRING. The extractor rewrites — it resolves
    pronouns, drops appositives, and normalises phrasing — so exact matching
    would score wording rather than correctness. A short list of rare tokens
    survives rewriting while still being specific enough not to match a
    different claim.

    Proper nouns first, because a capitalised token in a FEVER claim is almost
    always the entity the claim is about and is the least likely thing to be
    paraphrased away. Long words next, as a proxy for rarity.

    THIS IS THE ONE PLACE THIS SCRIPT CAN QUIETLY DISTORT A SCORE. Tokens that
    are too rare produce coverage misses on claims that were extracted fine,
    which reads as an extraction failure when it is a matching failure. If the
    report shows implausibly low extraction recall on FEVER — where the input is
    already a single atomic claim and recall should be near 1.0 — suspect this
    function before suspecting the pipeline.
    """
    words = re.findall(r"[A-Za-z0-9']+", claim)

    proper = [w.lower() for w in words[1:] if w[:1].isupper() and len(w) > 2]
    ordinary = sorted(
        (w.lower() for w in words
         if w.lower() not in _STOPWORDS and len(w) >= 5 and not w[:1].isupper()),
        key=len,
        reverse=True,
    )

    picked: list[str] = []
    for token in proper + ordinary:
        if token not in picked:
            picked.append(token)
        if len(picked) >= count:
            break

    # Never return nothing: a claim of only short common words still has to be
    # matchable, so fall back to its longest tokens whatever they are.
    if not picked:
        picked = sorted({w.lower() for w in words}, key=len, reverse=True)[:2]

    return picked


def convert(records: list[dict], per_label: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {label: [] for label in LABELS}
    for record in records:
        label = record.get("label")
        if label in buckets:
            buckets[label].append(record)

    rng = random.Random(seed)
    gold = []
    for label in LABELS:
        pool = buckets[label]
        if not pool:
            print(f"  WARNING no {label} claims found")
            continue
        if len(pool) < per_label:
            print(f"  WARNING only {len(pool)} {label} claims available")
        # Fixed seed so two people converting the same file get the same
        # sample, and so a re-run compares against the same claims.
        for record in rng.sample(pool, min(per_label, len(pool))):
            gold.append({
                "id": f"fever_{record['id']}",
                "text": record["claim"],
                "gold": [{
                    "match": match_tokens(record["claim"]),
                    "label": label,
                    "note": f"FEVER {record['id']}",
                }],
            })

    rng.shuffle(gold)
    return gold


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--input", required=True,
                    help="path to shared_task_dev.jsonl")
    ap.add_argument("--per-label", type=int, default=30,
                    help="claims per label (default 30 -> 90 total, ~3.75h)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="fever_gold.json")
    args = ap.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"  {source} not found.")
        print("  Download: https://fever.ai/download/fever/shared_task_dev.jsonl")
        return 1

    records = []
    with source.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Read {len(records)} claims from {source.name}")

    gold = convert(records, args.per_label, args.seed)
    Path(args.output).write_text(json.dumps(gold, indent=2), encoding="utf-8")

    estimate = len(gold) * 150 / 3600
    print(f"\n  Wrote {len(gold)} cases to {args.output}")
    print(f"  Estimated run time: ~{estimate:.1f} hours at ~150s/claim on CPU\n")
    print("  Sample of what was written:")
    for case in gold[:3]:
        print(f"    [{case['gold'][0]['label']:16}] match={case['gold'][0]['match']}")
        print(f"                       {case['text'][:70]}")

    print(f"\n  Next:  python -m evaluation run   --gold {args.output}")
    print(f"         python -m evaluation score --gold {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
