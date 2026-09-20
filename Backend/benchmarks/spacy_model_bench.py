"""
spacy_model_bench.py — does a bigger spaCy model change anything that matters?

THE QUESTION THIS ANSWERS IS "SHOULD WE?", NOT "WHICH IS BEST?". en_core_web_trf
is more accurate than en_core_web_sm on every published metric; that is not in
dispute and does not need re-measuring here. What is in dispute is whether that
accuracy reaches the pipeline's OUTPUT. If a better parse produces the same
query, the same subject and the same verdict, the upgrade costs ~2 GB of torch
and buys nothing.

So this benchmark deliberately does NOT score entity labels. Nobody is graded on
entity labels. It diffs the four things that actually leave this module:

    entities        the raw NER output, shown for context only
    subject         drives subject promotion, which heads every query
    QUERY SET       what is literally sent to Wikipedia — the decisive column
    predicate query the intent-aware query for superlative claims

An observed example of why the distinction matters: en_core_web_sm tags
'Jamestown' as PERSON, which looks alarming in a log. It is very possibly
harmless, because subject promotion is driven by the DEPENDENCY PARSE and
overrides entity ordering, so the right string reaches Wikipedia regardless. A
benchmark that scored entity labels would report a problem where none exists,
and would recommend an upgrade on the strength of a cosmetic defect.

It also measures the parse, not just NER — 10 of the 11 spaCy call sites in this
pipeline use POS tags and dependency labels rather than .ents, so a better parse
is the stronger reason to upgrade and the one most likely to be overlooked.

HOW IT WORKS. It swaps ner.NLP for each candidate model and calls the REAL
functions. No reimplementation: whatever quirks extract_queries has, they are
present in every row, so the diff is attributable to the model and nothing else.

Run from Backend/:

    python -m spacy download en_core_web_md      # optional
    python -m spacy download en_core_web_trf     # needs spacy-transformers
    python -m benchmarks.spacy_model_bench
    python -m benchmarks.spacy_model_bench --verbose

Missing models are reported and skipped, so this runs on a fresh clone with only
en_core_web_sm installed — it will simply have nothing to compare against, and
will say so.
"""

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pipeline import ner

# Candidates in increasing order of cost. 'sm' is the baseline every other model
# is diffed against, because it is what the project currently ships.
CANDIDATES = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]

# Claims drawn from real runs, plus shapes that stress the parse rather than the
# entity recogniser. The mislabelled cases are included on purpose: 'Jamestown'
# comes back PERSON from en_core_web_sm, and the point is to find out whether
# that changes a query.
CLAIMS = [
    "The English settlement of Jamestown was founded in May 1607.",
    "The English settlement of Jamestown is celebrated as the earliest European "
    "permanent settlement in what is now the United States.",
    "Jamestown paved the way for continuous European presence on the continent.",
    "The title of the world's longest river belongs to the Amazon River.",
    "The Amazon River discharges more water than any other drainage basin.",
    "The Amazon River is universally acknowledged as longer than the Nile by all "
    "international cartographers.",
    "Bosporus is located between Africa and Europe.",
    "You can walk between Africa and Europe with the Yavuz Sultan Selim Bridge.",
    "The Republic of Sale was a city-state on the coast of Morocco.",
    "Marie Curie, who was born in Warsaw, won two Nobel Prizes.",
    "Mount Everest is taller than K2.",
    "The museum opened in 1932 and holds over 400 paintings.",
]


def load(name: str):
    try:
        import spacy
        t0 = time.perf_counter()
        nlp = spacy.load(name)
        print(f"  [ok]   {name:22} loaded in {time.perf_counter() - t0:.1f}s")
        return nlp
    except Exception as e:
        first = str(e).split("\n")[0][:70]
        print(f"  [skip] {name:22} {first}")
        return None


def probe(claim: str) -> dict:
    """
    Run the real pipeline functions against whatever ner.NLP currently is.

    stdout is swallowed because extract_queries narrates heavily and 12 claims
    times 4 models would bury the comparison in log lines.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        entities = ner.extract_entities(claim)
        subject = ner.extract_subject(claim)
        queries = ner.extract_queries(claim)
        predicate = ner.build_predicate_query(claim)
    return {
        "entities": tuple(entities),
        "subject": subject,
        "queries": tuple(queries),
        "predicate": predicate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print every claim's outputs, not only the differences")
    args = ap.parse_args()

    print(f"\n  {len(CLAIMS)} claims\n")
    models = {name: nlp for name in CANDIDATES if (nlp := load(name)) is not None}

    if len(models) < 2:
        print("\n  Need at least two models to compare. Install one:")
        print("    python -m spacy download en_core_web_md")
        print("    pip install spacy-transformers && python -m spacy download en_core_web_trf")
        return 1

    original = ner.NLP
    results: dict[str, list[dict]] = {}
    timings: dict[str, float] = {}

    try:
        for name, nlp in models.items():
            ner.NLP = nlp
            t0 = time.perf_counter()
            results[name] = [probe(c) for c in CLAIMS]
            timings[name] = time.perf_counter() - t0
    finally:
        # Restore unconditionally. Leaving a benchmark's model installed in the
        # live module would silently change the behaviour of anything importing
        # ner afterwards in the same process.
        ner.NLP = original

    baseline = next(iter(models))

    # ── Differences ───────────────────────────────────────────────────────────
    print(f"\n  DIFFERENCES vs {baseline}\n")
    changed = {name: {"entities": 0, "subject": 0, "queries": 0, "predicate": 0}
               for name in models if name != baseline}

    for i, claim in enumerate(CLAIMS):
        base = results[baseline][i]
        rows = []
        for name in models:
            if name == baseline:
                continue
            other = results[name][i]
            diffs = [k for k in ("entities", "subject", "queries", "predicate")
                     if base[k] != other[k]]
            for k in diffs:
                changed[name][k] += 1
            if diffs or args.verbose:
                rows.append((name, other, diffs))

        if not rows:
            continue

        print(f"  {claim[:76]}")
        print(f"    {baseline:20} queries={list(base['queries'])}")
        print(f"    {'':20} subject={base['subject']!r}  entities={list(base['entities'])}")
        for name, other, diffs in rows:
            mark = ",".join(diffs) if diffs else "identical"
            print(f"    {name:20} queries={list(other['queries'])}   <- {mark}")
            print(f"    {'':20} subject={other['subject']!r}  entities={list(other['entities'])}")
        print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"  {'model':22} {'entities':>9} {'subject':>9} {'QUERIES':>9} "
          f"{'predicate':>10} {'seconds':>9}")
    print("  " + "-" * 74)
    print(f"  {baseline + ' (baseline)':22} {'-':>9} {'-':>9} {'-':>9} {'-':>10} "
          f"{timings[baseline]:8.2f}s")
    for name in models:
        if name == baseline:
            continue
        c = changed[name]
        print(f"  {name:22} {c['entities']:9d} {c['subject']:9d} {c['queries']:9d} "
              f"{c['predicate']:10d} {timings[name]:8.2f}s")

    print(f"\n  Counts are claims (out of {len(CLAIMS)}) whose output differs from "
          f"{baseline}.")
    print("  Per-request cost is roughly these seconds divided by "
          f"{len(CLAIMS)}, times ~10 parses per fact.")

    print("\n  HOW TO DECIDE")
    print("    QUERIES column 0  ->  the upgrade changes nothing that reaches")
    print("                          Wikipedia. Keep en_core_web_sm and save the")
    print("                          ~2 GB torch dependency.")
    print("    QUERIES column > 0 -> inspect the diffs above. A CHANGED query is")
    print("                          not automatically a BETTER one; read them.")
    print("    entities > 0 but QUERIES 0")
    print("                       -> the labels improved and nothing downstream")
    print("                          noticed, because subject promotion is driven")
    print("                          by the dependency parse. This is the outcome")
    print("                          the benchmark exists to detect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
