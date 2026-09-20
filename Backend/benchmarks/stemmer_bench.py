"""
stemmer_bench.py — which normaliser should _content_words use?

WHY THIS FILE EXISTS. claim_extractor.CONTENT_WORD_MODE is set to
"lemma_keep_propn" rather than to the suffix stripper it shipped with, and a
constant set on somebody's judgement is a constant nobody can argue with. This
reproduces the comparison, so the choice is evidence rather than preference —
and so that a future change (a new stemmer, a different language) is measured
against the same yardstick instead of a remembered impression.

Run from Backend/:

    python -m benchmarks.stemmer_bench
    python -m benchmarks.stemmer_bench --verbose    # show every failing item

No Ollama and no network. spaCy's en_core_web_sm is required for the two lemma
candidates; NLTK for Snowball and Lancaster. Missing dependencies are reported
and skipped rather than crashing, so the file still runs on a fresh clone.

HOW TO READ THE TABLE
    A collapse   % of inflectional pairs that normalise together.
                 Low = the faithfulness gate rejects faithful rewrites.
    B distinct   % of unrelated pairs that stay apart.
                 Low = invented vocabulary slips past the gate.
    C preserve   % of names and -s singulars left unchanged.
                 Low = names stop matching their own mentions in evidence.

No single column decides it. Lancaster is included precisely because it scores
well on A and is unusable — it is the control that proves one column is not
enough.
"""

import argparse
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from benchmarks.stemmer_data import COLLAPSE_PAIRS, DISTINCT_PAIRS, PRESERVE_WORDS


# ─────────────────────────────────────────────────────────────────────────────
# Candidates
# ─────────────────────────────────────────────────────────────────────────────
# Each candidate is a function str -> str, normalising ONE word. The two spaCy
# candidates need a parse to get a POS tag, which is why they are slower and why
# the production code batches whole documents rather than calling per word.
_STEM_SUFFIXES = ("ing", "es", "ed", "s")
MIN_STEM_LENGTH = 4


def suffix_stem(word: str) -> str:
    """The stripper claim_extractor shipped with. The baseline being replaced."""
    w = word.lower()
    for suffix in _STEM_SUFFIXES:
        if len(w) - len(suffix) >= MIN_STEM_LENGTH and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def build_candidates() -> dict:
    """
    Resolve every candidate that this machine can actually run.

    Reported-and-skipped rather than raising: a fresh clone without NLTK data
    should still be able to run the benchmark and see the comparison it CAN
    make, instead of a traceback.
    """
    candidates = {"suffix_stem": suffix_stem}

    try:
        from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer
        candidates["porter"] = PorterStemmer().stem
        candidates["snowball"] = SnowballStemmer("english").stem
        # Deliberately included as a CONTROL. Lancaster is famously
        # over-aggressive; if the scoring cannot show that, the scoring is wrong.
        candidates["lancaster"] = LancasterStemmer().stem
    except Exception as e:
        print(f"  [skip] NLTK stemmers unavailable ({e})")

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")

        def spacy_lemma(word: str) -> str:
            return nlp(word)[0].lemma_.lower()

        def spacy_lemma_propn(word: str) -> str:
            tok = nlp(word)[0]
            return tok.text.lower() if tok.pos_ == "PROPN" else tok.lemma_.lower()

        candidates["spacy_lemma"] = spacy_lemma
        candidates["spacy_lemma+propn"] = spacy_lemma_propn
    except Exception as e:
        print(f"  [skip] spaCy unavailable ({e})")

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
def score(fn, verbose: bool = False) -> dict:
    """
    Run one candidate over all three groups.

    IMPORTANT: single words are passed to the spaCy candidates in isolation, so
    the tagger has no sentence context. That understates PROPN detection — in
    production _content_words_lemma parses whole sentences and does better. The
    benchmark is therefore a LOWER BOUND on the lemma candidates, which is the
    safe direction for a comparison that recommends them.
    """
    failures = {"A": [], "B": [], "C": []}

    collapsed = 0
    for a, b in COLLAPSE_PAIRS:
        if fn(a) == fn(b):
            collapsed += 1
        else:
            failures["A"].append(f"{a}->{fn(a)} != {b}->{fn(b)}")

    distinct = 0
    for a, b in DISTINCT_PAIRS:
        if fn(a) != fn(b):
            distinct += 1
        else:
            failures["B"].append(f"{a} + {b} -> {fn(a)}")

    preserved = 0
    for w in PRESERVE_WORDS:
        if fn(w) == w.lower():
            preserved += 1
        else:
            failures["C"].append(f"{w} -> {fn(w)}")

    if verbose:
        for group, items in failures.items():
            for item in items[:12]:
                print(f"      {group}  {item}")
            if len(items) > 12:
                print(f"      {group}  ... and {len(items) - 12} more")

    return {
        "A": 100 * collapsed / len(COLLAPSE_PAIRS),
        "B": 100 * distinct / len(DISTINCT_PAIRS),
        "C": 100 * preserved / len(PRESERVE_WORDS),
        "failures": failures,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="list failing items")
    args = ap.parse_args()

    print(f"\n  {len(COLLAPSE_PAIRS)} collapse pairs, {len(DISTINCT_PAIRS)} distinct "
          f"pairs, {len(PRESERVE_WORDS)} preserve words\n")

    candidates = build_candidates()
    if not candidates:
        print("  no candidates available")
        return 1

    print(f"\n  {'candidate':22} {'A collapse':>12} {'B distinct':>12} {'C preserve':>12}")
    print("  " + "-" * 60)

    results = {}
    for name, fn in candidates.items():
        if args.verbose:
            print(f"\n  {name}:")
        r = score(fn, args.verbose)
        results[name] = r
        print(f"  {name:22} {r['A']:11.0f}% {r['B']:11.0f}% {r['C']:11.0f}%")

    print("\n  Reading the table:")
    print("    A low  -> faithful rewrites get rejected; wasted retries.")
    print("    B low  -> invented words collide with source words; the gate leaks.")
    print("    C low  -> names are mangled and stop matching the evidence.")
    print("    Lancaster is the control: strong on A, unusable overall.")

    if "spacy_lemma+propn" in results and "spacy_lemma" in results:
        d = results["spacy_lemma+propn"]["C"] - results["spacy_lemma"]["C"]
        print(f"\n  PROPN skip is worth {d:+.0f} points on column C — that gap is the "
              f"whole\n  argument for it, and it is invisible in columns A and B.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
