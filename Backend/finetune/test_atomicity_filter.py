# ─────────────────────────────────────────────────────────────────────────────
# test_atomicity_filter.py — Validate is_compound_fact() BEFORE trusting it
# to filter your overnight distillation run.
#
# Two checks:
#   1. Hand-labeled cases: known-compound sentences (should be flagged) and
#      known-atomic sentences that contain 'and'/relative clauses/long
#      qualifiers but are still single-predicate (should NOT be flagged).
#      Prints a pass/fail table + accuracy.
#   2. Retrospective scan of finetune/data/train.jsonl (if present): reports
#      what fraction of your ALREADY-COLLECTED facts would be flagged, and
#      prints samples so you can eyeball whether the filter agrees with your
#      own judgment on real teacher output — not just the hand-built cases.
#
# Run:
#   cd Backend
#   python finetune/test_atomicity_filter.py
#
# Exit code 0 if labeled accuracy >= 85%, else 1 (so this can gate a CI step
# or a "should I trust this filter" go/no-go before a long run).
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import is_compound_fact  # noqa: E402

ACCURACY_THRESHOLD = 0.85

# (fact, expected_is_compound)
LABELED_CASES: list[tuple[str, bool]] = [
    # ── True positives: two coordinated predicates (SHOULD be flagged) ─────
    ("Gustave Eiffel designed the tower and built the internal framework of the Statue of Liberty.", True),
    ("Marie Curie was born in Warsaw and won two Nobel Prizes.", True),
    ("The Eiffel Tower was completed in 1889 and became a global icon.", True),
    ("Napoleon was crowned emperor in 1804 but was exiled to Elba in 1814.", True),
    ("The bridge was built in 1930 while the tunnel was completed in 1935.", True),
    ("Istanbul was founded by Greek colonists and later became the Ottoman capital.", True),
    ("Tesla invented the AC motor and also worked for Edison.", True),
    ("The novel was published in 1926 and was adapted into a film in 1974.", True),
    ("The tower was designed by Eiffel and is 330 meters tall.", True),
    ("She studied physics in Paris and later taught at the Sorbonne.", True),

    # ── True positives: subordination with its own subject (new) ───────────
    ("The bridge was built in 1930 while the tunnel was completed in 1935.", True),  # the earlier miss
    ("Napoleon remained popular in France although he was defeated at Waterloo.", True),
    ("The east wing was completed in 1990 whereas the west wing opened in 2005.", True),
    ("Ticket sales rose sharply though attendance at away games declined.", True),

    # ── True negatives: single predicate despite length/'and'/clauses ──────
    ("William Meikleham LLD was Regius Professor of Astronomy at the University of Glasgow from 1799 to 1803.", False),
    ("Alexandre Gustave Eiffel and his company built the Eiffel Tower.", False),          # compound subject
    ("The museum contains paintings and sculptures.", False),                              # compound object
    ("Marie Curie, who was born in Warsaw, won two Nobel Prizes.", False),                 # relative clause
    ("The Statue of Liberty, which was designed by Bartholdi, was dedicated in 1886.", False),
    ("Istanbul is the largest city in Turkey.", False),
    ("The bridge, built in 1973, connects two continents.", False),                        # participial phrase
    ("The Eiffel Tower is named after the engineer Gustave Eiffel.", False),
    ("Bartholdi and Eiffel collaborated on the Statue of Liberty.", False),                 # compound subject
    ("The company that built the tower was founded in 1866.", False),                      # relative clause
    ("The device operates while stationary.", False),                                       # elliptical adverbial, no 2nd subject/verb
    ("Sales grew steadily since the product launched in 2015.", False),                      # "since" deliberately excluded (ambiguous temporal/causal)
    ("The museum, while under renovation, remained partially open.", False),                 # "while" but no finite verb in the clause
]


def run_labeled_test() -> float:
    print("=== Labeled test cases ===")
    correct = 0
    for fact, expected in LABELED_CASES:
        actual = is_compound_fact(fact)
        ok = actual == expected
        correct += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] expected={str(expected):5} got={str(actual):5} | {fact}")
    acc = correct / len(LABELED_CASES)
    print(f"\nAccuracy: {correct}/{len(LABELED_CASES)} ({acc:.0%})")
    return acc


def scan_existing_data() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "train.jsonl"
    if not data_path.exists():
        print("\n(no existing train.jsonl found — skipping retrospective scan)")
        return

    facts, flagged = [], []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            c = row["completion"].strip()
            if c.startswith("Fact_"):
                text = c.split(":", 1)[1].strip()
                facts.append(text)
                if is_compound_fact(text):
                    flagged.append(text)

    print(f"\n=== Retrospective scan of {data_path.name} ({len(facts)} facts) ===")
    if not facts:
        print("No Fact_N completions found.")
        return
    pct = len(flagged) / len(facts)
    print(f"{len(flagged)}/{len(facts)} already-collected facts would be flagged as compound ({pct:.1%})")
    print("Sample flagged facts — spot-check these manually, they are what future runs will drop:")
    for f in flagged[:10]:
        print(f"  - {f}")
    if not flagged:
        print("  (none)")


if __name__ == "__main__":
    accuracy = run_labeled_test()
    scan_existing_data()

    if accuracy < ACCURACY_THRESHOLD:
        print(
            f"\n[WARN] Accuracy {accuracy:.0%} is below the {ACCURACY_THRESHOLD:.0%} "
            f"threshold — review the FAIL rows above before trusting this filter "
            f"on new data."
        )
        sys.exit(1)

    print(f"\nFilter passes the {ACCURACY_THRESHOLD:.0%} threshold — safe to use going forward.")
