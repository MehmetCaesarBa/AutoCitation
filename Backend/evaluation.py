"""
evaluation.py — scored evaluation of the AutoCitation pipeline.

DIFFERENT FROM regression.py. That file asserts: a case passes or fails, and its
job is to stop a fix from silently undoing an earlier fix. This file MEASURES:
it produces numbers you can put in a thesis, compare across versions, and lose
ground on visibly.

    python -m evaluation run                 # run the pipeline, cache raw output
    python -m evaluation run --case amazon   # one case
    python -m evaluation score               # score the cache, print the report
    python -m evaluation score --markdown docs/evaluation.md

RUNNING AND SCORING ARE SEPARATE ON PURPOSE. A full run is roughly 500-900
seconds PER CASE on CPU, so a metric definition that can only be changed by
re-running an hour of inference is a metric definition nobody will improve.
`run` writes every raw result to evaluation_results.json; `score` reads that file
and computes everything. Change a formula, re-score in milliseconds.

── WHY THE METRICS ARE SHAPED THIS WAY ──────────────────────────────────────

This system ABSTAINS. NOT ENOUGH INFO is not a wrong answer, it is a refusal to
answer, and it is the correct output whenever retrieved evidence cannot settle a
claim. A plain accuracy score treats a refusal exactly like a wrong verdict,
which would reward a system that guesses — the opposite of what a fact-checker
should do.

So the headline is a PAIR, as in the selective-prediction literature:

    ANSWERED ACCURACY   of the decidable claims it chose to answer, how many
                        did it get right? This is the number that says whether
                        a verdict can be trusted.

    ABSTENTION RATE     of the decidable claims, how many did it decline? This
                        is the number that says how often it is useful.

Neither is meaningful alone. A system answering one claim in twenty at 100%
accuracy is useless; one answering everything at 60% is worse than useless,
because a fact-checker that is wrong 40% of the time launders misinformation
through an authoritative-looking citation.

── AND WHY COVERAGE IS MEASURED SEPARATELY ──────────────────────────────────

A claim that is never extracted can never be labelled, so it cannot appear in
any label metric. During development this pipeline silently dropped whole
sentences — including, on one occasion, the single false claim the input existed
to test — and every label-based score stayed perfect throughout, because the
dropped claim was simply absent from the denominator.

    EXTRACTION COVERAGE   how many gold propositions produced a claim
    EXTRACTION PRECISION  how many produced claims correspond to a gold one

Coverage is the metric that would have caught that class of failure. It is
reported first for that reason.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import Pipeline.claim_extractor as claim_extractor

RESULTS_FILE = Path(__file__).with_name("evaluation_results.json")

DECIDABLE = {"SUPPORTS", "REFUTES"}

# Column and row order for the confusion matrix. Fixed so two reports are
# comparable cell by cell.
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


# ─────────────────────────────────────────────────────────────────────────────
# Gold set
# ─────────────────────────────────────────────────────────────────────────────
# `match` is a list of lowercase substrings that must ALL appear in an extracted
# claim for it to count as covering that gold proposition. Substring matching
# rather than exact text because the extractor's wording varies run to run —
# "Jamestown was founded in May 1607" and "The English settlement of Jamestown
# was founded in May 1607" are the same proposition, and a metric that called
# them different would measure phrasing instead of correctness.
#
# `label` is what the evidence available on Wikipedia SUPPORTS, not what is true
# in the world. Where the sources genuinely disagree the gold label is NOT
# ENOUGH INFO, because a system reporting a contested question as settled is
# wrong even when it happens to pick the popular side.
GOLD = [
    {
        "id": "jamestown",
        "text": (
            "The English settlement of Jamestown was founded in May 1607 and is "
            "celebrated as the earliest European permanent settlement in what is "
            "now the United States. It paved the way for continuous European "
            "presence on the continent."
        ),
        "gold": [
            {"match": ["jamestown", "1607"], "label": "SUPPORTS",
             "note": "Wikipedia gives 14 May 1607."},
            {"match": ["earliest european"], "label": "REFUTES",
             "note": "Saint Augustine, Florida (1565) predates it."},
            {"match": ["paved the way"], "label": "NOT ENOUGH INFO",
             "note": "Vague causal claim; no source states it."},
        ],
    },
    {
        "id": "amazon",
        "text": (
            "The title of the world's longest river belongs to the Amazon River, "
            "which discharges more water than any other drainage basin. It is "
            "universally acknowledged as longer than the Nile by all "
            "international cartographers."
        ),
        "gold": [
            {"match": ["longest river", "amazon"], "label": "NOT ENOUGH INFO",
             "note": "Sources describe the Amazon/Nile length question as disputed."},
            {"match": ["discharges more water"], "label": "SUPPORTS"},
            {"match": ["universally acknowledged"], "label": "REFUTES",
             "note": "Sources record an active scientific dispute."},
        ],
    },
    {
        "id": "python_cpp",
        "text": (
            "Both Python and C++ are widely used object-oriented programming "
            "languages. Python uses automatic garbage collection for memory "
            "management, whereas standard C++ relies primarily on manual memory "
            "management and deterministic RAII rather than a default tracing "
            "garbage collector."
        ),
        "gold": [
            {"match": ["python", "c++", "object-oriented"], "label": "SUPPORTS",
             "note": "Conjunction claim — both halves are true."},
            {"match": ["python", "garbage collection"], "label": "SUPPORTS"},
            {"match": ["c++", "manual memory"], "label": "SUPPORTS"},
        ],
    },
    {
        "id": "empire_state",
        "text": (
            "The Empire State Building was completed in 1931. To this day, the "
            "Empire State Building remains taller than the Burj Khalifa in total "
            "structural height."
        ),
        "gold": [
            {"match": ["empire state", "1931"], "label": "SUPPORTS"},
            {"match": ["taller than the burj"], "label": "REFUTES",
             "note": "829.8 m vs 443.2 m."},
        ],
    },
    {
        "id": "bosporus",
        "text": (
            "Bosporus is located between Africa and Europe. The Bosporus "
            "connects the Black Sea to the Sea of Marmara."
        ),
        "gold": [
            {"match": ["africa and europe"], "label": "REFUTES",
             "note": "The founding case: extraction must PRESERVE the falsehood."},
            {"match": ["black sea"], "label": "SUPPORTS"},
        ],
    },
    {
        "id": "opinion",
        "text": "It is very beautiful and quite pleasant.",
        "gold": [],
        "note": "Pure opinion — the check-worthiness gate should yield no claims.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
def load_gold(path: str | None) -> list[dict]:
    """
    The built-in development cases, or an external gold file.

    External gold exists so FEVER can be scored by the same code. Its records
    use the identical shape — id, text, gold[{match, label}] — so nothing in
    scoring needs to know where a case came from. See
    benchmarks/fever_sample.py, which writes that shape.

    The built-in GOLD is NOT a test set. Every one of its cases was used to
    debug this pipeline, so its numbers are overfit by construction and belong
    in a report only as a regression check. Unseen numbers come from --gold.
    """
    if path is None:
        return GOLD
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"  Loaded {len(cases)} case(s) from {path}")
    return cases


def run_cases(case_ids: list[str] | None, gold: list[dict]) -> dict:
    cases = [c for c in gold if case_ids is None or c["id"] in case_ids]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": {}}

    for case in cases:
        print(f"\n{'=' * 72}\n  RUN  {case['id']}\n{'=' * 72}")
        t0 = time.perf_counter()
        try:
            results = claim_extractor.run(case["text"])
            error = None
        except Exception as e:
            results, error = [], f"{type(e).__name__}: {e}"
            print(f"  FAILED: {error}")
        elapsed = time.perf_counter() - t0

        out["cases"][case["id"]] = {
            "elapsed_s": round(elapsed, 2),
            "error": error,
            "claims": [
                {
                    "claim": r.get("claim", ""),
                    "label": r.get("label", ""),
                    "nei_kind": r.get("nei_kind"),
                    "program_kind": r.get("program_kind"),
                    "source_url": r.get("source_url", ""),
                    "timings": r.get("timings", {}),
                }
                for r in results
            ],
        }
        print(f"  {len(results)} claim(s) in {elapsed:.1f}s")

    # Merge rather than overwrite, so `run --case amazon` does not discard the
    # other five cases someone spent an hour producing.
    if RESULTS_FILE.exists():
        try:
            previous = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            merged = previous.get("cases", {})
            merged.update(out["cases"])
            out["cases"] = merged
        except Exception:
            pass

    RESULTS_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Written to {RESULTS_FILE.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Score
# ─────────────────────────────────────────────────────────────────────────────
def _covers(claim: str, spec: dict) -> bool:
    low = claim.lower()
    return all(token in low for token in spec["match"])


def confusion(pairs: list[tuple]) -> dict:
    """
    3x3 counts, rows = gold, columns = predicted.

    EVERYTHING ELSE IN THIS FILE IS DERIVED FROM THIS TABLE. Per-class
    precision/recall/F1, macro-F1, accuracy, answered accuracy and abstention
    rate are all different readings of the same nine numbers — which is why the
    "should NEI be a class or an abstention?" question never needed deciding.
    Build the matrix once and read it both ways.
    """
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for gold, predicted, _ in pairs:
        if gold in m and predicted in m[gold]:
            m[gold][predicted] += 1
    return m


def per_class(m: dict) -> dict:
    """
    One-vs-rest precision, recall and F1 for each label.

    For class C: TP is gold C predicted C; FP is anything else predicted C; FN
    is gold C predicted anything else. TN is omitted because it is never used —
    every metric below is built from the three counts that involve C.

    WHY PER-CLASS AND NOT JUST ACCURACY. The three labels have wildly different
    costs. A false REFUTES tells a user their true sentence is false AND hands
    them a citation for it, which is the most damaging output this system can
    produce. A false NOT ENOUGH INFO is merely unhelpful. An overall accuracy
    figure averages a safety property together with a convenience one and
    reports the mean, which is worse than reporting neither.

    So REFUTES precision is the safety number, REFUTES recall is the usefulness
    number, and they belong on their own rows where somebody can see them.
    """
    out = {}
    for c in LABELS:
        tp = m[c][c]
        fp = sum(m[g][c] for g in LABELS if g != c)
        fn = sum(m[c][p] for p in LABELS if p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {"tp": tp, "fp": fp, "fn": fn,
                  "precision": precision, "recall": recall, "f1": f1,
                  "support": tp + fn}
    return out


def score(data: dict, gold: list[dict]) -> dict:
    per_case = []
    matched_pairs = []          # (gold_label, system_label, nei_kind)
    total_gold = total_covered = 0
    total_extracted = total_expected_extraction = 0
    nei_kinds: dict[str, int] = {}
    latencies: list[float] = []

    for case in gold:
        observed = data["cases"].get(case["id"])
        if observed is None:
            continue

        claims = observed["claims"]
        latencies.append(observed["elapsed_s"])
        total_extracted += len(claims)

        used: set[int] = set()
        rows = []
        for spec in case["gold"]:
            total_gold += 1
            hit = next(
                (i for i, c in enumerate(claims)
                 if i not in used and _covers(c["claim"], spec)),
                None,
            )
            if hit is None:
                rows.append({"gold": spec, "system": None})
                continue
            used.add(hit)
            total_covered += 1
            got = claims[hit]
            rows.append({"gold": spec, "system": got})
            matched_pairs.append((spec["label"], got["label"], got.get("nei_kind")))
            total_expected_extraction += 1

        for c in claims:
            if c.get("nei_kind"):
                nei_kinds[c["nei_kind"]] = nei_kinds.get(c["nei_kind"], 0) + 1

        per_case.append({
            "id": case["id"],
            "rows": rows,
            "extracted": len(claims),
            "unmatched_claims": [c for i, c in enumerate(claims) if i not in used],
            "elapsed_s": observed["elapsed_s"],
            "error": observed["error"],
        })

    # ── Selective prediction ──────────────────────────────────────────────────
    decidable = [(g, s) for g, s, _ in matched_pairs if g in DECIDABLE]
    answered = [(g, s) for g, s in decidable if s in DECIDABLE]
    correct_answered = sum(1 for g, s in answered if g == s)

    undecidable = [(g, s) for g, s, _ in matched_pairs if g not in DECIDABLE]
    correct_abstentions = sum(1 for g, s in undecidable if s not in DECIDABLE)

    matrix = confusion(matched_pairs)
    classes = per_class(matrix)
    graded = sum(matrix[g][p] for g in LABELS for p in LABELS)
    correct = sum(matrix[c][c] for c in LABELS)

    return {
        "per_case": per_case,
        "matrix": matrix,
        "classes": classes,
        # Macro, not micro. A micro average is dominated by whichever label
        # happens to be most common in the gold set, so it would mostly report
        # how the test data was assembled. Macro weights all three equally,
        # which is what FEVER and AFEV report and what makes the number
        # comparable to theirs.
        "macro_f1": statistics.mean(classes[c]["f1"] for c in LABELS),
        "accuracy": correct / graded if graded else 0.0,
        "graded": graded,
        # Extraction, in the same TP/FP/FN terms. 'Coverage' is extraction
        # RECALL — the standard name, and the one that makes clear a missed
        # claim is a false negative rather than a mystery.
        "extraction_tp": total_covered,
        "extraction_fn": total_gold - total_covered,
        "extraction_fp": total_extracted - total_expected_extraction,
        "coverage": total_covered / total_gold if total_gold else 0.0,
        "precision": total_expected_extraction / total_extracted if total_extracted else 0.0,
        "total_gold": total_gold,
        "total_covered": total_covered,
        "total_extracted": total_extracted,
        "decidable": len(decidable),
        "answered": len(answered),
        "correct_answered": correct_answered,
        "answered_accuracy": correct_answered / len(answered) if answered else 0.0,
        "abstention_rate": 1 - len(answered) / len(decidable) if decidable else 0.0,
        "undecidable": len(undecidable),
        "correct_abstentions": correct_abstentions,
        "nei_kinds": nei_kinds,
        "median_latency_s": statistics.median(latencies) if latencies else 0.0,
        "total_latency_s": sum(latencies),
    }


def render(m: dict) -> str:
    L = []
    a = L.append

    a("# AutoCitation evaluation\n")
    a("## Extraction\n")
    a(f"| | count |")
    a(f"|---|---|")
    a(f"| TP — gold proposition extracted | {m['extraction_tp']} |")
    a(f"| FN — gold proposition missed | {m['extraction_fn']} |")
    a(f"| FP — claim matching no gold proposition | {m['extraction_fp']} |")
    a("")
    a(f"- **Recall (coverage)** {m['coverage']:.0%}")
    a(f"- **Precision** {m['precision']:.0%}\n")
    a("A claim that is never extracted can never be labelled, so recall bounds")
    a("every number below it. Precision catches fabrication — an invented claim")
    a("matches no gold proposition and lands in FP.\n")

    a("## Confusion matrix\n")
    a("Rows are gold, columns are predicted.\n")
    header = "| gold \\ predicted | " + " | ".join(LABELS) + " |"
    a(header)
    a("|---" * (len(LABELS) + 1) + "|")
    for g in LABELS:
        cells = " | ".join(str(m["matrix"][g][p]) for p in LABELS)
        a(f"| **{g}** | {cells} |")
    a("")

    a("## Per class\n")
    a("| label | TP | FP | FN | precision | recall | F1 | support |")
    a("|---|---|---|---|---|---|---|---|")
    for c in LABELS:
        k = m["classes"][c]
        a(f"| {c} | {k['tp']} | {k['fp']} | {k['fn']} | "
          f"{k['precision']:.2f} | {k['recall']:.2f} | {k['f1']:.2f} | {k['support']} |")
    a("")
    a(f"- **Macro-F1** {m['macro_f1']:.2f}   (comparable to FEVER / AFEV reporting)")
    a(f"- **Accuracy** {m['accuracy']:.0%}  ({m['graded']} graded claims)\n")
    a("Read the REFUTES row first. Its **precision** is the safety number — a")
    a("false REFUTES tells a user their true sentence is false and attaches a")
    a("citation. Its **recall** is the usefulness number — the errors caught.")
    a("A low SUPPORTS recall is the benign failure: unhelpful, not harmful.\n")

    a("## Verdicts — selective prediction\n")
    a(f"- **Answered accuracy** {m['answered_accuracy']:.0%} "
      f"({m['correct_answered']}/{m['answered']} of the decidable claims it chose to answer)")
    a(f"- **Abstention rate** {m['abstention_rate']:.0%} "
      f"({m['decidable'] - m['answered']}/{m['decidable']} decidable claims declined)")
    a(f"- **Correct abstentions** {m['correct_abstentions']}/{m['undecidable']} "
      f"(claims where the evidence genuinely cannot settle the question)\n")
    a("Read the first two together. High accuracy with high abstention is a")
    a("cautious system; high accuracy with low abstention is a good one; low")
    a("accuracy with low abstention is worse than no system, because a wrong")
    a("verdict arrives with a citation attached.\n")
    a("These are the same nine numbers as the matrix above, read differently:")
    a("answered accuracy is the S/R diagonal over the S/R quadrant, and the")
    a("abstention rate is the NOT ENOUGH INFO column over the S/R rows.\n")

    if m["nei_kinds"]:
        a("## Why it abstained\n")
        for kind, n in sorted(m["nei_kinds"].items(), key=lambda kv: -kv[1]):
            a(f"- `{kind}` {n}")
        a("")
        a("`RETRIEVAL_FAILURE` is recoverable — the evidence was not about the")
        a("claim. `GENUINE` is terminal. `EXTRACTION_FAILURE` means no claim was")
        a("produced at all and nothing was checked.\n")

    a("## Latency\n")
    a(f"- Median per document **{m['median_latency_s']:.0f}s**")
    a(f"- Total {m['total_latency_s']:.0f}s\n")

    a("## Per case\n")
    for case in m["per_case"]:
        flag = f"  ERROR: {case['error']}" if case["error"] else ""
        a(f"### {case['id']}  ({case['elapsed_s']:.0f}s, "
          f"{case['extracted']} claims){flag}\n")
        for row in case["rows"]:
            gold, sysv = row["gold"], row["system"]
            if sysv is None:
                a(f"- MISSING  expected `{gold['label']}`  "
                  f"match={gold['match']}")
                continue
            ok = "OK  " if gold["label"] == sysv["label"] else "WRONG"
            extra = f" [{sysv['program_kind']}]" if sysv.get("program_kind") else ""
            a(f"- {ok}  gold `{gold['label']}` / got `{sysv['label']}`{extra}  "
              f"— {sysv['claim'][:70]}")
        for c in case["unmatched_claims"]:
            a(f"- EXTRA    `{c['label']}`  — {c['claim'][:70]}")
        a("")

    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --gold on every subcommand, because run and score must agree on which
    # cases exist. Scoring a FEVER run against the built-in development cases
    # would silently report zero of everything.
    for name, helptext in [("run", "run the pipeline and cache raw results"),
                           ("score", "score the cached results"),
                           ("list", "list gold cases")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--gold", help="external gold file "
                                      "(e.g. fever_gold.json); omit for the "
                                      "built-in development cases")
        if name == "run":
            p.add_argument("--case", action="append", help="case id (repeatable)")
        if name == "score":
            p.add_argument("--markdown", help="also write the report to this path")

    args = ap.parse_args()
    gold = load_gold(args.gold)

    if args.cmd == "list":
        for c in gold:
            print(f"  {c['id']:20} {len(c['gold'])} gold proposition(s)  "
                  f"{c['text'][:50]}")
        return 0

    if args.cmd == "run":
        run_cases(args.case, gold)
        suffix = f" --gold {args.gold}" if args.gold else ""
        print(f"  Now run:  python -m evaluation score{suffix}")
        return 0

    if not RESULTS_FILE.exists():
        print(f"  {RESULTS_FILE.name} not found — run `python -m evaluation run` first.")
        return 1

    data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    report = render(score(data, gold))
    print("\n" + report)

    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        print(f"\n  Written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
