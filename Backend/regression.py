"""
regression.py — fixed inputs with known-correct outcomes.

WHY: every fix in this pipeline has been validated by reading one console log
and deciding it looked right. That does not survive the next change. A
regression suite pins the behaviours already paid for, so a later prompt edit
that quietly reintroduces "corrects the claim instead of extracting it" fails
loudly instead of passing unnoticed.

Run from the Backend/ directory with Ollama up:

    python regression.py                  # current settings
    python regression.py --sweep-thinking # compare think False / "low" / True

The sweep answers a specific open question: qwen3 emits reasoning tokens that
verifier.py discards, so disabling them is nearly free speed — UNLESS the
reasoning is what produces the correct verdict. Compare LABELS first and
wall-clock second; a configuration that is twice as fast and wrong is not an
improvement.
"""

import argparse
import json
import time
from pathlib import Path

import Pipeline.claim_extractor as claim_extractor
import Pipeline.verifier as verifier

BASELINE = Path(__file__).with_name("regression_baseline.json")


# ─────────────────────────────────────────────────────────────────────────────
# Cases — each is a behaviour that was broken at some point and then fixed.
# `expect` maps a distinctive substring of the expected claim to its label.
# ─────────────────────────────────────────────────────────────────────────────
CASES = [
    {
        "id": "faithful_false_claim",
        "text": "Bosporus is located between Africa and Europe.",
        "expect": {"Africa and Europe": "REFUTES"},
        "guards": "Extraction must PRESERVE the falsehood. phi3 originally "
                  "'fixed' it to Asia Minor and Thrace, after which "
                  "verification could only ever return SUPPORTS.",
    },
    {
        "id": "cross_sentence_coreference",
        "text": "Bosporus is located between Africa and Europe. "
                "You can walk between them with Yavuz Sultan Selim Bridge.",
        "expect": {"Africa and Europe": None},   # both sentences must yield a claim
        "min_claims": 2,
        "guards": "Sentence 2 must be reached at all — a stalled sentence 1 "
                  "used to consume the whole iteration budget — and 'them' "
                  "must resolve from sentence 1.",
    },
    {
        "id": "true_claim_supported",
        "text": "The Bosporus connects the Black Sea to the Sea of Marmara.",
        "expect": {"Black Sea": "SUPPORTS"},
        "guards": "The gates must not be so strict that TRUE claims fail. A "
                  "fact-checker that refutes everything is as useless as one "
                  "that supports everything.",
    },
    {
        "id": "unverifiable_opinion",
        "text": "It is very beautiful and quite pleasant.",
        "expect": {},
        "forbid_verification": True,
        "guards": "Pure opinion with no named entity must be caught by the "
                  "check-worthiness gate, not discovered 152s later as NEI.",
    },
]


def run_case(case: dict) -> dict:
    t0 = time.perf_counter()
    results = claim_extractor.run(case["text"])
    elapsed = time.perf_counter() - t0

    failures = []

    for needle, want_label in case.get("expect", {}).items():
        hit = next((r for r in results if needle.lower() in r["claim"].lower()), None)
        if hit is None:
            failures.append(f"no claim containing {needle!r}")
        elif want_label and hit["label"] != want_label:
            failures.append(f"{needle!r}: expected {want_label}, got {hit['label']}")

    if len(results) < case.get("min_claims", 0):
        failures.append(f"expected >= {case['min_claims']} claims, got {len(results)}")

    if case.get("forbid_verification"):
        verified = [r for r in results if r.get("nei_kind") != "UNVERIFIABLE"]
        if verified:
            failures.append(
                f"{len(verified)} claim(s) reached the verifier that should "
                f"have been filtered as unverifiable"
            )

    return {
        "id": case["id"],
        "passed": not failures,
        "failures": failures,
        "elapsed_s": round(elapsed, 1),
        "claims": [
            {"claim": r["claim"], "label": r["label"], "nei_kind": r.get("nei_kind")}
            for r in results
        ],
    }


def run_suite(tag: str) -> list[dict]:
    print(f"\n{'=' * 72}\n  {tag}\n{'=' * 72}")
    out = []
    for case in CASES:
        r = run_case(case)
        out.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{mark}] {r['id']}  ({r['elapsed_s']}s)")
        for c in r["claims"]:
            print(f"        [{c['label']}] {c['claim']}")
        for f in r["failures"]:
            print(f"        !! {f}")
    return out


def summarise(tag: str, results: list[dict]) -> None:
    passed = sum(r["passed"] for r in results)
    total_s = sum(r["elapsed_s"] for r in results)
    print(f"\n  {tag}: {passed}/{len(results)} passed, {total_s:.1f}s total")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-thinking", action="store_true",
                    help="run the suite once per think setting and compare")
    args = ap.parse_args()

    if not args.sweep_thinking:
        results = run_suite("current settings")
        summarise("current", results)
        BASELINE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  Baseline written to {BASELINE.name}")
        return

    table = {}
    for setting in (False, "low", True):
        # num_predict is derived from the thinking setting at import time, so
        # both must be set together or a disabled-thinking run keeps a 1024
        # token budget it cannot use.
        verifier.VERIFIER_THINKING = setting
        verifier.VERIFIER_NUM_PREDICT = 256 if setting is False else 1024

        results = run_suite(f"think={setting!r}")
        summarise(f"think={setting!r}", results)
        table[repr(setting)] = results

    print(f"\n\n{'=' * 72}\n  COMPARISON — labels first, seconds second\n{'=' * 72}")
    print(f"{'case':28} " + " ".join(f"{k:>18}" for k in table))
    print("-" * 88)
    for i, case in enumerate(CASES):
        cells = []
        for results in table.values():
            r = results[i]
            labels = ",".join(c["label"][:4] for c in r["claims"]) or "-"
            cells.append(f"{labels}/{r['elapsed_s']:.0f}s")
        print(f"{case['id']:28} " + " ".join(f"{c:>18}" for c in cells))

    print("\n  Read it this way: if the label columns are identical, take the "
          "fastest setting. If they differ, the reasoning tokens are doing "
          "real work and the speed is not free.")


if __name__ == "__main__":
    main()
