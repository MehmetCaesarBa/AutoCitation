"""
Single-call verifier probe.

WHY THIS EXISTS. Prompt edits to verifier.py have now produced three different
verdicts on one unchanged case, and each attempt cost a full pipeline run — 400
to 900 seconds of extraction, Wikipedia retrieval and clause handling — to test
a change that affects exactly one inference call. One experiment per sitting is
how guesses replace measurements.

This runs that one call. Claim in, evidence in, verdict out, in the time qwen3
takes to think. No spaCy, no network, no extraction. Decoding is greedy with a
fixed seed, so a repeat run on an unchanged prompt is bit-identical: any
difference you see was caused by the thing you changed.

    python verifier_probe.py --list
    python verifier_probe.py --all
    python verifier_probe.py --case python_gc --show-think
    python verifier_probe.py --all --no-supports-clause
    python verifier_probe.py --all --sweep-thinking

WHAT IT DOES NOT TEST. Retrieval. Every case hands the verifier a chunk chosen
by hand, which is the point: it separates "the verifier misread good evidence"
from "retrieval never delivered the evidence". Those look identical in a
pipeline log and need opposite fixes. A passing probe is not a passing pipeline.
"""

import argparse
import re
import sys
import time

import requests

import Pipeline.verifier as verifier

# Read defensively. An earlier version of this file called a helper that exists
# in some versions of ollama_client and not others; every probe run died on
# AttributeError and the blanket handler made it look like normal behaviour.
# Nothing here should break because a constant was renamed in a module this
# file only borrows from.
OLLAMA_URL = getattr(verifier, "OLLAMA_URL", "http://localhost:11434/api/generate")
KEEP_ALIVE = getattr(verifier, "KEEP_ALIVE", "30m")


# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────
# These originally held claim + evidence copied verbatim from real pipeline
# runs (python_gc, amazon_disputed, jamestown_nationality/augustine/date — see
# verifier.py's build_verification_prompt docstring, which still narrates the
# reasoning traces recorded under those names). They have been REPLACED with
# constructed analogs: same logical shape, different subject, evidence written
# from scratch rather than pulled from a real run.
#
# WHY REPLACE CASES THAT WERE WORKING. Every rule in the prompt was iterated
# against these five exact sentences — rule 6's SUPPORTS clause was kept
# because it moved jamestown_nationality specifically from SUPPORTS to NOT
# ENOUGH INFO. A prompt change validated only against the wording it was
# designed to fix proves it matched that sentence, not that the model learned
# the general rule. Re-running the same five sentences after every future
# prompt edit would let that gap grow invisibly — rules accumulating that
# happen to work on the probe's own text and nothing else, which is
# underfitting wearing the costume of a passing test. Renaming the cases and
# rewriting the evidence is what makes a pass mean the fix generalized rather
# than that it was tuned against exactly this input.
#
# `expect` is what the EVIDENCE justifies, not what is true in the world.
#
# THE CONTROLS ARE NOT OPTIONAL. A prompt that fixes over-refutation by
# answering NOT ENOUGH INFO to everything scores perfectly on the problem cases
# and has destroyed the system. manila_macau proves it can still refute;
# manila_date and git_immutable_commits prove it can still support. Read all
# five.
CASES: dict[str, dict] = {
    "git_immutable_commits": {
        "why": "SAME LOGIC AS THE ORIGINAL python_gc. The claim adds a "
               "qualifier ('immutable') that the evidence conveys in different "
               "words ('does not change') rather than verbatim — tests whether "
               "rule 6's SUPPORTS clause collaterally suppresses rule 4's "
               "legitimate synonymy permission.",
        "claim": "Git tracks a project's history using immutable, "
                 "content-addressed commits.",
        "evidence": [
            "In Git, every commit is identified by a SHA-1 hash generated from "
            "its contents. Once a commit is written, its hash and its content "
            "do not change; a later edit is recorded as a new commit rather "
            "than a modification of an existing one.",
        ],
        "expect": "SUPPORTS",
    },

    "university_age_disputed": {
        "why": "SAME LOGIC AS THE ORIGINAL amazon_disputed. The claim is a "
               "plain assertion; the evidence reports the same title as "
               "actively contested between two candidates. Tests whether the "
               "model refutes a claim for merely failing to be undisputed.",
        "claim": "The title of the world's oldest continuously operating "
                 "university belongs to the University of Bologna.",
        "evidence": [
            "The University of Bologna, founded in 1088, is often cited as the "
            "oldest continuously operating university in the world, though the "
            "title is disputed by proponents of the University of "
            "Al-Qarawiyyin in Fez, Morocco, founded in 859 as a madrasa and "
            "regarded by some as the oldest degree-granting institution.",
        ],
        "expect": "NOT ENOUGH INFO",
    },

    "manila_nationality": {
        "why": "SAME LOGIC AS THE ORIGINAL jamestown_nationality. Evidence "
               "says SPANISH, claim says EUROPEAN; first-among-a-subset does "
               "not establish first-among-the-superset.",
        "claim": "Manila is celebrated as the earliest permanent European "
                 "settlement in East Asia.",
        "evidence": [
            "Manila was the first permanent Spanish settlement in East Asia, "
            "established in 1571 after the arrival of Spanish colonizers "
            "under Miguel Lopez de Legazpi.",
        ],
        "expect": "NOT ENOUGH INFO",
    },

    "manila_macau": {
        "why": "CONTROL — can it still refute at all? Same claim as "
               "manila_nationality; different evidence names a real, earlier "
               "rival that closes the question the first evidence left open.",
        "claim": "Manila is celebrated as the earliest permanent European "
                 "settlement in East Asia.",
        "evidence": [
            "Portuguese traders had already established a permanent "
            "settlement at Macau, on the southern coast of China, by 1557 — "
            "more than a decade before the Spanish founded Manila.",
        ],
        "expect": "REFUTES",
    },

    "manila_date": {
        "why": "CONTROL — a narrower statement that DOES entail the broader "
               "claim. A specific date establishes the month.",
        "claim": "The Spanish settlement of Manila was founded in June 1571.",
        "evidence": [
            "Manila was formally founded as a Spanish city by Miguel Lopez de "
            "Legazpi on June 24, 1571, and became the capital of the Spanish "
            "East Indies.",
        ],
        "expect": "SUPPORTS",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt surgery — A/B a rule without editing verifier.py
# ─────────────────────────────────────────────────────────────────────────────
_SUPPORTS_MARKER = "THE SAME BAR APPLIES TO SUPPORTS"
_AFTER_RULES = "Output strictly in this format"


def strip_supports_clause(prompt: str) -> tuple[str, bool]:
    """
    Remove rule 6's SUPPORTS half, returning (prompt, whether it was removed).

    THE POINT IS TO ISOLATE ONE VARIABLE WITHOUT A FILE EDIT. The suspicion is
    that this clause — added to stop the verifier over-SUPPORTING a
    subset/superset superlative — also made it refuse to support a claim whose
    evidence merely paraphrases it ('automatic garbage collection' against
    'garbage collection'). Editing verifier.py to find out means the repository
    is in a half-tested state while the experiment runs, and means remembering
    to put it back.

    Returns the flag rather than failing silently, because a marker that stops
    matching after a prompt rewrite would otherwise report "no difference" for
    an ablation that never happened.
    """
    if _SUPPORTS_MARKER not in prompt or _AFTER_RULES not in prompt:
        return prompt, False
    head = prompt.split(_SUPPORTS_MARKER)[0].rstrip()
    tail = _AFTER_RULES + prompt.split(_AFTER_RULES, 1)[1]
    return head + "\n\n" + tail, True


# ─────────────────────────────────────────────────────────────────────────────
def call_raw(prompt: str) -> tuple[str, str, float]:
    """
    Post the prompt; return (answer, reasoning, seconds).

    Options are imported from verifier so this cannot drift from what the
    pipeline sends. The one deliberate difference: the reasoning is kept.
    """
    payload = {
        "model": verifier.REASONING_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": verifier.VERIFIER_THINKING,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": verifier.VERIFIER_NUM_CTX,
            "num_predict": verifier.VERIFIER_NUM_PREDICT,
            "temperature": 0, "top_p": 1, "top_k": 1,
            "repeat_penalty": 1.0, "seed": 0,
        },
    }
    t0 = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    elapsed = time.perf_counter() - t0
    body = response.json()

    tokens = body.get("eval_count", 0)
    print(f"    [{verifier.REASONING_MODEL}]  {tokens} tok in {elapsed:.1f}s "
          f"({tokens / elapsed if elapsed else 0:.1f} tok/s)")

    # REASONING ARRIVES IN ITS OWN FIELD when `think` is set — body["thinking"],
    # not inline in body["response"]. A version of this probe searched the
    # answer for a <think> block, found none, and printed "(no reasoning)" on a
    # run that had just spent 2063 tokens reasoning.
    return body.get("response", ""), body.get("thinking", "") or "", elapsed


def set_thinking(value) -> None:
    """
    Override the reasoning budget.

    Both constants move together. VERIFIER_NUM_PREDICT is derived from
    VERIFIER_THINKING at import, so changing one alone leaves a 256-token
    ceiling in front of a model just told to reason — which truncates
    mid-<think> and returns nothing.
    """
    verifier.VERIFIER_THINKING = value
    verifier.VERIFIER_NUM_PREDICT = 256 if value is False else 3072


def run_case(name: str, case: dict, args) -> bool:
    print("\n" + "=" * 78)
    print(f"CASE  {name}    thinking={verifier.VERIFIER_THINKING!r}"
          f"{'   [supports-clause REMOVED]' if args.no_supports_clause else ''}")
    print(f"WHY   {case['why']}")
    print("=" * 78)
    print(f"CLAIM   {case['claim']}")
    for i, chunk in enumerate(case["evidence"], 1):
        print(f"EV_{i}    {chunk[:140]}{'...' if len(chunk) > 140 else ''}")

    prompt = verifier.build_verification_prompt(case["claim"], case["evidence"])

    if args.no_supports_clause:
        prompt, removed = strip_supports_clause(prompt)
        if not removed:
            print("    WARNING: the SUPPORTS clause marker was not found — the "
                  "prompt is UNCHANGED and this is not an ablation.")

    if args.show_prompt:
        print("\n--- PROMPT ---\n" + prompt + "\n--- END PROMPT ---")

    raw, thinking, _ = call_raw(prompt)

    inline = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    reasoning = thinking.strip() or (inline.group(1).strip() if inline else "")
    if args.show_think:
        print("\n--- REASONING ---")
        print(reasoning or "(model returned no reasoning — is `think` set?)")
        print("--- END REASONING ---")
    elif reasoning:
        print(f"    (reasoning: {len(reasoning.split())} words — --show-think to read)")

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    result = verifier.parse_verification_response(cleaned, case["evidence"])
    ok = result.label == case["expect"]

    print(f"\n  EXPECTED  {case['expect']}")
    print(f"  GOT       {result.label}   {'PASS' if ok else '<<<< FAIL'}")
    print(f"  RATIONALE {result.rationale}")

    # A wrong verdict usually explains itself in the rationale. The Amazon bug
    # was found by noticing a word there that was not in the claim.
    if not ok:
        claim_words = set(re.findall(r"[a-z]+", case["claim"].lower()))
        said = re.findall(r"[a-z]+", result.rationale.lower())
        # Content words only. 'explicitly' was on this list and fired on every
        # careful rationale — it describes how the model read the evidence, not
        # what it read, so it carried no signal and drowned the ones that do.
        added = [w for w in ("unambiguously", "universally", "undisputed",
                             "definitively", "certain", "immutable",
                             "european", "spanish", "portuguese", "moroccan")
                 if w in said and w not in claim_words]
        if added:
            print(f"  NOTE      rationale leans on words not in the claim: {added}")

    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="Probe verifier.verify() with one call.")
    p.add_argument("--case", action="append", help="case name (repeatable)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--think", choices=["false", "low", "true"])
    p.add_argument("--sweep-thinking", action="store_true")
    p.add_argument("--no-supports-clause", action="store_true",
                   help="ablate rule 6's SUPPORTS half without editing verifier.py")
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--show-think", action="store_true")
    args = p.parse_args()

    if args.list:
        for name, case in CASES.items():
            print(f"{name:24} expect {case['expect']:16} {case['why']}")
        return 0

    if args.case:
        missing = [c for c in args.case if c not in CASES]
        if missing:
            print(f"unknown case(s): {missing}. --list to see them.")
            return 2
        cases = {c: CASES[c] for c in args.case}
    elif args.all:
        cases = CASES
    else:
        p.print_help()
        return 2

    if args.think:
        set_thinking({"false": False, "low": "low", "true": True}[args.think])

    budgets = [False, "low", True] if args.sweep_thinking else [verifier.VERIFIER_THINKING]

    results = []
    for budget in budgets:
        set_thinking(budget)
        for name, case in cases.items():
            results.append((name, budget, run_case(name, case, args)))

    print("\n" + "=" * 78)
    print("SUMMARY")
    for name, budget, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:24} thinking={budget!r}")
    passed = sum(1 for _, _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} passed")
    print("=" * 78)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
