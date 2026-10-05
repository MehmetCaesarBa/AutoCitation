"""
results.py — the two result types every verification path produces.

    VerificationResult   a VERDICT: label, the chunk behind it, one-line reason.
                         Produced by verifier.verify and program.execute.

    FactResult           the per-fact RECORD the rest of the system consumes:
                         the verdict plus where it came from and what it cost.
                         Built only by make_fact_result, called only by
                         verification.verify_claim.

WHY A SEPARATE MODULE. program.py and verifier.py both produce verdicts, and
verification.py imports both of them. Putting these types in any of those three
would make one import another for no reason but a class definition — or, in
verification.py's case, create an import cycle.

WHY FactResult IS A TypedDict AND NOT A DATACLASS. It is serialised to JSON and
read by key in postprocessor, evaluation, regression and claim_extractor. A
TypedDict is a plain dict at runtime, so none of those change, while a type
checker now catches a misspelled or missing key. The hazard VerificationResult
guards against — positional unpacking swapping two strings — cannot occur with
keyed access, so the dataclass argument does not carry over.
"""

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class VerificationResult:
    """
    A verdict, the chunk that produced it, and the one-line justification.

    Deliberately NOT a tuple or NamedTuple. This module used to return three
    bare strings, and the two functions that did so disagreed about the order:
    parse_verification_response gave (label, evidence, rationale) while verify
    gave (label, rationale, evidence). Positional unpacking cannot catch that
    mistake — all three fields are strings, so a swap raises nothing and simply
    puts a whole Wikipedia paragraph where a one-sentence rationale belongs,
    surfacing much later as a puzzling frontend bug.

    Attribute access removes the ordering from the interface entirely:
    result.evidence cannot be confused with result.rationale at any call site.
    Making it a NamedTuple would have preserved unpacking, and with it the
    hazard, so unpacking is left deliberately unavailable — stale positional
    code fails loudly instead of quietly.

    frozen=True because a verdict is a record of what happened, not a mutable
    working value.
    """
    label: str
    evidence: str
    rationale: str


class Timings(TypedDict, total=False):
    """Per-stage seconds. extraction_s is added later by the extraction loop."""
    retrieval_s: float
    verification_s: float
    extraction_s: float


class FactResult(TypedDict):
    claim: str
    label: str
    rationale: str
    evidence: str
    # Only meaningful for single-shot NOT ENOUGH INFO; None otherwise.
    nei_kind: str | None
    ner_query: str
    source_url: str
    # 'superlative' / 'comparative' on the program path; None on single-shot.
    # Always present, so consumers no longer need .get() to read it.
    program_kind: str | None
    timings: Timings


def make_fact_result(claim: str,
                     verdict: VerificationResult,
                     *,
                     ner_query: str,
                     source_url: str,
                     timings: Timings,
                     nei_kind: str | None = None,
                     program_kind: str | None = None) -> FactResult:
    """The single constructor for FactResult, so both paths fill every key."""
    return FactResult(
        claim=claim,
        label=verdict.label,
        rationale=verdict.rationale,
        evidence=verdict.evidence,
        nei_kind=nei_kind,
        ner_query=ner_query,
        source_url=source_url,
        program_kind=program_kind,
        timings=timings,
    )
