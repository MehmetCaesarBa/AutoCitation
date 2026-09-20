"""
Unit tests for Pipeline/verifier.py's response parser.

The verifier used to be asked to "copy the single most relevant evidence chunk
exactly", so every verdict cost ~120 tokens of verbatim Wikipedia prose —
generated one token at a time, in the stage that consumes ~70% of the pipeline's
runtime. It now writes a number and Python does the lookup. These tests cover
that contract and the ways a model can get it slightly wrong.
"""

import pytest

from Pipeline import verifier


CHUNKS = [
    "Chunk one is about annual rainfall in Africa.",
    "The Bosporus lies between Asia and Europe.",
    "Chunk three is about something else entirely.",
]

# Both functions now return a VerificationResult and are read by NAME.
#
# They previously returned bare 3-tuples in DIFFERENT orders — the parser gave
# (label, evidence, rationale), verify gave (label, rationale, evidence) — and
# the first draft of this file unpacked six of them wrongly. Every field is a
# string, so nothing raised; the assertions simply compared a rationale against
# a Wikipedia paragraph. Attribute access makes that mistake unrepresentable.


# ═════════════════════════════════════════════════════════════════════════════
# Label parsing
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("raw, expected", [
    ("LABEL: SUPPORTS\nEVIDENCE: 1\nRATIONALE: x.", "SUPPORTS"),
    ("LABEL: REFUTES\nEVIDENCE: 1\nRATIONALE: x.", "REFUTES"),
    ("LABEL: NOT ENOUGH INFO\nEVIDENCE: 1\nRATIONALE: x.", "NOT ENOUGH INFO"),
    ("label: supports\nEVIDENCE: 1\nRATIONALE: x.", "SUPPORTS"),
])
def test_labels_are_parsed(raw, expected):
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.label == expected


def test_missing_label_defaults_to_not_enough_info():
    """Never guess a verdict from a malformed response."""
    result = verifier.parse_verification_response("nonsense output", CHUNKS)
    assert result.label == "NOT ENOUGH INFO"


# ═════════════════════════════════════════════════════════════════════════════
# Evidence index → chunk text
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("evidence_field", [
    "2",
    "Evidence_2",
    "Evidence 2:",
    "chunk 2",
])
def test_index_forms_resolve_to_the_chunk(evidence_field):
    """
    qwen3 does not always write the bare number the prompt asks for. The first
    integer in the field is taken as the selection.
    """
    raw = f"LABEL: REFUTES\nEVIDENCE: {evidence_field}\nRATIONALE: x."
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.evidence == CHUNKS[1]


def test_out_of_range_index_falls_back_to_top_chunk():
    """A hallucinated index must not raise IndexError mid-request."""
    raw = "LABEL: REFUTES\nEVIDENCE: 9\nRATIONALE: x."
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.evidence == CHUNKS[0]


def test_full_chunk_text_is_still_accepted():
    """
    Backward compatibility: a model that ignores the instruction and pastes the
    chunk anyway should still yield a usable result rather than an empty one.
    """
    raw = f"LABEL: REFUTES\nEVIDENCE: {CHUNKS[1]}\nRATIONALE: x."
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.evidence == CHUNKS[1]


def test_missing_evidence_field_falls_back():
    raw = "LABEL: NOT ENOUGH INFO\nRATIONALE: nothing relevant."
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.evidence == CHUNKS[0]


def test_empty_chunk_list_does_not_raise():
    """
    The original fallback indexed evidence_chunks[0] unconditionally and raised
    IndexError whenever retrieval came back empty.
    """
    result = verifier.parse_verification_response(
        "LABEL: REFUTES\nEVIDENCE: 1\nRATIONALE: x.", []
    )
    assert result.evidence == ""
    assert result.label == "REFUTES"


# ═════════════════════════════════════════════════════════════════════════════
# Rationale
# ═════════════════════════════════════════════════════════════════════════════
def test_rationale_is_extracted():
    raw = "LABEL: REFUTES\nEVIDENCE: 2\nRATIONALE: The evidence names Asia, not Africa."
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.rationale == "The evidence names Asia, not Africa."


def test_fields_are_addressed_by_name():
    """
    The whole point of the dataclass: a field can only be reached by the name
    that describes it, so evidence and rationale can never trade places.
    """
    raw = "LABEL: SUPPORTS\nEVIDENCE: 2\nRATIONALE: short reason."
    result = verifier.parse_verification_response(raw, CHUNKS)
    assert result.label == "SUPPORTS"
    assert result.evidence == CHUNKS[1]
    assert result.rationale == "short reason."

    # Unpacking is deliberately unavailable: stale positional code must fail
    # loudly rather than silently swap two strings.
    with pytest.raises(TypeError):
        _a, _b, _c = result


# ═════════════════════════════════════════════════════════════════════════════
# Configuration invariants
# ═════════════════════════════════════════════════════════════════════════════
def test_num_predict_covers_the_thinking_budget():
    """
    VERIFIER_NUM_PREDICT is derived from VERIFIER_THINKING at import time. If
    thinking is enabled the ceiling must also cover the <think> block, or
    generation is cut off before the answer is ever written.
    """
    if verifier.VERIFIER_THINKING is False:
        assert verifier.VERIFIER_NUM_PREDICT >= 128
    else:
        # 1024 was the previous value and it was MEASURED to be too small: one
        # run hit exactly 1024 and returned nothing, and verifier_probe measured
        # 2063 tokens on a hard case. The floor is set above that measurement,
        # not above a guess.
        assert verifier.VERIFIER_NUM_PREDICT >= 2048


# ═════════════════════════════════════════════════════════════════════════════
# Truncation is not a verdict
# ═════════════════════════════════════════════════════════════════════════════
def test_truncated_generation_is_reported_as_a_failure(monkeypatch):
    """
    Generation that dies inside <think> leaves an empty string after stripping,
    and the parser's default is NOT ENOUGH INFO. Observed: 267 seconds, an empty
    raw response, a blank rationale, and a reported NEI that was indistinguish-
    able from the two genuine NEI verdicts in the same run.

    The label still has to be one of the three FEVER values — the frontend and
    the aggregator have no fourth — so the rationale is what must carry the
    truth. Assert on the rationale, because that is the only channel that can.
    """
    monkeypatch.setattr(verifier, "call_ollama", lambda _prompt: "")

    result = verifier.verify("Some claim.", ["Some evidence chunk."])

    assert "VERIFIER FAILURE" in result.rationale
    assert "unchecked" in result.rationale.lower()


def test_a_real_label_is_not_mistaken_for_truncation(monkeypatch):
    """The guard must not fire on a well-formed answer."""
    monkeypatch.setattr(
        verifier, "call_ollama",
        lambda _prompt: "LABEL: SUPPORTS\nEVIDENCE: 1\nRATIONALE: It says so.",
    )

    result = verifier.verify("Some claim.", ["Some evidence chunk."])

    assert result.label == "SUPPORTS"
    assert "VERIFIER FAILURE" not in result.rationale
