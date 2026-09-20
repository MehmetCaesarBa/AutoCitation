"""
Tests for main.py's response contract.

No Ollama, no network, no server. These validate the SHAPE of what /check
returns against the model it declares — which is the exact class of defect that
reached production here, and the one integration testing is worst at catching
because it only appears on a branch nobody exercises.
"""

import pytest
from pydantic import ValidationError

import main


# ═════════════════════════════════════════════════════════════════════════════
# The empty-result branch
# ═════════════════════════════════════════════════════════════════════════════
def test_empty_result_matches_the_response_model():
    """
    THE BUG. The zero-claims branch returned 'claims' and 'elapsed_s' while
    CheckResponse declares 'results' and 'timings', so FastAPI answered a
    perfectly ordinary "nothing checkable here" outcome with a 500:

        ResponseValidationError: 2 validation errors
          ('response', 'total_claims')  Field required
          ('response', 'results')       Field required

    It survived because response_model validation only runs on a response that
    is actually produced. The happy path builds its dict in
    postprocessor.aggregate() with the right names; this branch built its own
    with different ones, and the two exits from a single endpoint drifted apart
    unnoticed.

    Constructing the model directly is the cheapest possible guard: it needs no
    server, no client, and no request.
    """
    payload = {
        "total_claims": 0,
        "overall_label": "NO CLAIMS",
        "summary": {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0},
        "results": [],
        "timings": {"total_s": 0.0},
        "message": "No verifiable claims could be extracted from the input.",
    }
    response = main.CheckResponse(**payload)

    assert response.total_claims == 0
    assert response.results == []
    assert response.message


def test_old_empty_shape_is_rejected():
    """
    The shape that shipped. If this ever validates again, the model has been
    loosened and the guard above is worthless.
    """
    with pytest.raises(ValidationError):
        main.CheckResponse(
            claims=[],
            overall_label="NO CLAIMS",
            summary={"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0},
            message="...",
            elapsed_s=0.0,
        )


def test_message_is_optional():
    """A normal response carries no message and must not be forced to invent one."""
    response = main.CheckResponse(
        total_claims=1,
        overall_label="SUPPORTS",
        summary={"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 0},
        results=[],
    )
    assert response.message is None


# ═════════════════════════════════════════════════════════════════════════════
# The happy path — the other exit from the same endpoint
# ═════════════════════════════════════════════════════════════════════════════
def test_cpp_is_not_rejected():
    """
    THE OTHER BUG IN THE SAME REPORT. A character allowlist rejected

        "Both Python and C++ are widely used object-oriented programming
         languages..."

    on the '+' in C++. The request returned in 0.00s having read nothing, and
    the API answered "no verifiable claims could be extracted" — a description
    of text that was never examined.
    """
    from Pipeline import claim_extractor as ce

    text = (
        "Both Python and C++ are widely used object-oriented programming "
        "languages. Python uses automatic garbage collection for memory "
        "management, whereas standard C++ relies primarily on manual memory "
        "management and deterministic RAII rather than a default tracing "
        "garbage collector."
    )
    assert ce.validate_input(text) is None
    assert ce.precondition(text) is not None


@pytest.mark.parametrize("text", [
    "Water is 100% pure at that temperature.",          # percent
    "The ratio is 3/4 of the total.",                   # slash
    "He said: it opened in 1932.",                      # colon
    "The bridge — the longest of its kind — opened.",   # em dash
    "It's the world's largest lake.",                   # curly-safe apostrophes
    "Ağrı Dağı is the highest peak in Türkiye.",        # Turkish
    "H2O and CO2 are common molecules.",                # formulae
])
def test_ordinary_punctuation_survives(text):
    """
    Every one of these was rejected by the allowlist. Enumerating the characters
    that may appear in a true statement about the world is not a finite task,
    which is why the check is now a denylist of control characters.
    """
    from Pipeline import claim_extractor as ce
    assert ce.validate_input(text) is None, text


def test_control_characters_are_still_rejected():
    """The denylist still has to deny something."""
    from Pipeline import claim_extractor as ce
    assert ce.validate_input("Jamestown\x00was founded in 1607.") is not None


def test_empty_and_overlong_inputs_are_rejected_with_a_reason():
    from Pipeline import claim_extractor as ce
    assert "empty" in ce.validate_input("   ")
    assert "limit" in ce.validate_input("word " * (ce.MAX_INPUT_WORDS + 1))


def test_aggregate_output_matches_the_response_model():
    """
    Both exits must satisfy the same contract. This one feeds postprocessor's
    real aggregate() output into the model, so a change to either side that
    breaks the other fails here rather than in a 500 during a live request.
    """
    from Pipeline import postprocessor

    aggregated = postprocessor.process([{
        "claim": "Jamestown was founded in May 1607.",
        "label": "SUPPORTS",
        "rationale": "The evidence gives the founding date as May 14, 1607.",
        "evidence": "…founded on May 14, 1607.",
        "ner_query": "Jamestown",
        "source_url": "https://en.wikipedia.org/wiki/Jamestown",
        "timings": {"retrieval_s": 1.0, "verification_s": 2.0, "extraction_s": 3.0},
    }])
    aggregated.setdefault("timings", {})["total_s"] = 6.0

    response = main.CheckResponse(**aggregated)
    assert response.total_claims == 1
    assert response.overall_label == "SUPPORTS"
