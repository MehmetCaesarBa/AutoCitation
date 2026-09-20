"""
Unit tests for Pipeline/postprocessor.py's label normalization.

normalize_label is the last gate before the frontend, and the frontend maps the
label straight onto a CSS class. Anything it lets through unrecognized would
render as an unstyled element, so its contract is: always return one of exactly
three strings, no matter what arrives.
"""

import pytest

from Pipeline import postprocessor


@pytest.mark.parametrize("raw, expected", [
    ("SUPPORTS",          "SUPPORTS"),
    ("REFUTES",           "REFUTES"),
    ("NOT ENOUGH INFO",   "NOT ENOUGH INFO"),
])
def test_canonical_labels_pass_through(raw, expected):
    assert postprocessor.normalize_label(raw) == expected


@pytest.mark.parametrize("raw", ["supports", "Supports", "  SUPPORTS  ", "\tsupports\n"])
def test_casing_and_whitespace_are_normalized(raw):
    assert postprocessor.normalize_label(raw) == "SUPPORTS"


@pytest.mark.parametrize("raw", [
    "NOT ENOUGH",
    "NOT ENOUGH INFORMATION",
    "not enough info",
    "NOTENOUGH",
])
def test_truncated_not_enough_variants_are_recovered(raw):
    """
    A model that stops mid-label, or writes a longer synonym, must still land on
    the canonical value rather than falling through to the default by accident.
    """
    assert postprocessor.normalize_label(raw) == "NOT ENOUGH INFO"


@pytest.mark.parametrize("raw", ["INCONCLUSIVE", "MAYBE", "", "   ", "42"])
def test_unrecognized_labels_default_to_not_enough_info(raw):
    """
    Defaulting to NOT ENOUGH INFO is the safe direction: an unparseable verdict
    must never be reported as SUPPORTS or REFUTES, because both are assertions
    about the world that nothing in the response actually justified.
    """
    assert postprocessor.normalize_label(raw) == "NOT ENOUGH INFO"


def test_output_is_always_one_of_three_values():
    """The invariant the frontend's colour mapping depends on."""
    valid = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
    noisy = ["supports", "REFUTE", "nope", "", "NOT", "verified", "TRUE", "FALSE"]
    assert {postprocessor.normalize_label(r) for r in noisy} <= valid


def test_unverifiable_currently_degrades_to_not_enough_info():
    """
    Documents a deliberate limitation rather than asserting desired behaviour.

    The check-worthiness gate marks unverifiable claims via nei_kind, NOT via a
    fourth label, precisely because this function coerces anything outside the
    three canonical values. If UNVERIFIABLE is ever promoted to a first-class
    label, this test should fail — and that failure is the reminder that
    LABEL_COLORS and the frontend CSS need updating at the same time.
    """
    assert postprocessor.normalize_label("UNVERIFIABLE") == "NOT ENOUGH INFO"
