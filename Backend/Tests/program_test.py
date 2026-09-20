"""
Unit tests for Pipeline/program.py.

No Ollama and no network. Synthesis is a pure function of the parse, and the
handlers that decide labels (_handle_match, _handle_compare) are pure functions
of their inputs — which is the point of synthesising programs deterministically
rather than generating them with a model.
"""

import pytest

from Pipeline import program as prog


# ═════════════════════════════════════════════════════════════════════════════
# Synthesis — which claims get a program at all
# ═════════════════════════════════════════════════════════════════════════════
def test_superlative_claim_gets_a_superlative_program():
    """
    The case the module exists for. Evidence about Jamestown cannot refute a
    claim about Jamestown being first; the refutation is a passage about
    somewhere else, which never ranks against a query built from the subject.
    """
    p = prog.synthesize(
        "Jamestown is the earliest European permanent settlement in what is now "
        "the United States."
    )
    assert p is not None and p.kind == "superlative", p
    assert p.steps[0].op == "Question"
    assert p.steps[1].op == "Match"
    # The retrieval must aim at the CATEGORY, not the subject — that is the
    # whole mechanism.
    assert "jamestown" not in p.steps[0].args[0].lower(), p.steps[0].args


def test_comparative_claim_gets_a_comparative_program():
    p = prog.synthesize("Mount Everest is taller than K2.")
    assert p is not None and p.kind == "comparative", p
    assert [s.op for s in p.steps] == ["Quantity", "Quantity", "Compare"]
    assert p.steps[0].args[1] == "height"


def test_ordinary_claim_gets_no_program():
    """
    Most claims carry no operator and the existing verifier handles them. A
    module that fired on everything would be a rewrite rather than an addition.
    """
    assert prog.synthesize(
        "The English settlement of Jamestown was founded in May 1607."
    ) is None


def test_superlative_wins_over_comparative():
    """
    'longest' is JJS and would also satisfy a loose comparative test. Running
    the comparative handler on it would hunt for a second entity the claim
    never names.
    """
    p = prog.synthesize("The Amazon is the longest river in the world.")
    assert p is not None and p.kind == "superlative", p


def test_comparative_without_than_is_not_a_comparison():
    """
    "the river is longer now" compares one thing to itself across time. There
    is no second entity, so there is nothing to retrieve.
    """
    assert prog.synthesize("The river is longer now.") is None


def test_ambiguous_comparative_is_declined():
    """
    'larger' names no measurable dimension — larger by area, population or
    volume is undecidable from the adjective. Declining beats confidently
    comparing two unrelated numbers.
    """
    assert prog.synthesize("Canada is larger than Brazil.") is None


# ═════════════════════════════════════════════════════════════════════════════
# Grounding — the guard against a fabricated verdict
# ═════════════════════════════════════════════════════════════════════════════
EVIDENCE = [
    "The Spanish were the first Europeans to establish a permanent settlement "
    "in what became the United States, at Saint Augustine, Florida (1565)."
]


def test_answer_present_in_evidence_is_accepted():
    assert prog._answer_is_grounded("Saint Augustine", EVIDENCE)


def test_answer_absent_from_evidence_is_rejected():
    """
    phi3:mini knows a great deal about early American settlements and would
    answer from memory. An ungrounded answer would then be compared against the
    claim's subject and could produce a confident REFUTES citing a passage that
    never said it.
    """
    assert not prog._answer_is_grounded("Roanoke Colony", EVIDENCE)


def test_empty_answer_is_rejected():
    assert not prog._answer_is_grounded("", EVIDENCE)


# ═════════════════════════════════════════════════════════════════════════════
# Match — superlative verdicts
# ═════════════════════════════════════════════════════════════════════════════
def test_match_refutes_when_evidence_names_someone_else():
    """The Jamestown verdict, reached by asking who actually was first."""
    label, rationale = prog._handle_match("Saint Augustine", "Jamestown")
    assert label == "REFUTES"
    assert "Saint Augustine" in rationale


def test_match_supports_on_substring_either_way():
    """
    The two strings name the same thing at different lengths: 'Jamestown'
    against 'the Jamestown settlement'.
    """
    assert prog._handle_match("the Jamestown settlement", "Jamestown")[0] == "SUPPORTS"
    assert prog._handle_match("Jamestown", "the Jamestown settlement")[0] == "SUPPORTS"


def test_match_returns_nei_without_an_answer():
    """No answer must never become a verdict."""
    assert prog._handle_match(None, "Jamestown")[0] == "NOT ENOUGH INFO"


# ── Tier 2: shared head noun, compared modifier sets ─────────────────────────
def test_match_supports_a_less_specific_name_for_the_same_entity():
    """
    THE OBSERVED NEAR-MISS. Asked for the largest land animal, the evidence
    said 'African elephant' while the claim said 'African bush elephant'.
    Neither string contains the other, so the old substring-only Match
    returned REFUTES on a true claim.

    Same head ('elephant'), and {african} is contained in {african, bush}:
    one name is a narrower label for the same animal.
    """
    for answer, subject in [
        ("African elephant", "African bush elephant"),
        ("African bush elephant", "African elephant"),
    ]:
        label, rationale = prog._handle_match(answer, subject)
        assert label == "SUPPORTS", (answer, subject, rationale)


def test_match_refutes_a_different_member_of_the_same_category():
    """
    The counter-case that stops tier 2 from being merely permissive. Same head
    ('elephant'), but {asian} and {african} are disjoint — two different
    animals, so the superlative claim is genuinely refuted.

    No similarity score separates this from the pair above: GloVe rates this
    0.830 and the same-entity pair 0.667-0.932, and edit distance inverts them
    outright. Set containment is what makes the distinction categorical.
    """
    for answer, subject in [
        ("Asian elephant", "African bush elephant"),
        ("Asian elephant", "African elephant"),
        ("Amazon River", "Nile River"),
    ]:
        label, _ = prog._handle_match(answer, subject)
        assert label == "REFUTES", (answer, subject)


def test_match_refutes_when_the_heads_differ():
    """Different kinds of thing entirely — tier 3, unchanged behaviour."""
    assert prog._handle_match("Blue whale", "African bush elephant")[0] == "REFUTES"


def test_containment_wins_over_the_head_rule():
    """
    Tier 1 must run first. 'Jamestown' and 'the Jamestown settlement' have
    DIFFERENT heads ('jamestown' vs 'settlement'), so the head rule alone
    would split them; containment catches them before tier 2 is reached.
    """
    assert prog._handle_match("Jamestown", "the Jamestown settlement")[0] == "SUPPORTS"
    assert prog._handle_match("Everest", "Mount Everest")[0] == "SUPPORTS"
    assert prog._handle_match("Nile", "Nile River")[0] == "SUPPORTS"


def test_head_and_modifiers_splits_a_name():
    assert prog._head_and_modifiers("African bush elephant") == (
        "elephant", {"african", "bush"}
    )
    assert prog._head_and_modifiers("Jamestown") == ("jamestown", set())


# ═════════════════════════════════════════════════════════════════════════════
# Compare — arithmetic in Python, not in the model
# ═════════════════════════════════════════════════════════════════════════════
def test_compare_supports_when_subject_is_greater():
    label, rationale = prog._handle_compare(
        8848.86, 8611.0, "Mount Everest", "K2", "height", "8,848.86 m", "8,611 metres"
    )
    assert label == "SUPPORTS"
    assert "8,848.86 m" in rationale and "8,611 metres" in rationale


def test_compare_refutes_when_subject_is_smaller():
    assert prog._handle_compare(
        8611.0, 8848.86, "K2", "Mount Everest", "height", "8,611 m", "8,848 m"
    )[0] == "REFUTES"


@pytest.mark.parametrize("a,b", [(None, 8611.0), (8848.0, None), (None, None)])
def test_compare_returns_nei_when_a_value_is_missing(a, b):
    """A missing quantity is not evidence of anything."""
    assert prog._handle_compare(a, b, "X", "Y", "height", "", "")[0] == "NOT ENOUGH INFO"


def test_units_are_normalised_before_comparison():
    """
    29,031 ft and 8,848 m are the same mountain. Comparing the raw numbers
    would make feet beat metres every time.
    """
    feet = 29031 * prog._UNIT_TO_BASE["ft"]
    metres = 8611 * prog._UNIT_TO_BASE["metres"]
    assert feet > metres
    assert prog._handle_compare(
        feet, metres, "Everest", "K2", "height", "29,031 ft", "8,611 m"
    )[0] == "SUPPORTS"


# ═════════════════════════════════════════════════════════════════════════════
# Kill switch
# ═════════════════════════════════════════════════════════════════════════════
def test_stats_logger_is_self_contained():
    """
    REGRESSION. The first release called ollama_client.log_inference_stats,
    which exists in some versions of this project and not others. Every program
    run raised AttributeError, the blanket handler swallowed it, and the feature
    was dead for a whole run with one quiet line of warning.

    A module whose purpose is to work when the verifier cannot should not depend
    on another module for a print statement.
    """
    prog._log_stats({"eval_count": 42, "eval_duration": 2_000_000_000}, "phi3:mini")
    prog._log_stats({}, "phi3:mini")          # missing fields must not raise


def test_every_cross_module_call_actually_exists():
    """
    The same defect caught structurally rather than by example.

    Walks program.py's syntax tree, collects every `module.attribute` reference
    into ner and retriever, and asserts each one resolves. A helper that gets
    renamed or reverted out of another module now fails here in milliseconds
    instead of during a 700-second pipeline run, where it surfaced as one quiet
    fallback line that read like normal behaviour.
    """
    import ast
    import pathlib

    from Pipeline import ner, retriever

    modules = {"ner": ner, "retriever": retriever}
    tree = ast.parse(pathlib.Path(prog.__file__).read_text(encoding="utf-8"))

    checked = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in modules):
            module = modules[node.value.id]
            assert hasattr(module, node.attr), (
                f"program.py calls {node.value.id}.{node.attr}, "
                f"which does not exist in {module.__name__}"
            )
            checked += 1

    assert checked > 0, "the walk found nothing — the test is not testing anything"


def test_program_mode_can_be_disabled(monkeypatch):
    """
    PROGRAM_MODE=False is the comparison baseline. "Programs beat single-shot on
    superlatives" should be reproducible, not asserted.
    """
    monkeypatch.setattr(prog, "PROGRAM_MODE", False)
    assert prog.try_verify("Jamestown is the earliest European settlement.") is None
