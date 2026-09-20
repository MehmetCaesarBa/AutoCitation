"""
Unit tests for the pure functions in Pipeline/claim_extractor.py.

No Ollama, no network. Everything here is string in, string out — which is
precisely why these bugs were cheap to find once anyone looked.

    cd Backend
    pytest Tests/claim_extractor_test.py
"""

import pytest

from Pipeline import claim_extractor as ce


# ═════════════════════════════════════════════════════════════════════════════
# _normalize_claim — duplicate detection
# ═════════════════════════════════════════════════════════════════════════════
def test_article_variants_collapse():
    """
    A run verified "…with Yavuz Sultan Selim Bridge." and "…with THE Yavuz
    Sultan Selim Bridge." as two separate claims, at 118s and 130s, because the
    old normalizer compared raw lowercased strings.
    """
    a = ce._normalize_claim("You can walk between Africa and Europe with Yavuz Sultan Selim Bridge.")
    b = ce._normalize_claim("You can walk between Africa and Europe with the Yavuz Sultan Selim Bridge.")
    assert a == b


def test_leading_article_collapses():
    assert ce._normalize_claim("Bosporus is located in Turkey.") == \
           ce._normalize_claim("The Bosporus is located in Turkey.")


def test_opposite_claims_do_NOT_collapse():
    """
    The safety property of exact-match normalization. Claims differing by one
    content word can be direct opposites, and merging them would silently
    discard the disagreement the whole pipeline exists to surface.
    """
    assert ce._normalize_claim("Bosporus is between Africa and Europe.") != \
           ce._normalize_claim("Bosporus is between Asia and Europe.")


def test_punctuation_and_case_are_ignored():
    assert ce._normalize_claim("BOSPORUS, is  located!") == ce._normalize_claim("bosporus is located")


# ═════════════════════════════════════════════════════════════════════════════
# _content_words — the faithfulness gate's vocabulary
# ═════════════════════════════════════════════════════════════════════════════
def test_short_words_are_excluded():
    """Words of 3 characters or fewer are dropped, so 'the'/'and'/'is' never appear."""
    words = ce._content_words("The Bosporus separates Africa and Europe.")
    assert not any(len(w) <= 3 for w in words)
    assert "africa" in words and "europe" in words


def test_inflections_are_stemmed_together():
    """
    Rewriting a clause into a standalone sentence changes verb inflection.
    Without stemming, 'separated' would count as material absent from a source
    that says 'separates', and a faithful claim would be rejected.
    """
    assert ce._content_words("separates") == ce._content_words("separated")


def test_foreign_entity_is_detectable():
    """The Bosporus bug: the gate must see 'asia' as absent from the source."""
    source = ce._content_words("Bosporus is located between Africa and Europe.")
    claim = ce._content_words("Bosporus is located between Asia and Europe.")
    assert claim - source - ce.FUNCTION_WORDS == {"asia"}


def test_faithful_claim_leaves_no_residue():
    source = ce._content_words("Bosporus is located between Africa and Europe.")
    claim = ce._content_words("Bosporus is located between Africa and Europe.")
    assert not (claim - source - ce.FUNCTION_WORDS)


# ═════════════════════════════════════════════════════════════════════════════
# parse_fact_from_response — the strict output contract
# ═════════════════════════════════════════════════════════════════════════════
def test_plain_fact_line():
    assert ce.parse_fact_from_response("Fact_1: Bosporus is in Europe.", 1) == "Bosporus is in Europe."


def test_wrong_index_is_rejected():
    """
    Requiring the EXPECTED index stops the model re-emitting an earlier fact or
    inventing out-of-sequence ones.
    """
    assert ce.parse_fact_from_response("Fact_2: X.", 1) is None


def test_terminate_returns_the_sentinel():
    """
    TERMINATE must be distinguishable from None: one closes the loop, the other
    means 'malformed, try again'. Conflating them cost a whole iteration budget.
    """
    assert ce.parse_fact_from_response("Terminate", 1) == ce.TERMINATE


def test_fact_line_wins_over_terminate():
    """phi3 sometimes emits both; taking Terminate first threw the fact away."""
    assert ce.parse_fact_from_response("Fact_1: A real claim.\nTerminate", 1) == "A real claim."


def test_trailing_parenthetical_is_stripped():
    """Meta-commentary pollutes the NER query and the verifier prompt."""
    got = ce.parse_fact_from_response(
        "Fact_1: Marie was born in Warsaw. (extracted from the text)", 1
    )
    assert got == "Marie was born in Warsaw."


def test_separator_variants_are_tolerated():
    assert ce.parse_fact_from_response("Fact-1: dash separated.", 1) == "dash separated."


def test_garbage_is_rejected_with_no_fallback():
    """
    The loose original regex matched the word 'Fact' anywhere, which let leaked
    chain-of-thought be accepted as a claim.
    """
    assert ce.parse_fact_from_response("we should proceed with Fact extraction now", 1) is None


# ═════════════════════════════════════════════════════════════════════════════
# build_extraction_prompt
# ═════════════════════════════════════════════════════════════════════════════
DOC = "Bosporus is located between Africa and Europe. You can walk between them."
SENT2 = "You can walk between them."


def test_unscoped_prompt_shows_the_text_once():
    prompt = ce.build_extraction_prompt(DOC, [], [], [])
    assert "TARGET SENTENCE" not in prompt
    assert DOC in prompt


def test_scoped_prompt_separates_context_from_target():
    """
    Sentence scoping bounds the decomposition, but the document must stay
    visible or cross-sentence pronouns become unresolvable.
    """
    prompt = ce.build_extraction_prompt(DOC, [], [], [], None, target_sentence=SENT2)
    assert "CONTEXT ONLY" in prompt
    assert "TARGET SENTENCE" in prompt
    assert DOC in prompt and SENT2 in prompt


def test_prompt_changes_after_a_rejection():
    """
    THE reason rejection feedback exists. Greedy decoding is a deterministic
    function of the prompt, so an unchanged prompt reproduces the rejected
    output forever — one run burned iterations 2 through 10 on identical text.
    """
    clean = ce.build_extraction_prompt(DOC, [], [], [])
    retry = ce.build_extraction_prompt(
        DOC, [], [], [], [("A bad claim.", "faithfulness", "asia, strait")]
    )
    assert clean != retry
    assert "FAILED ATTEMPTS" in retry


@pytest.mark.parametrize("kind, detail, expected_phrase", [
    ("faithfulness", "asia, strait", "not in the document"),
    ("decontextualization", "'They' refers to something outside the claim", "refers to something outside"),
    ("fluency", "it has no main verb", "no main verb"),
    ("malformed", "no 'Fact_N:' line", "required output format"),
])
def test_each_rejection_kind_renders_its_own_wording(kind, detail, expected_phrase):
    """
    Every rejection once used the faithfulness template, so a claim rejected
    for a dangling pronoun was told "these words do not appear in the text:
    'its' refers to..." — false and self-contradictory. The model responded by
    repeating itself three times.
    """
    prompt = ce.build_extraction_prompt(DOC, [], [], [], [("attempt", kind, detail)])
    assert expected_phrase in prompt


def test_rationales_are_excluded_by_default():
    """
    INCLUDE_RATIONALES_IN_HISTORY is False because a REFUTES rationale sitting
    in the prompt led the extractor to propose the verifier's correction as its
    next 'fact'.
    """
    prompt = ce.build_extraction_prompt(
        DOC,
        ["Bosporus is located between Africa and Europe."],
        ["REFUTES"],
        ["Evidence says Asia and Europe."],
    )
    assert "Rationale_" not in prompt
    assert "Asia" not in prompt


def test_rationales_appear_when_the_flag_is_on(monkeypatch):
    """The flag exists so the paper's Eq. 2 configuration can be measured."""
    monkeypatch.setattr(ce, "INCLUDE_RATIONALES_IN_HISTORY", True)
    prompt = ce.build_extraction_prompt(
        DOC,
        ["Bosporus is located between Africa and Europe."],
        ["REFUTES"],
        ["Evidence says Asia and Europe."],
    )
    assert "Rationale_1:" in prompt


def test_mismatched_history_lengths_raise(monkeypatch):
    """
    strict=True turns a silent truncation into a loud error: plain zip() stops
    at the shortest input, so a missing rationale would quietly drop a verified
    fact out of the history the extractor reasons over.
    """
    monkeypatch.setattr(ce, "INCLUDE_RATIONALES_IN_HISTORY", True)
    with pytest.raises(ValueError):
        ce.build_extraction_prompt(DOC, ["a", "b"], ["SUPPORTS"], ["r"])


def test_next_fact_index_is_interpolated():
    """The instruction text renumbers itself; the prompt is never static."""
    prompt = ce.build_extraction_prompt(DOC, ["a", "b"], ["SUPPORTS", "REFUTES"], ["r1", "r2"])
    assert "Fact_3" in prompt


# ═════════════════════════════════════════════════════════════════════════════
# diagnose_nei — which token stands in for the subject
# ═════════════════════════════════════════════════════════════════════════════
def test_diagnosis_anchors_on_the_name_not_the_longest_word():
    """
    'English settlement of Jamestown' — longest token is 'settlement' (10), the
    name is 'Jamestown' (9). Anchoring on length searched the evidence for
    'settlement', found it in a passage that never mentions Jamestown, and
    reported GENUINE.

    The two diagnoses call for opposite responses. GENUINE is terminal: the
    right evidence was retrieved and is authentically silent, so no better query
    will help. RETRIEVAL_FAILURE means try again. Confusing them tells the
    operator to stop looking at the moment retrieval needs fixing.
    """
    chunks = [
        "Spain and Portugal established a permanent settlement in the New World "
        "long before other European powers attempted one."
    ]
    assert ce.diagnose_nei(
        "The English settlement of Jamestown paved the way for European presence.",
        chunks,
    ) == "RETRIEVAL_FAILURE"


def test_diagnosis_reports_genuine_when_the_name_is_present():
    """The other direction: the name IS in the evidence, so NEI is a verdict."""
    chunks = [
        "The Jamestown settlement in the Colony of Virginia was the first "
        "permanent English settlement in the Americas."
    ]
    assert ce.diagnose_nei(
        "The English settlement of Jamestown paved the way for European presence.",
        chunks,
    ) == "GENUINE"


def test_diagnosis_falls_back_to_length_without_a_proper_noun():
    """
    "water molecules" has no PROPN, so length remains the best available proxy
    for rarity. The fallback must still work rather than raising.
    """
    assert ce.diagnose_nei(
        "Water molecules are held together by hydrogen bonds.",
        ["Ionic bonds form between oppositely charged particles."],
    ) in {"RETRIEVAL_FAILURE", "GENUINE", "UNKNOWN"}
