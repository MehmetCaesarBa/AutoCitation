"""
Unit tests for Pipeline/retriever.py — ranking and selection only.

No network. Chunks are supplied directly, so these test the scoring and
selection logic rather than Wikipedia's search behaviour.
"""

from Pipeline import retriever


# ═════════════════════════════════════════════════════════════════════════════
# rarest_fact_token
# ═════════════════════════════════════════════════════════════════════════════
def test_rarest_token_is_the_one_that_narrows_the_search():
    """
    Document frequency is measured over the retrieved pool, so words appearing
    in every chunk weigh nothing and a year weighs heavily.
    """
    chunks = [
        ("The Empire State Building is a skyscraper in Manhattan.", "u1"),
        ("The Empire State Building was built from 1930 to 1931.", "u2"),
        ("The Empire State Building has an observation deck.", "u3"),
    ]
    token = retriever.rarest_fact_token(
        "The Empire State Building was completed in 1931.", chunks
    )
    assert token == "1931", token


def test_rarest_token_ignores_words_no_chunk_contains():
    """
    A guarantee that cannot be satisfied is not a guarantee. Insisting on a word
    nobody wrote would discard a good chunk for nothing.
    """
    chunks = [("Wholly unrelated text about rivers.", "u1")]
    token = retriever.rarest_fact_token("Jamestown was founded in 1607.", chunks)
    assert token is None or token in "wholly unrelated text about rivers"


def test_rarest_token_handles_empty_input():
    assert retriever.rarest_fact_token("", []) is None
    assert retriever.rarest_fact_token("A claim.", []) is None


# ═════════════════════════════════════════════════════════════════════════════
# select_top_chunks — the coverage guarantee
# ═════════════════════════════════════════════════════════════════════════════
def test_decisive_token_is_swapped_into_the_selection():
    """
    THE EMPIRE STATE FAILURE. Scoring treats tokens independently, so a chunk
    matching 'completed' twice — as an adjective about other buildings — beat
    the chunk carrying '1931', the only token that could settle the claim. The
    verifier answered NOT ENOUGH INFO about evidence it should never have had.
    """
    scored = [
        (0.76, "The tenth-tallest completed skyscraper, developed in 1893.", "u1"),
        (0.60, "In 1929, Empire State Inc. acquired the site.", "u2"),
        (0.40, "The Empire State Building was built from 1930 to 1931.", "u3"),
    ]
    chunks, url = retriever.select_top_chunks(scored, must_contain="1931")

    assert any("1931" in c for c in chunks), chunks
    # The strongest chunk survives — the citation URL is taken from it, so
    # trading it away would fix the evidence and break the source.
    assert chunks[0] == scored[0][1]
    assert url == "u1"


def test_no_swap_when_the_token_is_already_present():
    scored = [
        (0.90, "Founded on May 14, 1607, at Jamestown.", "u1"),
        (0.50, "Unrelated filler about the museum complex.", "u2"),
        (0.30, "Another chunk mentioning 1607 as well.", "u3"),
    ]
    chunks, _ = retriever.select_top_chunks(scored, must_contain="1607")
    assert chunks == [scored[0][1], scored[1][1]][:len(chunks)]


def test_no_swap_when_no_chunk_has_the_token():
    """Nothing to swap in — the selection is left alone rather than damaged."""
    scored = [
        (0.70, "Text without the decisive term.", "u1"),
        (0.50, "Also without it.", "u2"),
    ]
    chunks, _ = retriever.select_top_chunks(scored, must_contain="1931")
    assert chunks == ["Text without the decisive term.", "Also without it."]


def test_single_chunk_is_never_replaced():
    """
    Replacing the only selection is not a swap, it is a different search — and
    it would take the citation URL with it.
    """
    scored = [
        (0.70, "The only selected chunk.", "u1"),
        (0.10, "A chunk containing 1931.", "u2"),
    ]
    original = retriever.TOP_K_CHUNKS
    retriever.TOP_K_CHUNKS = 1
    try:
        chunks, url = retriever.select_top_chunks(scored, must_contain="1931")
        assert chunks == ["The only selected chunk."]
        assert url == "u1"
    finally:
        retriever.TOP_K_CHUNKS = original


def test_article_ceiling_exceeds_what_reaches_the_verifier():
    """
    DOC_CONTENT_CHARS_MAX was 4000 on the belief that it protected the model's
    context window. TOP_K_CHUNKS does that — two chunks of ~CHUNK_SIZE reach the
    verifier however long the article is.

    Asserted as a relationship rather than a literal, so the invariant survives
    someone retuning either number: the article ceiling must stay far above what
    selection lets through, or it is deleting sources instead of protecting
    anything.
    """
    reaches_verifier = retriever.TOP_K_CHUNKS * retriever.CHUNK_SIZE
    assert retriever.DOC_CONTENT_CHARS_MAX > reaches_verifier * 10, (
        f"the article ceiling ({retriever.DOC_CONTENT_CHARS_MAX}) is close to "
        f"what selection delivers ({reaches_verifier}) — it is truncating "
        f"evidence, not guarding the context window"
    )


def test_selection_without_a_guarantee_is_unchanged():
    """The parameter is optional; omitting it must not alter behaviour."""
    scored = [(0.9, "a", "u1"), (0.5, "b", "u2"), (0.1, "c", "u3")]
    chunks, url = retriever.select_top_chunks(scored)
    assert chunks == ["a", "b"][:retriever.TOP_K_CHUNKS]
    assert url == "u1"
