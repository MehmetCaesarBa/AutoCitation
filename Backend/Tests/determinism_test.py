"""
Determinism guards for token selection.

retriever.rarest_fact_token used to return a different token from run to run on
byte-identical input: _tokenize hands back a SET, max() returns the first
maximal element in the order it receives, and Python randomises string hashing
per process. Two tokens at exactly equal idf therefore swapped places according
to PYTHONHASHSEED — measured at 12 of 24 seeds one way, 12 the other.

A test that fails half the time is worse than one that fails always: it lands on
whoever next runs the suite, points at whatever they last edited, and costs an
afternoon of looking in the wrong file. These tests assert the RESULT and the
REASON, so a future change that restores the coin flip fails here rather than
somewhere unrelated.
"""
import math
from collections import Counter

import Pipeline.retriever as retriever


FACT = "The Empire State Building was completed in 1931."
CHUNKS = [
    ("The Empire State Building is a skyscraper in Manhattan.", "u1"),
    ("The Empire State Building was built from 1930 to 1931.", "u2"),
    ("The Empire State Building has an observation deck.", "u3"),
]


def _idf_table(fact, chunks):
    sets = [retriever._tokenize(c) for c, _ in chunks]
    total = len(sets)
    df = Counter(t for s in sets for t in s)
    return {t: math.log(1 + total / (1 + df.get(t, 0)))
            for t in retriever._tokenize(fact) if df.get(t, 0) > 0}


def test_the_tie_this_guards_against_really_is_a_tie():
    """If this ever stops being an exact tie the test below proves nothing."""
    idf = _idf_table(FACT, CHUNKS)
    assert idf["1931"] == idf["was"], (
        "fixture no longer produces the equal-idf tie these tests exist for"
    )


def test_numeral_wins_an_exact_idf_tie_against_a_function_word():
    """A year settles a dated claim; 'was' is an auxiliary that survived
    _tokenize only for being exactly MIN_SCORE_TOKEN_LENGTH characters long."""
    assert retriever.rarest_fact_token(FACT, CHUNKS) == "1931"


def test_result_is_stable_under_chunk_reordering():
    """Document frequency does not depend on chunk order, so neither may the
    winner. This catches a tie-break that leans on input order instead of the
    tokens themselves."""
    answers = {
        retriever.rarest_fact_token(FACT, list(perm))
        for perm in (
            CHUNKS,
            CHUNKS[::-1],
            [CHUNKS[1], CHUNKS[2], CHUNKS[0]],
            [CHUNKS[2], CHUNKS[0], CHUNKS[1]],
        )
    }
    assert answers == {"1931"}


def test_repeated_calls_agree_within_a_process():
    assert len({retriever.rarest_fact_token(FACT, CHUNKS) for _ in range(50)}) == 1


def test_longer_token_wins_a_tie_between_two_words():
    """With no numeral in play the length component decides, so the outcome is
    still fixed rather than hash-dependent."""
    chunks = [
        ("Alpha beta gamma delta.", "u1"),
        ("Alpha beta gamma epsilon.", "u2"),
        ("Alpha beta gamma zeta.", "u3"),
    ]
    # 'delta' (df=1) and 'kappa' (df=0, excluded); 'epsilon' df=1 too.
    fact = "Delta and epsilon appear here."
    idf = _idf_table(fact, chunks)
    assert idf["delta"] == idf["epsilon"], "expected an exact tie"
    assert retriever.rarest_fact_token(fact, chunks) == "epsilon"


def test_absent_tokens_are_never_selected():
    """A guarantee the evidence cannot satisfy is not a guarantee."""
    assert "completed" not in [t for t in _idf_table(FACT, CHUNKS)]
    assert retriever.rarest_fact_token(FACT, CHUNKS) != "completed"
