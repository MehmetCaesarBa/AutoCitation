"""
Regression tests for the superlative query path.

Three defects are covered, all in the stretch between claim and Wikipedia search:

  1. ner.build_predicate_query deleted the ANALYTIC superlative marker ('most'),
     producing a category query that ranked nothing.
  2. program._type_word read the ranked category off that query — a
     preposition-stripped fragment — and returned the SCOPE noun whenever the
     scope was a common noun ('planet', 'history', 'today').
  3. program._superlative_program never ran the subject query, though
     build_predicate_query's docstring says the predicate query is "an
     ADDITIONAL query, never a replacement".

Every test here is offline: spaCy parses, no Wikipedia, no Ollama. That is the
point — all three defects were deterministic and observable without a single
network call or token generated.
"""
import pytest

import Pipeline.ner as ner
import Pipeline.program as program


# ── 1. The superlative marker survives into the query ─────────────────────────

ANALYTIC = [
    ("The blue whale is considered the most gigantic creature known to have "
     "existed in history.", "most gigantic creature history"),
    ("The Sahara is the most expansive subtropical arid zone on the planet.",
     "most expansive subtropical arid zone planet"),
]


@pytest.mark.parametrize("claim,expected", ANALYTIC)
def test_analytic_superlative_keeps_its_marker(claim, expected):
    """'most' is RBS/ADV/is_stop=True and was dropped by the keep loop.

    Without the marker the query asks no ranking question at all, so
    _handle_question builds "What is the expansive subtropical arid zone
    planet?" and _handle_match compares the subject to whatever comes back.
    """
    query = ner.build_predicate_query(claim)
    assert query == expected
    assert query.startswith("most "), "the ranking operator must lead the query"


INFLECTED = [
    ("The blue whale is the largest animal known to have lived.",
     "largest animal"),
    ("The African bush elephant is the largest living terrestrial animal on "
     "Earth today.", "largest living terrestrial animal Earth today"),
    ("Jamestown is the earliest European permanent settlement in the United "
     "States.", "earliest European permanent settlement United States"),
]


@pytest.mark.parametrize("claim,expected", INFLECTED)
def test_inflected_superlative_query_is_unchanged(claim, expected):
    """Inflected superlatives are JJS/ADJ/not-stop and always passed the POS
    whitelist. The fix must not touch them — it appends via a new branch whose
    `continue` prevents the old branch appending the same token twice."""
    assert ner.build_predicate_query(claim) == expected


def test_no_superlative_still_returns_none():
    assert ner.build_predicate_query(
        "The Amazon is longer than the Nile."
    ) is None


# ── 2. The ranked category comes from the claim, not the query ────────────────

TYPE_WORDS = [
    ("The Sahara is the most expansive subtropical arid zone on the planet.",
     "zone"),
    ("The blue whale is considered the most gigantic creature known to have "
     "existed in history.", "creature"),
    ("The African bush elephant is the largest living terrestrial animal on "
     "Earth today.", "animal"),
    ("Jamestown is the earliest European permanent settlement in the United "
     "States.", "settlement"),
    ("The blue whale is the largest animal known to have lived.", "animal"),
]


@pytest.mark.parametrize("claim,expected", TYPE_WORDS)
def test_type_word_read_from_the_claim(claim, expected):
    assert ner.superlative_type_word(claim) == expected


@pytest.mark.parametrize("claim,scope", [
    ("The Sahara is the most expansive subtropical arid zone on the planet.",
     "planet"),
    ("The blue whale is considered the most gigantic creature known to have "
     "existed in history.", "history"),
    ("The African bush elephant is the largest living terrestrial animal on "
     "Earth today.", "today"),
])
def test_scope_noun_is_never_the_type_word(claim, scope):
    """_type_word's last-NOUN rule holds only while the trailing scope is a
    PROPER noun ('United States', 'East Asia'). 'on the planet', 'in history'
    and 'on Earth today' end in a COMMON noun, so it returned the scope —
    producing "The answer is the today itself" for a question about an animal.
    """
    assert ner.superlative_type_word(claim) != scope


def test_type_word_none_without_a_superlative():
    assert ner.superlative_type_word("The Amazon flows through Brazil.") is None


def test_program_carries_the_claim_derived_type_word():
    claim = "The Sahara is the most expansive subtropical arid zone on the planet."
    prog = program.synthesize(claim)
    assert prog is not None and prog.kind == "superlative"
    assert prog.type_word == "zone"
    # ...and it disagrees with what the query alone would have said, which is
    # the whole reason the field exists.
    assert program._type_word(prog.query) == "planet"


# ── 3. The subject query is a fallback, never a pool ──────────────────────────

def test_fallback_is_not_consulted_when_the_category_query_answers(monkeypatch):
    """THE JAMESTOWN NO-REGRESSION TEST.

    The predicate query reaches the rival (Saint Augustine) only because the
    subject's own article is absent from the evidence. If the subject query ran
    unconditionally — or were pooled in — the model would read Jamestown-centric
    prose, answer 'Jamestown', and Match would return a false SUPPORTS.

    So: when the primary attempt answers, the fallback must never be called.
    """
    calls = []

    def fake_once(query, rank_against=None, type_word="", **_kwargs):
        calls.append(query)
        return "Saint Augustine", ["chunk"], "http://example/1"

    monkeypatch.setattr(program, "_question_once", fake_once)

    answer, chunks, url, from_fallback = program._handle_question(
        "earliest European permanent settlement United States",
        fallback_queries=("Jamestown",),
        type_word="settlement",
    )

    assert answer == "Saint Augustine"
    assert from_fallback is False
    assert calls == ["earliest European permanent settlement United States"], \
        "the subject query must not run when the category query answered"


def test_fallback_runs_only_after_the_primary_finds_nothing(monkeypatch):
    calls = []

    def fake_once(query, rank_against=None, type_word="", **_kwargs):
        calls.append((query, rank_against))
        if query == "most expansive subtropical arid zone planet":
            return None, [], ""
        return "Sahara", ["the largest hot desert in the world"], "http://example/2"

    monkeypatch.setattr(program, "_question_once", fake_once)

    answer, chunks, url, from_fallback = program._handle_question(
        "most expansive subtropical arid zone planet",
        fallback_queries=("The Sahara",),
        type_word="zone",
    )

    assert answer == "Sahara"
    assert from_fallback is True, "the caller must be able to see this was weaker"
    # Fetched by subject, ranked by category — the subject name appears in most
    # chunks of its own article and so discriminates nothing.
    assert calls[1] == ("The Sahara", "most expansive subtropical arid zone planet")


def test_fallback_identical_to_the_primary_is_skipped(monkeypatch):
    calls = []

    def fake_once(query, rank_against=None, type_word="", **_kwargs):
        calls.append(query)
        return None, [], ""

    monkeypatch.setattr(program, "_question_once", fake_once)

    answer, _chunks, _url, from_fallback = program._handle_question(
        "largest animal", fallback_queries=("largest animal", ""),
    )

    assert answer is None and from_fallback is False
    assert calls == ["largest animal"], "no duplicate or empty retry"


def test_retrieve_fetches_by_one_query_and_ranks_by_another(monkeypatch):
    seen = {}

    monkeypatch.setattr(program.retriever, "fetch_articles",
                        lambda q: seen.setdefault("fetch", q) and [("body", "url")]
                        or [("body", "url")], raising=False)
    monkeypatch.setattr(program.retriever, "chunk_articles",
                        lambda a: [("chunk", "url")], raising=False)

    def fake_rank(q, chunks, top_k):
        seen["rank"] = q
        return ["chunk"], "url"

    monkeypatch.setattr(program.retriever, "rank_chunks", fake_rank, raising=False)

    program._retrieve("The Sahara", 2, rank_against="most expansive arid zone")

    assert seen["fetch"] == "The Sahara"
    assert seen["rank"] == "most expansive arid zone"


def test_retrieve_defaults_to_ranking_by_the_fetch_query(monkeypatch):
    seen = {}
    monkeypatch.setattr(program.retriever, "fetch_articles",
                        lambda q: [("body", "url")], raising=False)
    monkeypatch.setattr(program.retriever, "chunk_articles",
                        lambda a: [("chunk", "url")], raising=False)

    def fake_rank(q, chunks, top_k):
        seen["rank"] = q
        return ["chunk"], "url"

    monkeypatch.setattr(program.retriever, "rank_chunks", fake_rank, raising=False)

    program._retrieve("largest animal", 2)
    assert seen["rank"] == "largest animal", "existing callers must be unaffected"


def test_program_ranks_chunks_exactly_like_the_verifier_path(monkeypatch):
    """The two paths share retriever.rank_chunks: the configured scorer AND
    the rarest-token selection, with only the chunk count differing."""
    seen = {}
    monkeypatch.setattr(program.retriever, "fetch_articles", lambda q: [("body", "url")])
    monkeypatch.setattr(program.retriever, "chunk_articles", lambda a: [("chunk", "url")])

    def fake_rank(text, chunks, top_k):
        seen.update(text=text, top_k=top_k)
        return ["chunk"], "url"

    monkeypatch.setattr(program.retriever, "rank_chunks", fake_rank)
    assert program._retrieve("most gigantic creature history", program.QUESTION_TOP_K) ==         (["chunk"], "url")
    assert seen == {"text": "most gigantic creature history", "top_k": program.QUESTION_TOP_K}
