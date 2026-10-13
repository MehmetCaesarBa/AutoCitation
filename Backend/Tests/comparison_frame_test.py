"""
Tests for the comparison frame in Pipeline/program.py.

One frame replaces the superlative/comparative split, because claims carry
both at once: "K2 is the second-highest peak on Earth" is a superlative with a
rank, and "The Nile is longer than any other river in Africa" is a comparative
that ranks. Offline: spaCy parses and pure functions, no Ollama, no Wikipedia.
"""
import pytest

import Pipeline.ner as ner
from Pipeline import program as prog


# ═════════════════════════════════════════════════════════════════════════════
# Rank — the ordinal is part of the question
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("claim, rank, query", [
    ("K2 is the second-highest peak on Earth.", 2, "second highest peak Earth"),
    ("Kangchenjunga is the third-highest mountain in the world.", 3,
     "third highest mountain world"),
    ("K2 is the 2nd highest mountain.", 2, "2nd highest mountain"),
    ("Mount Everest is the highest mountain in the world.", 1, "highest mountain world"),
])
def test_ordinal_rank_reaches_the_question(claim, rank, query):
    """
    THE K2 BUG. 'second' is tagged RB and was dropped from the predicate query,
    so the claim was checked against "what is the highest peak?" — Everest —
    and a true claim could never be supported.
    """
    frame = prog.build_frame(claim)
    assert frame is not None and frame.kind == "superlative"
    assert frame.rank == rank
    assert frame.query == query


def test_ordinal_not_beside_a_superlative_is_not_a_rank():
    """'second' alone ranks nothing a lookup can check ("the second day")."""
    doc = ner.NLP("Everest is the highest mountain.")
    sup = next(t for t in doc if t.text == "highest")
    assert ner.superlative_rank(sup) == (1, None)


def test_rank_two_gets_no_numeric_check():
    """
    A rival named second-highest may stand above or below the subject, so a
    numeric comparison against it corroborates nothing.
    """
    p = prog.synthesize("K2 is the second-highest peak on Earth.")
    assert [s.op for s in p.steps] == ["Question", "Match"]


def test_rank_one_with_a_dimension_gets_the_numeric_check():
    p = prog.synthesize("Lake Baikal is the deepest lake in the world.")
    assert [s.op for s in p.steps] == ["Question", "Match", "Compare"]


# ═════════════════════════════════════════════════════════════════════════════
# "than any other X" — a superlative written as a comparison
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("claim", [
    "The Nile is longer than any other river in Africa.",
    "The Nile is longer than all other rivers in Africa.",
])
def test_than_any_other_is_a_ranking_claim(claim):
    """
    Used to compare Quantity('Nile') against Quantity('river in Africa').
    """
    frame = prog.build_frame(claim)
    assert frame is not None and frame.rank == 1 and frame.reference is None
    assert frame.category == "river" and frame.scope == "Africa"
    assert frame.query == "longest river Africa"
    assert frame.dimension == "length"


def test_than_a_named_entity_stays_pairwise():
    frame = prog.build_frame("The Amazon is longer than the Nile.")
    assert frame is not None and frame.rank is None and frame.reference == "Nile"


@pytest.mark.parametrize("comparative, superlative", [
    ("longer", "longest"), ("bigger", "biggest"), ("heavier", "heaviest"),
    ("better", "best"),
])
def test_superlative_form(comparative, superlative):
    token = ner.NLP(f"It is {comparative} than any other.")[2]
    assert prog._superlative_form(token) == superlative


# ═════════════════════════════════════════════════════════════════════════════
# Direction — which value wins
# ═════════════════════════════════════════════════════════════════════════════
def test_less_populous_points_down():
    """Was compared as if it said MORE populous."""
    frame = prog.build_frame("India is less populous than China.")
    assert frame.dimension == "population" and frame.direction == "min"


def test_lowest_points_down():
    frame = prog.build_frame("The Dead Sea is the lowest point on land.")
    assert frame.dimension == "height" and frame.direction == "min"


def test_compare_respects_direction():
    # "A is less populous than B" claims a < b.
    assert prog._handle_compare(1.0e9, 1.4e9, "A", "B", "population",
                                "1.0 billion", "1.4 billion", "min")[0] == "SUPPORTS"
    assert prog._handle_compare(1.4e9, 1.0e9, "A", "B", "population",
                                "1.4 billion", "1.0 billion", "min")[0] == "REFUTES"


# ═════════════════════════════════════════════════════════════════════════════
# Where the dimension comes from
# ═════════════════════════════════════════════════════════════════════════════
def test_copular_category_noun_uses_the_adjective():
    """'Everest is a higher PEAK': the noun is what Everest IS, not an attribute."""
    frame = prog.build_frame("Everest is a higher peak than K2.")
    assert frame is not None and frame.dimension == "height"


def test_ambiguous_superlative_resolved_by_a_by_phrase():
    """'largest' alone is ambiguous; 'by area' says which."""
    frame = prog.build_frame("Russia is the largest country by area.")
    assert frame.dimension == "area" and frame.direction == "max"


def test_ambiguous_superlative_without_a_by_phrase_has_no_dimension():
    frame = prog.build_frame("Russia is the largest country in the world.")
    assert frame is not None and frame.dimension is None


# ═════════════════════════════════════════════════════════════════════════════
# Quantities — pint, guarded against English
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("sentence, family, expected", [
    ("The Nile is about 6,650 km (4,130 mi) long.", "length", 6_650_000),
    ("K2 is 8,611 metres high.", "length", 8_611),
    ("Russia covers 17,098,246 km2.", "area", 17_098_246e6),
    ("It covers 9.8 million km² of land.", "area", 9.8e12),
    ("Russia covers 6,601,670 sq mi.", "area", 17_098_246e6),
    ("The blue whale can weigh up to 199 tonnes.", "mass", 199_000),
    ("India has a population of 1.4 billion people.", "count", 1.4e9),
    ("In 2020 the population was 1,411,778,724.", "count", 1_411_778_724),
])
def test_measurements_convert_to_base_units(sentence, family, expected):
    values = [v for v, _ in prog._measurements(sentence, family)]
    assert values and values[0] == pytest.approx(expected, rel=1e-3), values


@pytest.mark.parametrize("sentence, family", [
    ("It was measured at 8,849 in 2020.", "length"),   # pint reads 'in' as inch
    ("Cars travel at 120 km/h.", "length"),            # a speed is not a length
    ("K2 is in the Karakoram.", "length"),             # '2' glued to 'K'
    ("The survey was published in 2020.", "count"),    # a year is not a tally
])
def test_measurements_reject_false_units(sentence, family):
    assert prog._measurements(sentence, family) == []


def test_wrong_family_is_rejected():
    """A length in a sentence about mass must not be read as a mass."""
    assert prog._to_base("50", "km", "mass") is None


# ═════════════════════════════════════════════════════════════════════════════
# Numeric corroboration of a rank-1 REFUTES
# ═════════════════════════════════════════════════════════════════════════════
def _stub_quantities(monkeypatch, values):
    def fake(entity, dimension, family):
        v = values.get(entity)
        return v, (f"{v:g} m" if v is not None else ""), [f"chunk about {entity}"], "url"
    monkeypatch.setattr(prog, "_handle_quantity", fake)


DEEPEST = prog.ComparisonFrame(subject="Lake Tanganyika", marker="deepest",
                               dimension="depth", direction="max", rank=1)


def test_numeric_route_corroborates_when_the_rival_wins(monkeypatch):
    _stub_quantities(monkeypatch, {"Lake Baikal": 1642.0, "Lake Tanganyika": 1470.0})
    verdict, trace, _, _ = prog._numeric_corroborate("Lake Baikal", "Lake Tanganyika", DEEPEST)
    assert verdict is True and "rival wins" in trace


def test_numeric_route_disagrees_when_the_subject_wins(monkeypatch):
    _stub_quantities(monkeypatch, {"Lake Baikal": 1470.0, "Lake Tanganyika": 1642.0})
    verdict, _, _, _ = prog._numeric_corroborate("Lake Baikal", "Lake Tanganyika", DEEPEST)
    assert verdict is False


def test_numeric_route_is_undecided_without_values(monkeypatch):
    _stub_quantities(monkeypatch, {"Lake Baikal": 1642.0})
    verdict, _, _, _ = prog._numeric_corroborate("Lake Baikal", "Lake Tanganyika", DEEPEST)
    assert verdict is None


# ═════════════════════════════════════════════════════════════════════════════
# The model-filled frame — validation, with the model stubbed
# ═════════════════════════════════════════════════════════════════════════════
def _stub_model(monkeypatch, reply: str):
    from models import llm
    monkeypatch.setattr(llm, "generate", lambda *a, **k: llm.Generation(
        text=reply, thinking="", truncated=False, eval_count=0, eval_seconds=0.0))


K2 = "K2 is the second-highest peak on Earth."


def test_llm_frame_is_used_when_it_validates(monkeypatch):
    _stub_model(monkeypatch, '{"comparison":"ranking","subject":"K2","reference":"",'
                '"superlative":"highest","dimension":"height","direction":"max",'
                '"rank":2,"category":"peak","scope":"Earth"}')
    monkeypatch.setattr(prog, "FRAME_PARSER", "llm")
    frame = prog.build_frame(K2)
    assert frame.source == "llm" and frame.rank == 2
    assert frame.query == "second highest peak Earth"


@pytest.mark.parametrize("reply", [
    # Subject not in the claim: a name from the model's own knowledge.
    '{"comparison":"ranking","subject":"Mount Everest","reference":"","superlative":"highest",'
    '"dimension":"height","direction":"max","rank":2,"category":"peak","scope":"Earth"}',
    # Rank disagrees with the ordinal the parse found.
    '{"comparison":"ranking","subject":"K2","reference":"","superlative":"highest",'
    '"dimension":"height","direction":"max","rank":1,"category":"peak","scope":"Earth"}',
    # Not JSON at all.
    'K2 is a mountain.',
    # Says "none" where the parse found a ranking.
    '{"comparison":"none","subject":"K2","reference":"","superlative":"",'
    '"dimension":"none","direction":"max","rank":1,"category":"","scope":""}',
])
def test_invalid_llm_frame_falls_back_to_the_parse(monkeypatch, reply):
    _stub_model(monkeypatch, reply)
    monkeypatch.setattr(prog, "FRAME_PARSER", "llm")
    frame = prog.build_frame(K2)
    assert frame is not None and frame.source == "rules" and frame.rank == 2


# ═════════════════════════════════════════════════════════════════════════════
# TableRank — rows selected by their rank cell, not by similarity
# ═════════════════════════════════════════════════════════════════════════════
_M = prog.retriever.TABLE_MARKER
_ARTICLES = [
    ("Prose about mountains. Rank: 2; a sentence in the prose is not a row." + _M +
     "List of highest mountains on Earth - Rank: 1; Mountain name(s): Mount Everest; Height: 8,849 m.\n"
     "List of highest mountains on Earth - Rank: 2; Mountain name(s): K2; Height: 8,611 m.\n"
     "List of highest mountains on Earth - Rank: 12; Mountain name(s): Broad Peak; Height: 8,051 m.",
     "https://en.wikipedia.org/wiki/List_of_highest_mountains_on_Earth"),
    ("River prose." + _M +
     "List of rivers by length - Rank: 2.; River: Amazon; Length (km): 6,400.",
     "https://en.wikipedia.org/wiki/List_of_rivers_by_length"),
]


def test_ranked_rows_select_by_the_rank_cell():
    """
    Similarity ranking picked the rows ranked 11, 12, 57 and 60 for 'second
    highest peak Earth' — the ones named '... Peak'. The lookup reads the cell.
    """
    rows = prog._ranked_rows(_ARTICLES, 2)
    assert [r for r, _ in rows] == [
        "List of highest mountains on Earth - Rank: 2; Mountain name(s): K2; Height: 8,611 m.",
        "List of rivers by length - Rank: 2.; River: Amazon; Length (km): 6,400.",
    ]


def test_ranked_rows_do_not_match_a_longer_number():
    """'Rank: 1' must not match the row ranked 12."""
    rows = prog._ranked_rows(_ARTICLES, 1)
    assert len(rows) == 1 and "Mount Everest" in rows[0][0]


def test_ranked_rows_come_first_and_carry_the_citation(monkeypatch):
    monkeypatch.setattr(prog.retriever, "fetch_articles", lambda q: _ARTICLES)
    monkeypatch.setattr(prog.retriever, "rank_chunks",
                        lambda text, chunks, top_k: (["a prose chunk"], "prose-url"))
    chunks, url = prog._retrieve("second highest peak Earth", 4, rank=2)
    assert chunks[0].startswith("List of highest mountains on Earth - Rank: 2;")
    assert chunks[-1] == "a prose chunk"
    assert url == "https://en.wikipedia.org/wiki/List_of_highest_mountains_on_Earth"


def test_table_rows_are_not_chunked():
    """Rows in the similarity pool displaced the prose that answered Jamestown."""
    chunks = prog.retriever.chunk_articles(_ARTICLES[:1])
    assert chunks and all("Mount Everest" not in c for c, _ in chunks)


# ═════════════════════════════════════════════════════════════════════════════
# A SUPPORTS from the subject's own article fails closed
# ═════════════════════════════════════════════════════════════════════════════
def test_fallback_supports_becomes_not_enough_info(monkeypatch):
    """
    Live: Jamestown's article ("first permanent ENGLISH settlement") was read
    as entailing "earliest EUROPEAN", and a false claim came back SUPPORTS.
    """
    monkeypatch.setattr(prog, "_handle_question",
                        lambda *a, **k: ("Jamestown", ["Jamestown chunk"], "url", True))
    claim = "Jamestown is the earliest European permanent settlement in the United States."
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "NOT ENOUGH INFO"


def test_fallback_rival_still_refutes(monkeypatch):
    """The subject's article naming someone else is evidence against it."""
    monkeypatch.setattr(prog, "_handle_question",
                        lambda *a, **k: ("K2", ["K2 chunk"], "url", True))
    claim = "Kangchenjunga is the second-highest peak on Earth."
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "REFUTES"
