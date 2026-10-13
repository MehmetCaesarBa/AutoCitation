"""
Tests for the Wikidata route: Pipeline/wikidata.py and how program.py uses it.

Offline. wikidata.value is stubbed with the values measured live (K2 8,611 m,
Mount Everest 8,850 m), so these test the DECISIONS, not the endpoint.
"""
import pytest

from Pipeline import program as prog
from Pipeline import wikidata

MEASURED = {
    ("K2", "mountain"): wikidata.WikidataValue("K2", "Q43512", 8611.0),
    ("Mount Everest", "mountain"): wikidata.WikidataValue("Mount Everest", "Q513", 8850.0),
    ("Mauna Kea", "mountain"): wikidata.WikidataValue("Mauna Kea", "Q131230", 4207.3),
}


@pytest.fixture
def stub_wikidata(monkeypatch):
    calls = []

    def fake(title, category, dimension):
        calls.append(title)
        return MEASURED.get((title, category))

    monkeypatch.setattr(wikidata, "value", fake)
    return calls


def _no_confirmed_answer(rejected_names):
    """_handle_question as it behaved live: candidates named, none confirmed."""
    def fake(*args, rejected=None, **kwargs):
        if rejected is not None:
            rejected.extend(rejected_names)
        return None, ["evidence chunk"], "url", False
    return fake


# ═════════════════════════════════════════════════════════════════════════════
# Mapping — what may be asked at all
# ═════════════════════════════════════════════════════════════════════════════
def test_supports_only_mapped_category_and_dimension_pairs():
    assert wikidata.supports("mountain", "height")
    assert wikidata.supports("Peak", "height")
    assert not wikidata.supports("mountain", "population")   # no such property pairing
    assert not wikidata.supports("settlement", "height")      # unmapped category
    assert not wikidata.supports(None, "height")


def test_value_refuses_unmapped_queries_without_a_request(monkeypatch):
    monkeypatch.setattr(wikidata.requests, "get",
                        lambda *a, **k: pytest.fail("no request for an unmapped pair"))
    assert wikidata.value("Jamestown", "settlement", "height") is None


def test_value_fails_quiet_on_a_network_error(monkeypatch):
    def boom(*a, **k):
        raise wikidata.requests.ConnectionError("down")
    monkeypatch.setattr(wikidata.requests, "get", boom)
    wikidata.value.cache_clear()
    assert wikidata.value("K2", "mountain", "height") is None


# ═════════════════════════════════════════════════════════════════════════════
# Wikidata alone may refute — and only under its three conditions
# ═════════════════════════════════════════════════════════════════════════════
def test_named_candidate_that_beats_the_subject_refutes(monkeypatch, stub_wikidata):
    """
    THE LIVE CASE. The Everest table row entails "Everest is the highest
    mountain" at only p=0.80, so no holder was confirmed and the claim went to
    NOT ENOUGH INFO. Any mountain above K2 refutes it.
    """
    monkeypatch.setattr(prog, "_handle_question", _no_confirmed_answer(["Mount Everest"]))
    claim = "K2 is the highest mountain in the world."
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "REFUTES"
    assert "Q513" in outcome.verdict.rationale and "8,850 m" in outcome.verdict.rationale


def test_candidate_below_the_subject_does_not_refute(monkeypatch, stub_wikidata):
    monkeypatch.setattr(prog, "_handle_question", _no_confirmed_answer(["Mauna Kea"]))
    claim = "K2 is the highest mountain in the world."
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "NOT ENOUGH INFO"


@pytest.mark.parametrize("claim, why", [
    ("K2 is the highest mountain in Pakistan.", "scope is not world-wide"),
    ("K2 is the highest unclimbed mountain in the world.", "category is narrowed"),
    ("K2 is the second-highest mountain in the world.", "rank is not 1"),
])
def test_wikidata_alone_does_not_refute_outside_its_conditions(monkeypatch, stub_wikidata,
                                                              claim, why):
    """Everest beats K2, and still refutes none of these."""
    monkeypatch.setattr(prog, "_handle_question", _no_confirmed_answer(["Mount Everest"]))
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "NOT ENOUGH INFO", why


# ═════════════════════════════════════════════════════════════════════════════
# Wikidata corroborates a text-route REFUTES before the prose numbers do
# ═════════════════════════════════════════════════════════════════════════════
def test_wikidata_confirms_a_refutes_without_reading_prose(monkeypatch, stub_wikidata):
    monkeypatch.setattr(prog, "_handle_question",
                        lambda *a, **k: ("Mount Everest", ["chunk"], "url", False))
    monkeypatch.setattr(prog, "_numeric_corroborate",
                        lambda *a: pytest.fail("prose numbers are the fallback only"))
    claim = "K2 is the highest mountain in the world."
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "REFUTES"
    assert "The values agree" in outcome.verdict.rationale


def test_wikidata_disagreement_withholds_the_verdict(monkeypatch, stub_wikidata):
    """A 'rival' Wikidata puts below the subject: the two routes disagree."""
    monkeypatch.setattr(prog, "_handle_question",
                        lambda *a, **k: ("Mauna Kea", ["chunk"], "url", False))
    claim = "K2 is the highest mountain in the world."
    outcome = prog.execute(prog.synthesize(claim), claim)
    assert outcome.verdict.label == "NOT ENOUGH INFO"


def test_frame_scope_and_bareness():
    f = prog.build_frame("K2 is the second-highest peak on Earth.")
    assert f.scope == "Earth" and f.world_scope and f.bare
    f = prog.build_frame("K2 is the highest mountain in Pakistan.")
    assert not f.world_scope
    f = prog.build_frame("Gangkhar Puensum is the highest unclimbed mountain in the world.")
    assert not f.bare


def test_unreadable_by_phrase_drops_the_adjectives_dimension():
    """'highest BY PROMINENCE' is not elevation, whatever 'highest' says."""
    f = prog.build_frame("Denali is the highest mountain by prominence in North America.")
    assert f.dimension is None
