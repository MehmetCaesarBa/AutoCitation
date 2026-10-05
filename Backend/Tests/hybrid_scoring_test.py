"""
Tests for dense and hybrid retrieval scoring (Pipeline/retriever.py).

NO NETWORK. embed_texts is monkeypatched with a deterministic fake so the RRF
fusion, the shape contract, the lexical fallback, and the determinism guarantee
are all testable offline — the scoring path calls no generative LLM, exactly so
this is possible.
"""
import math

import pytest

import config
from Pipeline import retriever


# A tiny toy embedding: map text to a vector by keyword presence. Deterministic,
# no model needed. 'collider' and 'accelerator' share a dimension so they are
# near each other while sharing no surface token — the synonym case the whole
# feature exists for.
_DIMS = [
    ("accelerator", "collider", "particle"),  # dim 0: particle-physics apparatus
    ("cern", "geneva"),                        # dim 1: location
    ("boson", "higgs"),                        # dim 2: what it found
]

def _fake_vec(text: str) -> list[float]:
    t = text.lower()
    return [float(sum(1 for w in group if w in t)) for group in _DIMS]

def _fake_embed(texts, *, is_query):
    # Prefixing is the real embed_texts' job; the fake ignores it but proves the
    # keyword argument contract is honoured by the callers.
    return [_fake_vec(t) for t in texts]


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(retriever, "embed_texts", _fake_embed)


CHUNKS = [
    ("The Large Hadron Collider is a particle accelerator at CERN.", "u_lhc"),
    ("Geneva is a city in Switzerland on Lake Geneva.", "u_geneva"),
    ("The Higgs boson was discovered in 2012.", "u_higgs"),
]


# ── dense scoring ─────────────────────────────────────────────────────────────

def test_dense_returns_score_chunk_url_sorted_desc(fake_embeddings):
    out = retriever.score_chunks_dense("particle collider experiment", CHUNKS)
    assert all(len(row) == 3 for row in out)
    assert [row[1] for row in out][0] == CHUNKS[0][0]      # LHC chunk on top
    scores = [row[0] for row in out]
    assert scores == sorted(scores, reverse=True)


def test_dense_finds_the_paraphrase_lexical_misses(fake_embeddings):
    """
    'collider' vs 'accelerator': zero shared surface tokens on the apparatus
    word, so IDF cannot rank the LHC chunk first on that term alone — dense can,
    because both words share the physics-apparatus dimension.
    """
    out = retriever.score_chunks_dense("Where is the particle collider?", CHUNKS)
    assert out[0][1] == CHUNKS[0][0]
    assert out[0][0] > 0.0


def test_dense_empty_chunks_is_empty(fake_embeddings):
    assert retriever.score_chunks_dense("anything", []) == []


def test_cosine_basics():
    assert retriever._cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert retriever._cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert retriever._cosine([0, 0], [1, 1]) == 0.0     # zero vector, no crash


# ── hybrid / RRF ──────────────────────────────────────────────────────────────

def test_hybrid_shape_and_sort(fake_embeddings):
    out = retriever.score_chunks_hybrid("particle collider at CERN", CHUNKS)
    assert all(len(row) == 3 for row in out)
    scores = [row[0] for row in out]
    assert scores == sorted(scores, reverse=True)
    # Every input chunk survives fusion exactly once.
    assert sorted(r[1] for r in out) == sorted(c for c, _ in CHUNKS)


def test_hybrid_rrf_math_is_correct(fake_embeddings):
    """
    A chunk ranked r_lex by lexical and r_den by dense must fuse to exactly
    1/(K+r_lex) + 1/(K+r_den). Recompute independently and compare.
    """
    fact = "particle collider at CERN"
    lex = retriever.score_chunks(fact, CHUNKS)
    den = retriever.score_chunks_dense(fact, CHUNKS)
    lex_rank = {c: i for i, (_, c, _) in enumerate(lex)}
    den_rank = {c: i for i, (_, c, _) in enumerate(den)}
    K = retriever.RRF_K

    out = retriever.score_chunks_hybrid(fact, CHUNKS)
    for score, chunk, _ in out:
        expected = 1.0 / (K + lex_rank[chunk]) + 1.0 / (K + den_rank[chunk])
        assert score == pytest.approx(expected)


def test_hybrid_falls_back_to_lexical_when_embeddings_fail(monkeypatch):
    """
    Retrieval failure must degrade, never crash. If the embed endpoint is down,
    hybrid returns the pure lexical ranking and the pipeline continues.
    """
    def boom(texts, *, is_query):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(retriever, "embed_texts", boom)

    fact = "particle collider at CERN"
    out = retriever.score_chunks_hybrid(fact, CHUNKS)
    assert out == retriever.score_chunks(fact, CHUNKS)   # identical to lexical


def test_hybrid_empty_chunks_is_empty(fake_embeddings):
    assert retriever.score_chunks_hybrid("anything", []) == []


# ── determinism ───────────────────────────────────────────────────────────────

def test_hybrid_is_deterministic_across_calls(fake_embeddings):
    fact = "particle collider at CERN"
    runs = {tuple(r[1] for r in retriever.score_chunks_hybrid(fact, CHUNKS))
            for _ in range(20)}
    assert len(runs) == 1


# ── the fetch() switch honours the config flag ────────────────────────────────

def test_fetch_switch_selects_the_configured_scorer(monkeypatch, fake_embeddings):
    """
    fetch() must route to the scorer named by config.RETRIEVAL_SCORING and leave
    the rarest-token coverage guarantee in place for every mode.
    """
    monkeypatch.setattr(retriever.ner, "extract_queries", lambda f: ["q"])
    monkeypatch.setattr(retriever, "fetch_articles", lambda q: [("body", "url")])
    monkeypatch.setattr(retriever, "chunk_articles", lambda a: list(CHUNKS))

    # Record call ORDER, not just "the last scorer that ran": hybrid legitimately
    # calls score_chunks and score_chunks_dense internally, so the scorer fetch()
    # routed to is the FIRST entry, not the last.
    calls = []
    for name in ("score_chunks", "score_chunks_dense", "score_chunks_hybrid"):
        orig = getattr(retriever, name)
        def wrap(fact, chunks, _n=name, _o=orig):
            calls.append(_n)
            return _o(fact, chunks)
        monkeypatch.setattr(retriever, name, wrap)

    for mode, expected in [("idf", "score_chunks"),
                           ("dense", "score_chunks_dense"),
                           ("hybrid", "score_chunks_hybrid")]:
        monkeypatch.setattr(config, "RETRIEVAL_SCORING", mode)
        calls.clear()
        retriever.fetch("The particle collider is at CERN.")
        assert calls[0] == expected, (mode, calls)
