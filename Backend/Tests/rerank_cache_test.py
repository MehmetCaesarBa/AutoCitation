"""
Tests for the two CPU-latency fixes in retriever.py:

  1. Dense rerank  — embed only the top-DENSE_CANDIDATES lexical chunks, not all
     of them, and append the rest at score 0.0 so the coverage swap keeps reach.
  2. Article cache — a title resolved once is served from cache on repeat, with
     permanent misses cached and transient failures not.

No network. embed_texts and wikipedia.page are monkeypatched.
"""
import pytest

import config
from Pipeline import retriever


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_chunks(n):
    # The tokenizer is SET-based, so repeating a word does not raise its score.
    # Instead: chunks 1..n-1 each contain the claim token 'alpha' (so they all
    # match and rank above), while chunk 0 contains NO claim token and the
    # unique 'needle' — so it scores 0, sorts last, and lands in the tail. That
    # is exactly the case the coverage swap must still reach.
    chunks = [("Chunk 0: needle unique-zero marker only.", "u0")]
    for i in range(1, n):
        chunks.append((f"Chunk {i}: alpha body text segment number {i}.", f"u{i}"))
    return chunks


def _recording_embed(calls):
    def embed(texts, *, is_query):
        calls.append(("query" if is_query else "docs", len(texts)))
        # vector = [count of 'alpha'] so cosine tracks the lexical signal
        return [[float(t.lower().count("alpha")) or 0.01] for t in texts]
    return embed


@pytest.fixture
def big_chunks():
    return _make_chunks(150)      # >> DENSE_CANDIDATES (60)


# ── 1. only DENSE_CANDIDATES chunks are embedded ──────────────────────────────

def test_dense_embeds_only_the_candidate_pool(monkeypatch, big_chunks):
    calls = []
    monkeypatch.setattr(retriever, "embed_texts", _recording_embed(calls))

    out = retriever.score_chunks_hybrid("alpha claim", big_chunks)

    doc_calls = [n for kind, n in calls if kind == "docs"]
    assert doc_calls == [retriever.DENSE_CANDIDATES], (
        f"expected one doc-embed batch of {retriever.DENSE_CANDIDATES}, got {doc_calls}"
    )
    # Every chunk still comes back (pool reranked + tail), shape preserved.
    assert len(out) == len(big_chunks)
    assert sorted(r[1] for r in out) == sorted(c for c, _ in big_chunks)


def test_tail_is_scored_zero_and_sorts_last(monkeypatch, big_chunks):
    monkeypatch.setattr(retriever, "embed_texts", _recording_embed([]))
    out = retriever.score_chunks_hybrid("alpha claim", big_chunks)

    reranked = [r for r in out if r[0] > 0.0]
    tail     = [r for r in out if r[0] == 0.0]
    assert len(reranked) == retriever.DENSE_CANDIDATES
    assert len(tail) == len(big_chunks) - retriever.DENSE_CANDIDATES
    # Monotonic non-increasing: no tail row outranks a reranked row.
    assert out == sorted(out, key=lambda x: -x[0]) or \
           all(out[i][0] >= out[i + 1][0] for i in range(len(out) - 1))


def test_dense_only_mode_also_caps_embeddings(monkeypatch, big_chunks):
    calls = []
    monkeypatch.setattr(retriever, "embed_texts", _recording_embed(calls))
    out = retriever.score_chunks_dense("alpha claim", big_chunks)
    doc_calls = [n for kind, n in calls if kind == "docs"]
    assert doc_calls == [retriever.DENSE_CANDIDATES]
    assert len(out) == len(big_chunks)


# ── 2. coverage swap still reaches a chunk outside the pool ────────────────────

def test_coverage_swap_reaches_a_tail_chunk(monkeypatch, big_chunks):
    """
    'needle' is in chunk 0 — the lexically WEAKEST chunk, which falls outside the
    embedded pool and lands in the 0.0 tail. select_top_chunks must still be able
    to swap it in, proving the tail preserved the guarantee's reach.
    """
    monkeypatch.setattr(retriever, "embed_texts", _recording_embed([]))
    scored = retriever.score_chunks_hybrid("alpha claim", big_chunks)

    # Sanity: the needle chunk really is in the tail (score 0.0), not the pool.
    needle_rows = [r for r in scored if "needle" in r[1]]
    assert needle_rows and needle_rows[0][0] == 0.0

    chunks, _url = retriever.select_top_chunks(scored, must_contain="needle")
    assert any("needle" in c for c in chunks), "coverage swap could not reach the tail"


# ── 3. embedding failure still degrades to lexical, never crashes ─────────────

def test_hybrid_falls_back_to_lexical_on_embed_failure(monkeypatch, big_chunks):
    def boom(texts, *, is_query):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(retriever, "embed_texts", boom)
    out = retriever.score_chunks_hybrid("alpha claim", big_chunks)
    assert out == retriever.score_chunks("alpha claim", big_chunks)


def test_dense_only_falls_back_to_lexical_on_embed_failure(monkeypatch, big_chunks):
    def boom(texts, *, is_query):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(retriever, "embed_texts", boom)
    out = retriever.score_chunks_dense("alpha claim", big_chunks)
    assert out == retriever.score_chunks("alpha claim", big_chunks)


# ── 4. article content cache ──────────────────────────────────────────────────

class _FakePage:
    def __init__(self, title):
        self.content = f"Full body text of {title}. " * 20
        self.url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"


def test_article_is_fetched_once_then_cached(monkeypatch):
    retriever._article_cache.clear()
    hits = []
    monkeypatch.setattr(retriever.wikipedia, "page",
                        lambda title, **k: hits.append(title) or _FakePage(title))

    a = retriever._load_article("Jupiter")
    b = retriever._load_article("Jupiter")      # repeat in same run
    c = retriever._load_article("jupiter")      # normalised to same key

    assert a == b == c
    assert hits == ["Jupiter"], f"page() should be called once, was {hits}"


def test_permanent_miss_is_cached_transient_is_not(monkeypatch):
    retriever._article_cache.clear()

    # Permanent: DisambiguationError — cached, not retried.
    calls = []
    def disamb(title, **k):
        calls.append(title)
        raise retriever.wikipedia.exceptions.DisambiguationError(title, ["A", "B"])
    monkeypatch.setattr(retriever.wikipedia, "page", disamb)
    assert retriever._load_article("Mercury") is None
    assert retriever._load_article("Mercury") is None
    assert calls == ["Mercury"], "a permanent miss must be cached, not retried"

    # Transient: every attempt raises a generic error — NOT cached, so a later
    # good call still reaches the network.
    retriever._article_cache.clear()
    monkeypatch.setattr(retriever, "RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(retriever, "RETRY_BACKOFF_SECONDS", 0)
    state = {"fail": True, "calls": 0}
    def flaky(title, **k):
        state["calls"] += 1
        if state["fail"]:
            raise ConnectionError("timeout")
        return _FakePage(title)
    monkeypatch.setattr(retriever.wikipedia, "page", flaky)

    assert retriever._load_article("Saturn") is None      # transient fail, uncached
    state["fail"] = False
    got = retriever._load_article("Saturn")               # retried, now succeeds
    assert got is not None and got[1].endswith("Saturn")
    assert state["calls"] == 2, "transient failure must not be cached"
