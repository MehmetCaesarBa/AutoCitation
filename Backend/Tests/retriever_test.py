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
    chunks, url = retriever.select_top_chunks(scored, must_contain="1931", top_k=1)
    assert chunks == ["The only selected chunk."]
    assert url == "u1"


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


# ═════════════════════════════════════════════════════════════════════════════
# Article search — DuckDuckGo finds, Wikipedia loads
# ═════════════════════════════════════════════════════════════════════════════
import pytest


@pytest.mark.parametrize("url, title", [
    ("https://en.wikipedia.org/wiki/Mount_Everest", "Mount Everest"),
    ("https://en.wikipedia.org/wiki/Mount_Everest#Climbing", "Mount Everest"),
    ("https://en.m.wikipedia.org/wiki/K2", "K2"),
    ("https://en.wikipedia.org/wiki/Summits_farthest_from_the_Earth%27s_center",
     "Summits farthest from the Earth's center"),
    # A colon inside an article title is not a namespace.
    ("https://en.wikipedia.org/wiki/Star_Wars:_Episode_IV", "Star Wars: Episode IV"),
])
def test_wikipedia_title_reads_english_article_urls(url, title):
    assert retriever.wikipedia_title(url) == title


@pytest.mark.parametrize("url", [
    "https://simple.wikipedia.org/wiki/Mount_Everest",   # other edition
    "https://tr.wikipedia.org/wiki/Everest_Da%C4%9F%C4%B1",
    "https://en.wikipedia.org/wiki/Talk:Mount_Everest",  # non-article pages
    "https://en.wikipedia.org/wiki/Category:Mountains",
    "https://en.wikipedia.org/w/index.php?title=K2",
    "https://www.britannica.com/place/Mount-Everest",
    "",
])
def test_wikipedia_title_rejects_everything_else(url):
    assert retriever.wikipedia_title(url) is None


def test_fetch_articles_loads_duckduckgo_hits_in_rank_order(monkeypatch):
    monkeypatch.setattr(retriever, "SEARCH_PROVIDER", "ddgs")
    monkeypatch.setattr(retriever, "_search_duckduckgo",
                        lambda q: ["List of highest mountains on Earth", "Missing page",
                                   "Tallest mountain", "Mount Everest"])
    loaded = {"List of highest mountains on Earth": ("list text", "u1"),
              "Tallest mountain": ("tallest text", "u2"),
              "Mount Everest": ("everest text", "u3")}
    monkeypatch.setattr(retriever, "_load_article", lambda t: loaded.get(t))
    monkeypatch.setattr(retriever, "_fetch_via_wikipedia_search",
                        lambda q: pytest.fail("fallback must not run when DuckDuckGo answered"))

    articles = retriever.fetch_articles("tallest mountain Earth")

    # An unloadable hit is skipped, not counted against TOP_K_ARTICLES.
    assert articles == [("list text", "u1"), ("tallest text", "u2")][:retriever.TOP_K_ARTICLES]


def test_fetch_articles_falls_back_to_wikipedia_search(monkeypatch):
    """DuckDuckGo rate-limits bursts; a failed search must not cost the claim
    its evidence."""
    monkeypatch.setattr(retriever, "SEARCH_PROVIDER", "ddgs")
    monkeypatch.setattr(retriever, "_search_duckduckgo", lambda q: [])
    monkeypatch.setattr(retriever, "_fetch_via_wikipedia_search",
                        lambda q: [("fallback text", "u9")])

    assert retriever.fetch_articles("tallest mountain Earth") == [("fallback text", "u9")]


# ── The DuckDuckGo client ─────────────────────────────────────────────────────
class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code, self.text = status_code, text


def _results_page(*hrefs):
    return "".join(f'<a rel="nofollow" class="result__a" href="{h}">t</a>' for h in hrefs)


@pytest.fixture
def ddg(monkeypatch):
    """Fresh client state, no real sleeping, and a scripted response queue."""
    monkeypatch.setattr(retriever, "_ddg_cache", {})
    monkeypatch.setattr(retriever, "_ddg_blocked_until", 0.0)
    monkeypatch.setattr(retriever, "_ddg_last_request", 0.0)
    monkeypatch.setattr(retriever.time, "sleep", lambda s: None)
    responses, sent = [], []

    def post(url, data, timeout):
        sent.append(data["q"])
        return responses.pop(0)

    monkeypatch.setattr(retriever._ddg_session, "post", post)
    monkeypatch.setattr(retriever, "SEARCH_PROVIDER", "direct")
    return responses, sent


def test_ddg_result_urls_unwraps_redirect_links_and_drops_ads():
    page = _results_page(
        "https://en.wikipedia.org/wiki/K2",
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FMount_Everest&amp;rut=abc",
        "https://duckduckgo.com/y.js?ad_domain=example.com",
    )
    assert retriever._ddg_result_urls(page) == [
        "https://en.wikipedia.org/wiki/K2",
        "https://en.wikipedia.org/wiki/Mount_Everest",
    ]


def test_search_keeps_english_articles_in_rank_order(ddg):
    responses, sent = ddg
    responses.append(_FakeResponse(200, _results_page(
        "https://en.wikipedia.org/wiki/List_of_highest_mountains_on_Earth",
        "https://simple.wikipedia.org/wiki/Mount_Everest",
        "https://en.wikipedia.org/wiki/Mount_Everest",
        "https://en.m.wikipedia.org/wiki/Mount_Everest",
    )))
    assert retriever._search_direct("tallest mountain Earth") == \
        ["List of highest mountains on Earth", "Mount Everest"]
    assert sent == ["tallest mountain Earth site:wikipedia.org"]


def test_a_block_is_not_retried_and_opens_a_cooldown(ddg):
    """Retrying during a block only extends it."""
    responses, sent = ddg
    responses.append(_FakeResponse(202, "anomaly"))

    assert retriever._search_direct("q1") == []
    assert retriever._search_direct("q2") == []   # skipped: no request queued
    assert sent == ["q1 site:wikipedia.org"]


def test_a_block_is_not_cached_as_an_empty_result(ddg, monkeypatch):
    responses, _ = ddg
    responses.append(_FakeResponse(202))
    retriever._search_direct("q")

    monkeypatch.setattr(retriever, "_ddg_blocked_until", 0.0)   # cooldown over
    responses.append(_FakeResponse(200, _results_page("https://en.wikipedia.org/wiki/K2")))
    assert retriever._search_direct("q") == ["K2"]


def test_a_repeated_query_is_answered_from_the_cache(ddg):
    responses, sent = ddg
    responses.append(_FakeResponse(200, _results_page("https://en.wikipedia.org/wiki/K2")))
    retriever._search_duckduckgo("q")
    assert retriever._search_duckduckgo("q") == ["K2"]
    assert len(sent) == 1


def test_unreachable_duckduckgo_returns_nothing(ddg, monkeypatch):
    def boom(*a, **k):
        raise retriever.requests.ConnectionError("offline")
    monkeypatch.setattr(retriever._ddg_session, "post", boom)
    assert retriever._search_direct("q") == []


def test_ddgs_client_is_used_when_selected(monkeypatch):
    monkeypatch.setattr(retriever, "_ddg_cache", {})
    monkeypatch.setattr(retriever, "SEARCH_PROVIDER", "ddgs")
    monkeypatch.setattr(retriever, "_search_ddgs", lambda q: ["K2"])
    monkeypatch.setattr(retriever, "_search_direct",
                        lambda q: pytest.fail("direct client must not run"))
    assert retriever._search_duckduckgo("q") == ["K2"]


# ── wiki_rewrite: Wikipedia search over the query plus LLM rewrites ──────────
class _FakeOllama:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self.text}


@pytest.fixture
def rewrite(monkeypatch):
    monkeypatch.setattr(retriever, "_rewrite_cache", {})
    return monkeypatch


def test_a_name_wikipedia_already_matches_is_not_rewritten(rewrite):
    """Rewriting 'Nepal' would cost ~9s of LLM time for nothing."""
    rewrite.setattr(retriever, "_wiki_search", lambda q: ["Nepal", "Nepali language"])
    rewrite.setattr(retriever, "_rewrite_query",
                    lambda q: pytest.fail("an exact title match must skip the rewrite"))
    assert retriever._search_wiki_rewrite("Nepal") == ["Nepal", "Nepali language"]


def test_rankings_are_fused_so_agreement_beats_one_top_hit(rewrite):
    """The measured case: the original query and one bad rewrite both rank a
    myth article first; two good rewrites rank the real articles near the top."""
    results = {
        "most gigantic creature history": ["Greek mythological creatures", "Kraken"],
        "Largest creature in history":    ["Largest organisms", "Largest and heaviest animals"],
        "Biggest animal ever lived":      ["Largest and heaviest animals", "Largest organisms"],
    }
    rewrite.setattr(retriever, "_wiki_search", lambda q: results[q])
    rewrite.setattr(retriever, "_rewrite_query",
                    lambda q: ["Largest creature in history", "Biggest animal ever lived"])

    titles = retriever._search_wiki_rewrite("most gigantic creature history")

    assert set(titles[:2]) == {"Largest organisms", "Largest and heaviest animals"}
    assert "Greek mythological creatures" in titles, "fusion reorders, never drops"


def test_rewrite_output_is_cleaned_and_capped(rewrite):
    rewrite.setattr(retriever.requests, "post", lambda *a, **k: _FakeOllama(
        '<think>hm</think>1. "Largest animal ever"\n- Biggest animal\n\n'
        'most gigantic creature history\n* Heaviest animal\n5. Fifth one'))
    assert retriever._rewrite_query("most gigantic creature history") == \
        ["Largest animal ever", "Biggest animal", "Heaviest animal"]


def test_rewrite_failure_searches_the_original_only(rewrite):
    def down(*a, **k):
        raise retriever.requests.ConnectionError("ollama not running")
    rewrite.setattr(retriever.requests, "post", down)
    assert retriever._rewrite_query("q") == []


def test_rewrites_are_cached_per_query(rewrite):
    calls = []
    def post(*a, **k):
        calls.append(1)
        return _FakeOllama("Largest animal")
    rewrite.setattr(retriever.requests, "post", post)
    retriever._rewrite_query("q")
    retriever._rewrite_query("q")
    assert len(calls) == 1


def test_wiki_rewrite_provider_returns_more_articles_for_the_chunk_scorer(rewrite):
    rewrite.setattr(retriever, "SEARCH_PROVIDER", "wiki_rewrite")
    rewrite.setattr(retriever, "_search_wiki_rewrite", lambda q: [f"T{i}" for i in range(10)])
    rewrite.setattr(retriever, "_load_article", lambda t: (f"text {t}", f"url {t}"))
    articles = retriever.fetch_articles("q")
    assert len(articles) == retriever.REWRITE_TOP_ARTICLES
