import html
import math
import re
import threading
import time
from collections import Counter
from urllib.parse import parse_qs, unquote, urlparse

import requests
import wikipedia
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_community.retrievers import WikipediaRetriever

import Pipeline.ner as ner
import config

# ── Wikipedia client identification ───────────────────────────────────────────
# The `wikipedia` package (used internally by LangChain's WikipediaRetriever)
# ships a generic default User-Agent that Wikimedia's bot policy now blocks
# or rate-limits — when that happens the API answers with an HTML error page,
# and the library dies with "Expecting value: line 1 column 1 (char 0)"
# (it tried to json-parse HTML). Per https://meta.wikimedia.org/wiki/User-Agent_policy
# clients must send an identifying UA with contact info.
WIKI_USER_AGENT = "AutoCitation/0.1 (PoC fact-checker; contact: mehmet17b.b@gmail.com)"

# ── Dense / hybrid retrieval scoring ──────────────────────────────────────────
# The embedding endpoint is the /api/embed sibling of the /api/generate endpoint
# verifier.py and program.py already call. Built from config.OLLAMA_BASE_URL so a
# port or host change stays a one-line edit there, as config.py documents.
OLLAMA_EMBED_URL = f"{config.OLLAMA_BASE_URL}/api/embed"

# Reciprocal Rank Fusion constant (Cormack et al. 2009). Fuses by RANK, not
# score: cosine similarity (~0.3-0.9) and IDF coverage (0-1) live on different
# scales, and any weighted sum of the two would bake in an arbitrary exchange
# rate. Ranks have no units. k=60 damps the top-rank advantage so one scorer's
# confident mistake cannot override a consensus of the other's.
RRF_K = 60

# DENSE_CANDIDATES: how many chunks the dense scorer actually embeds.
#
# WHY NOT ALL OF THEM. score_chunks_dense used to embed every chunk a claim
# retrieved — 237, 249, 284 — on CPU, to pick the TOP_K_CHUNKS (2) that reach
# the verifier. Measured on a live run, embedding ~560 chunks across the two
# retrieval rounds of one superlative claim was ~200s of a ~250s request: the
# dense pass, not the LLM, was the pipeline's dominant cost.
#
# This is a first-stage/rerank split, the standard shape: the cheap IDF scorer
# (microseconds over all chunks) nominates the top DENSE_CANDIDATES, and only
# those are embedded and reranked. ~250 embeddings -> ~60 is a ~4x cut with no
# measurable recall loss, because a chunk worth embedding almost always shares
# at least one claim token and so already ranks well lexically.
#
# THE KNOWN BLIND SPOT, STATED. A pure paraphrase that shares ZERO tokens with
# the claim ("particle collider" vs a chunk saying only "particle accelerator"
# would still share 'particle'; a chunk sharing nothing at all) ranks low
# lexically and can fall outside the pool before dense ever sees it. That is the
# accepted limitation of every first-stage-then-rerank retriever; raise this
# number to widen the net at the cost of more embedding time. 60 of ~250 keeps
# the top ~quarter, which covers every genuine candidate observed so far.
DENSE_CANDIDATES = 60

try:
    wikipedia.set_user_agent(WIKI_USER_AGENT)
except AttributeError:
    # Older package versions without the setter: patch the module global
    # the request layer reads at call time.
    wikipedia.USER_AGENT = WIKI_USER_AGENT

# Throttle successive API calls — the per-fact loop fires many requests in
# quick bursts, which is exactly the pattern that triggers rate limiting.
wikipedia.set_rate_limiting(True)

# ── Fetch retry config ────────────────────────────────────────────────────────
# One transient block/timeout should not cost the fact its entire evidence
# set (a miss cascades into a guaranteed NOT ENOUGH INFO).
RETRY_ATTEMPTS        = 3
RETRY_BACKOFF_SECONDS = 1.5

# ── Wikipedia Retriever config ────────────────────────────────────────────────
# top_k_results: Number of Wikipedia articles fetched per query.
# Two articles provide enough surface area to cover multi-hop claims
# without flooding chunk_articles with irrelevant documents.
TOP_K_ARTICLES = 2

# ── Article search ────────────────────────────────────────────────────────────
# SEARCH_PROVIDER picks how article TITLES are found; articles are always LOADED
# from Wikipedia afterwards, and any failure falls back to plain Wikipedia search.
#   "wiki_rewrite" — Wikipedia's search API, fed the query plus LLM rewrites of it
#                    into Wikipedia's own wording. Free, official, never blocked.
#   "ddgs"         — DuckDuckGo through the ddgs package.
#   "direct"       — DuckDuckGo through the hand-written client below.
SEARCH_PROVIDER = "wiki_rewrite"    # "wiki_rewrite" | "ddgs" | "direct"

# ── wiki_rewrite ──────────────────────────────────────────────────────────────
# WHY REWRITE. Wikipedia search matches WORDS, not meaning. Measured on its API
# with srlimit=20: 'most gigantic creature in history' returns Greek mythological
# creatures, Chimera, Kraken, Lake Placid (film) — and not one useful article in
# the top 20, because the article that answers ('Largest and heaviest animals')
# says "largest animal" and shares none of the claim's words. qwen3:8b rewriting
# the query into Wikipedia's wording closes that gap:
#
#   'Biggest animal ever lived'   -> #1 Largest and heaviest animals
#   'Largest creature in history' -> #1 Largest organisms
#
# WHAT IT DOES NOT FIX. When the answer sits in an article ABOUT something
# broader, rewording cannot reach it: 'earliest European permanent settlement
# United States' needs 'List of North American settlements by year of
# foundation', and no rewrite found it (best: European colonization of the
# Americas at #8). That needs meaning-based search (a search API), not wording.
#
# SKIPPED FOR NAMES. Every search passes through here, most of them plain names
# ('Nepal', 'Mount Everest'). The original query is searched first, and when its
# top hit IS the query the rewrite is skipped: ~9s of LLM time per query bought
# nothing for a name Wikipedia already matches exactly.
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
REWRITE_MODEL = "qwen3:8b"
REWRITE_COUNT = 3              # rewrites requested from the model
WIKI_SEARCH_LIMIT = 10         # titles per search
# Articles returned per query. More than TOP_K_ARTICLES because the merged title
# ranking is a rough cut: a bad rewrite can agree with the original query on an
# off-topic article, and the chunk scorer downstream is what separates them.
REWRITE_TOP_ARTICLES = 4

# ── DuckDuckGo ("ddgs" / "direct") ────────────────────────────────────────────
# Articles are FOUND with DuckDuckGo ("{query} site:wikipedia.org") and then
# LOADED from Wikipedia. Wikipedia's own search matches titles and words
# literally, so a category query like 'earliest European permanent settlement
# United States' landed on articles about one settlement, never on the list or
# overview article that names the record holder. A web search engine ranks by
# what pages are ABOUT and surfaces exactly those articles:
#
#   'tallest mountain Earth'  -> List of highest mountains on Earth,
#                                Tallest mountain, Mount Everest, ...
#
# WHY A HAND-WRITTEN CLIENT, NOT THE `ddgs` PACKAGE. ddgs posts to the same
# html.duckduckgo.com endpoint used here, but disguises every request as a
# RANDOM browser (primp, impersonate="random"). Some of those fingerprints are
# flagged as bots, and ddgs reports the resulting challenge page as "No results
# found" — indistinguishable from a genuinely empty search. Measured: 7 of 9
# ddgs searches in one live run failed that way. A fixed, ordinary browser
# identity from one reused session behaves like one person searching.
#
# BLOCKS ARE PER IP AND ARE REPORTED AS HTTP 202. After a burst of ~30 searches
# in a few minutes DuckDuckGo refused every request from this machine,
# whatever the headers. Retrying during a block only extends it, so a 202 is
# not retried: it opens a cooldown during which DuckDuckGo is skipped outright
# and every search goes straight to the Wikipedia-search fallback.
#
# The two DuckDuckGo clients:
#   "ddgs"   — the ddgs package: retried on failure, no block detection, since
#              it reports a block as "No results found".
#   "direct" — the hand-written client below: fixed browser identity, 202 read
#              as a block, cooldown, minimum interval.
# Both cache results per query and fall back to Wikipedia search on failure.
DDG_MAX_RESULTS = 10   # results requested from ddgs

DDG_URL = "https://html.duckduckgo.com/html/"
DDG_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://html.duckduckgo.com/",
}
DDG_TIMEOUT_SECONDS = 15

# Minimum gap between two DuckDuckGo requests. A claim fires several searches
# back to back (entity queries, predicate query, program fallbacks); spacing
# them is what keeps a normal run below the burst that triggers a block.
DDG_MIN_INTERVAL_SECONDS = 3.0

# How long DuckDuckGo is skipped after it answers with a block. Measured: a
# block from this machine lasted ~24 minutes (probed every 2 minutes), and the
# first requests after it lifted were blocked again by the third search, 3s
# apart. So the tolerated rate is a handful of searches, not a sustained stream,
# and a short cooldown would only re-trigger the block.
DDG_COOLDOWN_SECONDS = 1800

# Only English article pages. site:wikipedia.org also returns other-language
# editions (simple.wikipedia.org, tr.wikipedia.org, ...) and non-article pages.
WIKI_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org"}
WIKI_NON_ARTICLE_NAMESPACES = {
    "special", "file", "image", "category", "talk", "user", "user talk",
    "wikipedia", "help", "template", "portal", "draft", "module", "mediawiki",
}

# doc_content_chars_max: Hard character ceiling per fetched Wikipedia article.
#
# RAISED FROM 4000, WHICH WAS GUARDING A DOOR THAT WAS ALREADY LOCKED.
#
# The original reasoning was that uncapped retrieval would overflow an 8B
# model's context window. It cannot: TOP_K_CHUNKS decides how much text reaches
# the verifier, and that is two chunks — about 1000 characters — however long
# the article is. The cap sat UPSTREAM of a selection stage that already bounds
# the output, so the only thing it actually did was delete most of every source
# before anything had a chance to score it.
#
# And it deleted the same part every time. The ceiling is applied AT FETCH TIME,
# before the query is consulted, so every query against one article receives an
# identical opening window:
#
#   "The blue whale is the largest animal ever known to have lived."
#       query 'largest animal'                    -> answered from the window
#   "The African bush elephant is the largest living land animal on Earth."
#       query 'largest terrestrial land animal'   -> NOT ENOUGH INFO
#
# Both facts retrieved 'Largest and heaviest animals'. Both got its first 4000
# characters, which are about the whale. The sentence naming the elephant sits
# further down the same article and was never fetched, so it could not be
# chunked, could not be scored, and could not be selected. Retrieval restarts
# per claim — correctly — but restarting a search over a truncated corpus finds
# the same nothing each time.
#
# 20000 characters covers the body of most articles. The cost is more Python
# string work in chunk_articles and score_chunks: no extra network request (the
# API returns the extract regardless), and not one additional token to a model.
#
# TWO CONSEQUENCES WORTH WATCHING. Five times as many chunks means the IDF
# document frequencies are computed over a larger pool, so scores shift and a
# top-2 chosen from 150 candidates is not the top-2 chosen from 30 — better
# estimates, but not the same rankings. And the wider net makes an off-topic
# chunk reachable that previously was not. select_top_chunks' rarest-token
# guarantee is what holds the line there.
DOC_CONTENT_CHARS_MAX = 20000

# ── Chunk config ──────────────────────────────────────────────────────────────
# CHUNK_SIZE: Target character length per semantic chunk.
# Empirically, 400-600 characters captures one coherent topic unit from
# Wikipedia prose without splitting mid-argument or mid-sentence.
CHUNK_SIZE = 500

# OVERLAP_SENTENCES: Number of trailing sentences repeated at the start of
# the next chunk. Sentence-level overlap replaces the old 50-character
# overlap: same purpose (evidence straddling a chunk boundary is never
# lost) without ever cutting a sentence in half.
OVERLAP_SENTENCES = 1

# TOP_K_CHUNKS: Maximum chunks returned to verifier.py per atomic fact.
# AFEV found that 1-2 evidence pieces per atomic fact yields optimal
# verification accuracy.
#
# THIS IS THE CONSTANT THAT PROTECTS THE CONTEXT WINDOW — not
# DOC_CONTENT_CHARS_MAX above, which was raised precisely because it was
# believed to be doing this job. Two chunks of ~500 characters is what the
# verifier ever sees, so the article ceiling can be as generous as chunking
# time allows without a single extra token reaching the model.
TOP_K_CHUNKS = 2


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Wikipedia Article Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def fetch_articles(query: str) -> list[tuple[str, str]]:
    """
    Wikipedia articles for a search query, each kept with its canonical URL.

    Titles come from the search SEARCH_PROVIDER selects; the articles are then
    loaded from Wikipedia. Falls back to plain Wikipedia search
    (_fetch_via_wikipedia_search) when the provider yields no loadable article.

    Why keep the URL?
    The citation shown to the user must point at the article the evidence
    actually came from. Resolving a URL separately from the NER query (as
    postprocessor previously did) could cite an article the verifier never
    saw — and in practice always fell back to Special:Search.

    Args:
        query : clean search string produced by ner.extract_queries()

    Returns:
        List of (article_content, article_url) tuples.
        Empty list if retrieval fails or query yields no results.
    """
    if SEARCH_PROVIDER == "wiki_rewrite":
        print(f"[Retriever] Searching Wikipedia (with rewrites) for: '{query}'")
        titles, limit = _search_wiki_rewrite(query), REWRITE_TOP_ARTICLES
    else:
        print(f"[Retriever] Searching DuckDuckGo for: '{query} site:wikipedia.org'")
        titles, limit = _search_duckduckgo(query), TOP_K_ARTICLES

    articles = []
    for title in titles:
        article = _load_article(title)
        if article is not None:
            articles.append(article)
        if len(articles) == limit:
            break

    if articles:
        print(f"[Retriever] Retrieved {len(articles)} article(s): "
              f"{', '.join(url for _, url in articles)}")
        return articles

    print(f"[Retriever] {SEARCH_PROVIDER} found no usable article — falling back "
          f"to Wikipedia search.")
    return _fetch_via_wikipedia_search(query)


_wiki_session = requests.Session()
_wiki_session.headers.update({"User-Agent": WIKI_USER_AGENT})
_rewrite_cache: dict[str, list[str]] = {}


def _search_wiki_rewrite(query: str) -> list[str]:
    """
    Article titles for `query`: Wikipedia search over the query itself plus its
    LLM rewrites, merged by Reciprocal Rank Fusion.

    Fusing by rank rather than concatenating: an article that several phrasings
    rank highly beats one that a single phrasing ranks first, which damps the
    one rewrite that goes astray ('Gigantic creature from history' -> Greek
    mythological creatures).
    """
    original = _wiki_search(query)
    if original and _normalise_title(original[0]) == _normalise_title(query):
        print(f"[Retriever] '{query}' names an article — no rewrite needed.")
        return original

    rankings = [original] + [_wiki_search(r) for r in _rewrite_query(query)]
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, title in enumerate(ranking, 1):
            scores[title] = scores.get(title, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores, key=scores.get, reverse=True)


def _normalise_title(text: str) -> str:
    return re.sub(r"[\s_]+", " ", text).strip().lower()


def _wiki_search(query: str) -> list[str]:
    """Article titles from Wikipedia's search API (list=search), best first."""
    params = {"action": "query", "list": "search", "srsearch": query,
              "srnamespace": 0, "srlimit": WIKI_SEARCH_LIMIT, "format": "json"}
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = _wiki_session.get(WIKI_API_URL, params=params, timeout=15)
            response.raise_for_status()
            return [hit["title"] for hit in response.json()["query"]["search"]]
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"[Retriever] Wikipedia search attempt {attempt}/{RETRY_ATTEMPTS} "
                  f"for '{query}' failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return []


_REWRITE_PROMPT = """Rewrite this search query as {n} short Wikipedia search queries. Use the plain, common wording a Wikipedia article title or first sentence would use (for example, prefer "highest" over "loftiest" and "city" over "metropolis"). One query per line, nothing else.

Query: {query}"""

_LIST_MARKER = re.compile(r'^\s*(?:\d+[.)]|[-*•])\s*')


def _rewrite_query(query: str) -> list[str]:
    """
    Up to REWRITE_COUNT rephrasings of `query` in Wikipedia's wording, from
    REWRITE_MODEL. Empty if Ollama cannot be reached — the original query's own
    results still stand, so a missing rewrite costs quality, never the claim.
    """
    if query in _rewrite_cache:
        return _rewrite_cache[query]

    payload = {
        "model": REWRITE_MODEL,
        "prompt": _REWRITE_PROMPT.format(n=REWRITE_COUNT, query=query),
        "stream": False,
        # Top-level field, NOT an option. The answer is three short lines; a
        # <think> block would spend the whole num_predict budget before them.
        "think": False,
        "options": {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 0,
                    "num_predict": 80},
    }
    try:
        response = requests.post(f"{config.OLLAMA_BASE_URL}/api/generate",
                                 json=payload, timeout=180)
        response.raise_for_status()
        raw = response.json()["response"]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"[Retriever] Query rewrite failed ({e}) — searching the original only.")
        return []

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    rewrites = []
    for line in raw.splitlines():
        line = _LIST_MARKER.sub('', line).strip().strip('"\'')
        if line and len(line.split()) <= 12 and \
                _normalise_title(line) != _normalise_title(query) and line not in rewrites:
            rewrites.append(line)
    rewrites = rewrites[:REWRITE_COUNT]

    print(f"[Retriever] Rewrote '{query}' -> {rewrites}")
    _rewrite_cache[query] = rewrites
    return rewrites


_ddg_session = requests.Session()
_ddg_session.headers.update(DDG_HEADERS)
_ddg_lock = threading.Lock()
_ddg_last_request = 0.0     # time.monotonic() of the last request sent
_ddg_blocked_until = 0.0    # time.monotonic() before which DuckDuckGo is skipped
_ddg_cache: dict[str, list[str]] = {}


def _search_duckduckgo(query: str) -> list[str]:
    """
    English Wikipedia article titles for `query`, in DuckDuckGo's rank order,
    from the client SEARCH_PROVIDER selects ("ddgs" or "direct"). Empty on failure — fetch_articles then
    falls back to Wikipedia search. Only a real answer is cached, so a failure
    is never remembered as "no results".
    """
    if query in _ddg_cache:
        return _ddg_cache[query]
    if SEARCH_PROVIDER == "ddgs":
        return _search_ddgs(query)
    return _search_direct(query)


def _search_ddgs(query: str) -> list[str]:
    """The search through the ddgs package, retried with backoff on failure."""
    results = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            results = DDGS().text(f"{query} site:wikipedia.org",
                                  max_results=DDG_MAX_RESULTS,
                                  backend="duckduckgo")
            break
        except DDGSException as e:
            # Either a genuinely empty search or a block — ddgs reports both as
            # "No results found", so the two cannot be told apart here.
            print(f"[Retriever] ddgs attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if results is None:
        return []

    titles = []
    for result in results:
        title = wikipedia_title(result.get("href", ""))
        if title and title not in titles:
            titles.append(title)
    _ddg_cache[query] = titles
    return titles


def _search_direct(query: str) -> list[str]:
    """
    The search through the hand-written client. Empty when DuckDuckGo is
    cooling down after a block, blocks this request, or cannot be reached.
    """
    global _ddg_last_request, _ddg_blocked_until

    with _ddg_lock:
        now = time.monotonic()
        if now < _ddg_blocked_until:
            print(f"[Retriever] DuckDuckGo cooling down after a block "
                  f"({_ddg_blocked_until - now:.0f}s left) — skipping.")
            return []

        wait = _ddg_last_request + DDG_MIN_INTERVAL_SECONDS - now
        if wait > 0:
            time.sleep(wait)

        try:
            response = _ddg_session.post(DDG_URL,
                                         data={"q": f"{query} site:wikipedia.org"},
                                         timeout=DDG_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            print(f"[Retriever] DuckDuckGo unreachable: {e}")
            return []
        finally:
            _ddg_last_request = time.monotonic()

        if response.status_code == 202:
            _ddg_blocked_until = time.monotonic() + DDG_COOLDOWN_SECONDS
            print(f"[Retriever] DuckDuckGo blocked this machine (HTTP 202) — "
                  f"skipping it for {DDG_COOLDOWN_SECONDS}s.")
            return []
        if response.status_code != 200:
            print(f"[Retriever] DuckDuckGo answered HTTP {response.status_code}.")
            return []

    titles = []
    for url in _ddg_result_urls(response.text):
        title = wikipedia_title(url)
        if title and title not in titles:
            titles.append(title)

    _ddg_cache[query] = titles
    return titles


_DDG_RESULT_LINK = re.compile(r'<a\b[^>]*\bclass="result__a"[^>]*>', re.I)
_HREF = re.compile(r'\bhref="([^"]+)"', re.I)


def _ddg_result_urls(page: str) -> list[str]:
    """
    Destination URLs of the organic results on an html.duckduckgo.com page.

    A result link is either the destination itself or a DuckDuckGo redirect
    ('//duckduckgo.com/l/?uddg=<encoded url>&rut=...'); both are unwrapped to
    the destination. Ads link to duckduckgo.com/y.js and are dropped.
    """
    urls = []
    for anchor in _DDG_RESULT_LINK.findall(page):
        match = _HREF.search(anchor)
        if not match:
            continue
        href = html.unescape(match.group(1))
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if parsed.netloc.endswith("duckduckgo.com"):
            target = parse_qs(parsed.query).get("uddg")
            if not target:
                continue
            href = target[0]
        urls.append(href)
    return urls


def wikipedia_title(url: str) -> str | None:
    """
    'https://en.wikipedia.org/wiki/Mount_Everest#Climbing' -> 'Mount Everest'

    None for other-language editions and for non-article pages (Talk:,
    Category:, File:, ...). Article titles may themselves contain a colon
    ('Star Wars: Episode IV'), so only a KNOWN namespace prefix is rejected.
    """
    parsed = urlparse(url)
    if parsed.netloc.lower() not in WIKI_HOSTS or not parsed.path.startswith("/wiki/"):
        return None
    title = unquote(parsed.path[len("/wiki/"):]).replace("_", " ").strip()
    if not title:
        return None
    namespace, sep, _ = title.partition(":")
    if sep and namespace.strip().lower() in WIKI_NON_ARTICLE_NAMESPACES:
        return None
    return title


# Article content cache, keyed by normalised title, for the lifetime of the
# process. A single /check can resolve the same title several times — one run
# searched 'Jupiter', its fallback searched 'Jupiter' again, and sentence 2
# searched it a third time, downloading, lxml-parsing and re-chunking the same
# four articles each round. Multi-query retrieval compounds it: several rewrites
# routinely rank the same article. Caching the (content, url) a title resolves
# to turns every repeat into a dict hit. A definitive miss (disambiguation,
# missing page, empty body) is cached too, so a known-bad title is not retried;
# a TRANSIENT failure (network/timeout after retries) is NOT cached, so it can
# succeed on a later attempt.
_article_cache: dict[str, tuple[str, str] | None] = {}


def _load_article(title: str) -> tuple[str, str] | None:
    """(content, url) for one exact title — cached for the process lifetime."""
    key = _normalise_title(title)
    if key in _article_cache:
        cached = _article_cache[key]
        print(f"[Retriever] '{title}' served from article cache"
              + ("" if cached else " (known miss)") + ".")
        return cached

    result, cacheable = _load_article_uncached(title)
    if cacheable:
        _article_cache[key] = result
    return result


def _load_article_uncached(title: str) -> tuple[tuple[str, str] | None, bool]:
    """
    Returns (result, cacheable).

    cacheable is True for a PERMANENT outcome — a loaded article, or a title
    that is definitively not loadable (disambiguation, missing page, empty
    body). It is False only when every attempt failed transiently, so the
    caller leaves that title uncached and a later call may still succeed.
    """
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            page = wikipedia.page(title, auto_suggest=False, redirect=True)
            content = page.content[:DOC_CONTENT_CHARS_MAX]
            return ((content, page.url), True) if content.strip() else (None, True)
        except (wikipedia.exceptions.DisambiguationError,
                wikipedia.exceptions.PageError) as e:
            # Permanent: this title is not a loadable article. Skip it.
            print(f"[Retriever] Skipping '{title}': {type(e).__name__}")
            return None, True
        except Exception as e:
            print(f"[Retriever] Loading '{title}' attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return None, False


def _fetch_via_wikipedia_search(query: str) -> list[tuple[str, str]]:
    """The original search path: Wikipedia's own search via LangChain's
    WikipediaRetriever. Now only the fallback for a failed DuckDuckGo search."""
    print(f"[Retriever] Fetching Wikipedia articles for query: '{query}'")

    retriever = WikipediaRetriever(
        top_k_results=TOP_K_ARTICLES,
        doc_content_chars_max=DOC_CONTENT_CHARS_MAX
    )

    docs = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            docs = retriever.invoke(query)
            break
        except Exception as e:
            # Typically a transient block/rate-limit (HTML error page →
            # JSON parse failure) or a network hiccup. Back off and retry;
            # a permanent failure must not crash the pipeline — return
            # empty and let verifier yield NOT ENOUGH INFO.
            print(f"[Retriever] Wikipedia fetch attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    if docs is None:
        print(f"[Retriever] All fetch attempts failed for query '{query}'.")
        return []

    articles = [
        (doc.page_content, doc.metadata.get("source", ""))
        for doc in docs
        if doc.page_content.strip()
    ]

    print(f"[Retriever] Retrieved {len(articles)} article(s).")
    return articles


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Article Chunking
# ─────────────────────────────────────────────────────────────────────────────
def _split_sentences(text: str) -> list[str]:
    """
    Lightweight deterministic sentence splitter.

    Splits after sentence-final punctuation (. ! ?) when followed by
    whitespace and an uppercase letter, digit, or opening quote/paren.
    Not perfect on abbreviations, but adequate for Wikipedia prose and
    dependency-free (no spaCy inference per article).
    """
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(“])', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_articles(articles: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Splits Wikipedia articles into sentence-aware chunks of ~CHUNK_SIZE
    characters, tagging every chunk with its article URL so the final
    citation always matches the evidence actually used.

    Why sentence-aware instead of fixed character windows?
    Character windows cut mid-sentence ("...built the Statue of Liberty
    in New Yor"), destroying exactly the evidence span the verifier needs
    and rendering unreadable evidence in the UI. Packing whole sentences
    up to the size budget keeps every chunk self-contained. A chunk may
    slightly exceed CHUNK_SIZE when a single sentence is longer than the
    budget — emitted whole rather than split.

    OVERLAP_SENTENCES repeats the last sentence(s) of each chunk at the
    start of the next so boundary evidence is never lost.

    Args:
        articles : list of (article_content, article_url) tuples

    Returns:
        Flat list of (chunk, article_url) tuples across all articles.
        Empty list if no articles were provided.
    """
    if not articles:
        print("[Retriever] No articles to chunk.")
        return []

    chunks = []
    for article, url in articles:
        # Normalize whitespace so sentence boundaries are detectable.
        clean     = re.sub(r'\s+', ' ', article).strip()
        sentences = _split_sentences(clean)

        current: list[str] = []
        current_len = 0

        for sent in sentences:
            # Flush the current chunk if adding this sentence would exceed
            # the budget.
            if current and current_len + len(sent) + 1 > CHUNK_SIZE:
                chunks.append((" ".join(current), url))
                # Sentence-level overlap into the next chunk
                current = current[-OVERLAP_SENTENCES:] if OVERLAP_SENTENCES > 0 else []
                current_len = sum(len(s) + 1 for s in current)

            current.append(sent)
            current_len += len(sent) + 1

        if current:
            chunks.append((" ".join(current), url))

    print(f"[Retriever] Generated {len(chunks)} chunk(s) across {len(articles)} article(s).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Lexical Chunk Scoring
# ─────────────────────────────────────────────────────────────────────────────
# Named "Semantic" until it was noticed that nothing here computes meaning.
# This is term matching weighted by rarity — a chunk saying "particle
# accelerator" scores zero against a claim saying "particle collider", because
# the strings differ. Calling it semantic invites a reader to assume embedding
# machinery exists somewhere in the project. It does not.

# ── Ranking parameters ────────────────────────────────────────────────────────
# MIN_SCORE_TOKEN_LENGTH: tokens this short are articles and prepositions that
# add noise without meaning. Previously the inline `len(t) > 2`.
MIN_SCORE_TOKEN_LENGTH = 3

# USE_IDF_WEIGHTING: weight each matched token by how RARE it is across the
# chunks retrieved for this claim, instead of counting every token equally.
#
# The unweighted version treated 'bosporus' and 'europe' as worth the same,
# which is how a passage about African rainfall scored 0.42 against a claim
# about a strait — it matched 'africa', 'europe' and 'between' while missing the
# only word that identified the subject. Rare terms are what distinguish a
# relevant passage; common ones are satisfied by almost anything.
#
# The document frequencies come from the ~30-90 chunks just retrieved, not a
# global corpus. That is deliberate: it needs no index, no download and no
# dependency, and it is arguably better suited to the task — a term appearing in
# every chunk fetched for THIS claim is uninformative for choosing between them,
# whatever its frequency in English at large.
#
# KNOWN BIAS, worth stating plainly because it shapes what this can and cannot
# do: the rarest token in a claim is almost always its SUBJECT NAME, so chunks
# mentioning the subject dominate. Evidence that would REFUTE "X was first" is
# typically a passage about someone ELSE, which by construction does not contain
# X. Rarity weighting therefore sharpens relevance and does nothing for
# counter-evidence — arguably it makes that harder. Fixing it needs diversity in
# selection or a differently-built query, not a different weighting.
#
# Set False to restore plain term coverage, which is the point of comparison
# when measuring whether this helped.
USE_IDF_WEIGHTING = True


def _tokenize(text: str) -> set[str]:
    """Lowercase content tokens used by the ranker."""
    return {
        t for t in re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        if len(t) >= MIN_SCORE_TOKEN_LENGTH
    }


def score_chunks(fact: str, chunks: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
    """
    Scores each chunk against the atomic fact by IDF-weighted term coverage.

    Scoring formula:
        score = Σ idf(t) for t in (fact ∩ chunk)  /  Σ idf(t) for t in fact

    where idf(t) = log(1 + N / (1 + df(t))), N = number of chunks retrieved for
    this claim and df(t) = how many of them contain t.

    The +1 inside the log keeps the weight strictly positive: a term present in
    every retrieved chunk would otherwise score log(1) = 0, drop out of the
    denominator entirely, and can make it zero.

    The denominator keeps the score in [0, 1] and preserves its meaning as "how
    much of the claim this chunk covers", so the printed Top chunk score stays
    comparable in magnitude to earlier runs — but a chunk now has to cover the
    claim's DISTINCTIVE words to score highly, not merely three common ones.

    Args:
        fact   : atomic claim string from claim_extractor
        chunks : list of (chunk, article_url) tuples from chunk_articles()

    Returns:
        List of (score, chunk, article_url) tuples sorted by descending score.
        Empty list if no chunks provided.
    """
    if not chunks:
        print("[Retriever] No chunks available to score.")
        return []

    fact_tokens = _tokenize(fact)

    if not fact_tokens:
        # If the fact contains no meaningful tokens after filtering,
        # return all chunks unscored with equal weight of 0.0
        print("[Retriever] No scoreable tokens found in fact. Returning unscored chunks.")
        return [(0.0, chunk, url) for chunk, url in chunks]

    chunk_token_sets = [_tokenize(chunk) for chunk, _ in chunks]

    if USE_IDF_WEIGHTING:
        total = len(chunk_token_sets)
        df = Counter(t for tokens in chunk_token_sets for t in tokens)
        weight = {t: math.log(1 + total / (1 + df.get(t, 0))) for t in fact_tokens}
    else:
        weight = {t: 1.0 for t in fact_tokens}

    denominator = sum(weight[t] for t in fact_tokens) or 1.0

    scored = []
    # strict=True: a length mismatch between chunks and their token sets would
    # silently truncate under plain zip, pairing a chunk with another chunk's
    # tokens and scoring both wrongly.
    for (chunk, url), chunk_tokens in zip(chunks, chunk_token_sets, strict=True):
        matched = fact_tokens & chunk_tokens
        score = sum(weight[t] for t in matched) / denominator
        scored.append((score, chunk, url))

    # Sort descending: highest relevance chunks surface to the top
    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        mode = "IDF-weighted" if USE_IDF_WEIGHTING else "term-coverage"
        print(f"[Retriever] Top chunk score: {scored[0][0]:.2f} ({mode})")

    return scored


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3b — Dense & Hybrid Chunk Scoring
# ─────────────────────────────────────────────────────────────────────────────
# score_chunks above is exact-token overlap: a chunk saying "particle
# accelerator" scores 0 against a claim saying "particle collider", so a
# paraphrase of the evidence is unreachable and the claim wrongly returns NOT
# ENOUGH INFO. Dense scoring embeds claim and chunks into a vector space where
# meaning, not spelling, decides nearness, and hybrid fuses the two so a chunk
# must satisfy BOTH — dense catches paraphrase, IDF keeps the subject anchor
# that dense alone drifts off (a passage merely "about whales" is close to a
# whale claim without settling it).
#
# NO GENERATIVE LLM RUNS HERE, DELIBERATELY. Embeddings from a fixed model are
# deterministic and the scoring path never samples, so the determinism the rest
# of the pipeline is careful about (Tests/determinism_test.py) is preserved. Do
# not add a sampled or generative step to scoring.

def embed_texts(texts: list[str], *, is_query: bool) -> list[list[float]]:
    """
    Embed a batch of texts with EMBED_MODEL via Ollama's /api/embed.

    nomic-embed-text is trained with TASK PREFIXES and retrieval quality drops
    measurably without them: the claim is a 'search_query', the chunks are
    'search_document'. The prefixes are prepended here, not by the caller, so
    the distinction lives in one place.

    Returns one vector per input text, in input order. Raises on HTTP or model
    error — callers that must degrade rather than fail catch it (see
    score_chunks_hybrid).
    """
    prefix = "search_query: " if is_query else "search_document: "
    payload = {"model": config.EMBED_MODEL, "input": [prefix + t for t in texts]}
    r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity — no numpy. Zero for a zero vector."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _dense_rerank(fact: str,
                  lex: list[tuple[float, str, str]]) -> list[tuple[float, str, str]]:
    """
    Embed and cosine-rank ONLY the top-DENSE_CANDIDATES lexical chunks.

    `lex` is the full lexical ranking (from score_chunks), already sorted. The
    pool is its head, so the one query vector plus at most DENSE_CANDIDATES chunk
    vectors are the only embeddings computed — the whole point of the rerank
    split. Raises on embedding failure; callers decide whether that is fatal.
    """
    pool = [(chunk, url) for _, chunk, url in lex[:DENSE_CANDIDATES]]
    if not pool:
        return []

    q_vec  = embed_texts([fact], is_query=True)[0]
    c_vecs = embed_texts([c for c, _ in pool], is_query=False)

    scored = [
        (_cosine(q_vec, cv), chunk, url)
        # strict=True: a length mismatch between the pool and its vectors would
        # otherwise pair a chunk with another chunk's embedding, silently, and
        # score both wrongly — the same guard score_chunks uses.
        for (chunk, url), cv in zip(pool, c_vecs, strict=True)
    ]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored


def _with_lexical_tail(ranked: list[tuple[float, str, str]],
                       lex: list[tuple[float, str, str]]
                       ) -> list[tuple[float, str, str]]:
    """
    Append the chunks NOT in the reranked pool, in lexical order, scored 0.0.

    select_top_chunks picks the top TOP_K_CHUNKS and then, for the rarest-token
    coverage guarantee, searches the REST for a chunk containing that token.
    With only the embedded pool returned, that search could no longer reach a
    needle chunk that fell outside the pool. The tail restores its full reach at
    zero embedding cost: every RRF/cosine score is > 0, so a 0.0 tail always
    sorts below the real candidates and never displaces one — it is reachable by
    the swap, never selected on its own merit.
    """
    pooled = {chunk for _, chunk, _ in ranked}
    tail = [(0.0, chunk, url) for _, chunk, url in lex if chunk not in pooled]
    return ranked + tail


def score_chunks_dense(fact: str,
                       chunks: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
    """
    Rank chunks by embedding cosine similarity to the claim, reranking only the
    top-DENSE_CANDIDATES lexical candidates (see _dense_rerank, DENSE_CANDIDATES).

    Same (score, chunk, url) shape as score_chunks, sorted descending, with the
    un-embedded remainder appended at score 0.0 so the coverage swap keeps its
    reach (see _with_lexical_tail). Degrades to the pure-lexical ranking if the
    embedding endpoint is unavailable — retrieval must never crash the pipeline.
    """
    if not chunks:
        return []

    lex = score_chunks(fact, chunks)      # cheap over all chunks; prints its line
    try:
        den = _dense_rerank(fact, lex)
    except Exception as e:
        print(f"[Retriever] Dense scoring failed ({e}); falling back to IDF.")
        return lex

    if not den:
        return lex

    print(f"[Retriever] Top chunk score: {den[0][0]:.2f} (dense cosine, "
          f"reranked {len(den)} of {len(chunks)})")
    return _with_lexical_tail(den, lex)


def score_chunks_hybrid(fact: str,
                        chunks: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
    """
    Fuse the lexical and dense rankings with Reciprocal Rank Fusion.

        rrf(chunk) = 1 / (RRF_K + lex_rank) + 1 / (RRF_K + dense_rank)

    Same (score, chunk, url) shape as score_chunks, sorted descending — but the
    score is an RRF fusion score, NOT the [0,1] coverage fraction score_chunks
    prints. Any threshold tuned against that fraction is meaningless here, which
    is why the lexical scorer still runs first and still prints its own "Top
    chunk score" line: the coverage number a downstream reader might depend on
    is preserved even under hybrid.

    DEGRADES TO LEXICAL, NEVER CRASHES. If the embedding endpoint or model is
    unavailable, dense scoring raises and this returns the pure-lexical ranking
    rather than failing the claim — retrieval failure must always fall back to
    fewer/worse evidence, never an exception (the verifier then yields NOT
    ENOUGH INFO gracefully).
    """
    if not chunks:
        return []

    lex = score_chunks(fact, chunks)      # also prints the lexical coverage line
    try:
        den = _dense_rerank(fact, lex)    # embeds only the top DENSE_CANDIDATES
    except Exception as e:
        print(f"[Retriever] Dense scoring failed ({e}); falling back to IDF.")
        return lex

    if not den:
        return lex

    # Rank position (0 = best). lex ranks is over ALL chunks (so a chunk's
    # lexical standing is its true one); den ranks is over the embedded pool.
    # Fusion is WITHIN the pool — a chunk must be a lexical candidate to be
    # reranked and fused — and the rest trails behind via _with_lexical_tail.
    # Built from the last occurrence, so identical chunk texts (rare, from
    # sentence overlap) collapse to one rank consistently in both maps.
    lex_rank = {chunk: i for i, (_, chunk, _) in enumerate(lex)}
    den_rank = {chunk: i for i, (_, chunk, _) in enumerate(den)}

    fused = [
        (1.0 / (RRF_K + lex_rank[chunk]) + 1.0 / (RRF_K + den_rank[chunk]), chunk, url)
        for _, chunk, url in den
    ]
    fused.sort(key=lambda x: (-x[0], x[1]))

    print(f"[Retriever] Top chunk RRF score: {fused[0][0]:.4f} "
          f"(hybrid: IDF + dense, reranked {len(den)} of {len(chunks)} chunks)")
    return _with_lexical_tail(fused, lex)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Top-K Chunk Selection
# ─────────────────────────────────────────────────────────────────────────────
def rarest_fact_token(fact: str, chunks: list[tuple[str, str]]) -> str | None:
    """
    The claim token that narrows the search most, provided some chunk has it.

    THE TOKEN THE EVIDENCE MUST NOT OMIT. Scoring treats tokens independently,
    so a claim's decisive word can lose to several cheap ones. Observed:

        claim     "The Empire State Building was completed in 1931."
        selected  "...the tenth-tallest COMPLETED skyscraper in the United
                   States, and the 59th-tallest COMPLETED skyscraper in the
                   world. The site... was developed in 1893... In 1929, Empire
                   State Inc. acquired the site..."
        verdict   NOT ENOUGH INFO

    'completed' twice, '1931' never. The winning chunk matched a moderately
    rare word used as an adjective about OTHER buildings, while the chunk
    carrying the date — the only token that could settle the claim — ranked
    below it and was cut. The verifier answered correctly about evidence it
    should never have been given.

    Rarity is measured over the retrieved pool, the same document frequencies
    score_chunks uses, so 'the' and 'building' weigh nothing and a year weighs
    heavily. Tokens absent from every chunk are excluded: a guarantee that
    cannot be satisfied is not a guarantee, and insisting on a word nobody wrote
    would discard the top chunk for nothing.

    THE TIE-BREAK IS LOAD-BEARING, AND ITS ABSENCE MADE THIS FUNCTION RETURN A
    DIFFERENT ANSWER FROM RUN TO RUN ON IDENTICAL INPUT.

    _tokenize returns a SET, `present` inherits that set's iteration order, and
    max() returns the FIRST maximal element in the order it is given. Python
    randomises string hashing per process by default (PYTHONHASHSEED), so set
    iteration order — and therefore the winner of any exact tie — changed on
    every invocation.

    Measured on the fixture in Tests/retriever_test.py:

        1931      df=1  idf=0.916291
        was       df=1  idf=0.916291     <- an exact tie, not a near one
        empire    df=3  idf=0.559616
        state     df=3  idf=0.559616
        building  df=3  idf=0.559616
        the       df=3  idf=0.559616

    and across 24 PYTHONHASHSEED values: 12 runs returned '1931', 12 returned
    'was'. A 50% failure rate, invisible in any single run and unrelated to any
    edit made near it — which is the worst way for a defect to present, because
    it lands on whoever happens to run the suite next.

    'was' is an auxiliary verb that survives _tokenize only because
    MIN_SCORE_TOKEN_LENGTH is 3 and 'was' is exactly three characters. It
    reached idf parity with the year because df is measured over the three
    chunks of a test fixture, where it happens to appear in one of them. In a
    real 30-90 chunk pool it would appear in most and sink on its own.

    So the key is now a TOTAL ORDER, and each component is a claim about what
    makes a token decisive:

        idf         rarity within the retrieved pool — unchanged, still dominant
        isdigit     a year or quantity settles a claim; a function word does not
        len         between two words of equal rarity, the longer carries more
        token       alphabetical, so the order is total and the result is
                    reproducible whatever the hash seed

    Only exact ties are affected. Any pair with different document frequencies
    has different idf, so the first component still decides and no existing
    ranking moves.

    WHAT THIS DOES NOT FIX, stated so it is not mistaken for a full repair: the
    tie-break rescues equal-idf cases only. Had '1931' appeared in two chunks
    and 'was' in one, 'was' would win outright on rarity and no tie-break would
    run. Excluding auxiliaries and copulas from _tokenize would close that, but
    it is a lexicon with a wider blast radius — it changes score_chunks' document
    frequencies too, and with them the regression baseline — so it is a separate
    decision, not a line to slip in here.
    """
    fact_tokens = _tokenize(fact)
    if not fact_tokens or not chunks:
        return None

    chunk_token_sets = [_tokenize(chunk) for chunk, _ in chunks]
    total = len(chunk_token_sets)
    df = Counter(t for tokens in chunk_token_sets for t in tokens)

    present = [t for t in fact_tokens if df.get(t, 0) > 0]
    if not present:
        return None

    return max(present, key=lambda t: (
        math.log(1 + total / (1 + df[t])),
        t.isdigit(),
        len(t),
        t,
    ))


def select_top_chunks(scored_chunks: list[tuple[float, str, str]],
                      must_contain: str | None = None,
                      top_k: int = TOP_K_CHUNKS) -> tuple[list[str], str]:
    """
    Selects the top-k highest scoring chunks for delivery to verifier.py,
    plus the URL of the best-scoring chunk's article — used as the citation.

    Why cap at TOP_K_CHUNKS (3)?
    AFEV paper Section 5.5 Figure 5(a) demonstrates that verification
    accuracy peaks at k=1-2 evidence pieces per atomic fact. Beyond k=3,
    noise from lower-relevance chunks begins to degrade reasoning model
    performance. We use k=3 as a slight buffer to account for cases where
    the top chunk is partially relevant but not sufficient alone.

    Args:
        scored_chunks : list of (score, chunk, article_url) tuples sorted descending

    Returns:
        (list of up to TOP_K_CHUNKS chunk strings, best_chunk_article_url)
        ([], "") if no scored chunks provided.
    """
    if not scored_chunks:
        print("[Retriever] No scored chunks to select from.")
        return [], ""

    selected = list(scored_chunks[:top_k])

    # COVERAGE GUARANTEE for the claim's decisive token — see rarest_fact_token.
    #
    # Swaps the WEAKEST selection, never the strongest: the top chunk is what
    # establishes the claim's subject and the citation URL is taken from it, so
    # trading it away to chase one word would fix the evidence and break the
    # source. And only when at least two slots exist, since replacing the sole
    # chunk is not a swap, it is a different search.
    if must_contain and len(selected) > 1:
        needle = must_contain.lower()
        if not any(needle in chunk.lower() for _, chunk, _ in selected):
            replacement = next(
                (row for row in scored_chunks[top_k:]
                 if needle in row[1].lower()),
                None,
            )
            if replacement is not None:
                dropped = selected[-1]
                selected[-1] = replacement
                print(f"[Retriever] No selected chunk contained '{must_contain}', "
                      f"the claim's rarest term — swapped the weakest selection "
                      f"({dropped[0]:.2f}) for one that does ({replacement[0]:.2f}).")
            else:
                print(f"[Retriever] '{must_contain}' is the claim's rarest term and "
                      f"appears in no retrieved chunk at all.")

    top      = [chunk for _, chunk, _ in selected]
    best_url = selected[0][2]

    print(f"[Retriever] Selected {len(top)} chunk(s) for verifier.")
    return top, best_url


def rank_chunks(text: str, chunks: list[tuple[str, str]],
                top_k: int = TOP_K_CHUNKS) -> tuple[list[str], str]:
    """
    The best `top_k` chunks for `text`, plus the citation URL of the best one.

    THE ONE PLACE CHUNKS ARE RANKED. Both evidence paths call this: fetch() for
    the verifier, and program._retrieve for the superlative and comparative
    programs. program._retrieve used to call score_chunks directly, so when
    config.RETRIEVAL_SCORING moved from "idf" to "hybrid" only the verifier path
    followed — the programs kept ranking by exact word overlap alone. Measured
    on 'most gigantic creature history': the category search retrieved
    'Largest and heaviest animals', but IDF ranked chunks of 'List of Greek
    mythological creatures' above it on the single word 'creature', the answer
    model read about a mythical boar, and the passage saying the blue whale is
    "the largest animal known to have ever existed" never reached it.

    Scorer selected by config.RETRIEVAL_SCORING; dense/hybrid return the SAME
    (score, chunk, url) shape, so the selection below is identical for all.
    """
    mode = config.RETRIEVAL_SCORING
    if mode == "dense":
        scored = score_chunks_dense(text, chunks)
    elif mode == "hybrid":
        scored = score_chunks_hybrid(text, chunks)
    else:
        scored = score_chunks(text, chunks)

    # The rarest-token coverage guarantee is a LEXICAL invariant and survives
    # regardless of scorer — it must always be passed. See select_top_chunks.
    return select_top_chunks(scored, must_contain=rarest_fact_token(text, chunks),
                             top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# Called by main.py orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def fetch(fact: str) -> dict:
    """
    Executes the full retrieval pipeline for a single atomic fact.
    Orchestrates NER query extraction, Wikipedia article fetching,
    chunking, scoring, and top-k selection into a single callable.

    Called by verification.verify_claim() inside the AFEV loop.

    Execution Flow:
        [Atomic Fact]
            → ner.extract_queries()        # combined + per-entity queries
            → fetch_articles(q) per query  # Wikipedia API → (text, url) pairs,
                                           #   deduplicated across queries
            → chunk_articles(articles)     # sentence-aware → url-tagged chunks
            → rank_chunks(fact, chunks)    # configured scorer, then top-k
                                           # + rarest-token coverage
                                           #   → evidence + citation

    Args:
        fact : atomic claim string from claim_extractor

    Returns:
        {
            "chunks"     : list of up to TOP_K_CHUNKS most relevant chunks
                           (empty if retrieval failed — verifier yields
                           NOT ENOUGH INFO gracefully),
            "source_url" : URL of the article the best chunk came from
                           ("" if unavailable),
            "query"      : the NER search query used (kept for logging and
                           for postprocessor's fallback URL resolution),
        }
    """
    print(f"\n[Retriever] Starting retrieval for fact: '{fact}'")

    # Multi-query retrieval: one combined query plus one standalone query
    # per top entity (see ner.extract_queries). A single query anchored on
    # a hallucinated or mis-typed entity used to poison the whole evidence
    # pool (e.g. "Gustave Eiffel Bedford Basin" → World Heritage list).
    queries = ner.extract_queries(fact)

    articles: list[tuple[str, str]] = []
    seen = set()
    for query in queries:
        for content, url in fetch_articles(query):
            # Deduplicate articles retrieved by more than one query.
            key = url or content[:80]
            if key in seen:
                continue
            seen.add(key)
            articles.append((content, url))

    chunks = chunk_articles(articles)
    top_chunks, source = rank_chunks(fact, chunks)

    print(
        f"[Retriever] Retrieval complete. {len(queries)} queries, "
        f"{len(articles)} unique article(s), {len(top_chunks)} chunk(s) ready for verifier."
    )
    return {"chunks": top_chunks, "source_url": source, "query": queries[0]}