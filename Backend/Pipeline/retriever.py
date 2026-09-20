import math
import re
import time
from collections import Counter

import requests
import wikipedia
from langchain_community.retrievers import WikipediaRetriever

import Pipeline.ner as ner

# ── Wikipedia client identification ───────────────────────────────────────────
# The `wikipedia` package (used internally by LangChain's WikipediaRetriever)
# ships a generic default User-Agent that Wikimedia's bot policy now blocks
# or rate-limits — when that happens the API answers with an HTML error page,
# and the library dies with "Expecting value: line 1 column 1 (char 0)"
# (it tried to json-parse HTML). Per https://meta.wikimedia.org/wiki/User-Agent_policy
# clients must send an identifying UA with contact info.
WIKI_USER_AGENT = "AutoCitation/0.1 (PoC fact-checker; contact: mehmet17b.b@gmail.com)"

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
    Fetches Wikipedia article content for a search query using LangChain's
    WikipediaRetriever, keeping each article's canonical URL alongside its
    text (doc.metadata["source"]).

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

    return max(present, key=lambda t: math.log(1 + total / (1 + df[t])))


def select_top_chunks(scored_chunks: list[tuple[float, str, str]],
                      must_contain: str | None = None) -> tuple[list[str], str]:
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

    selected = list(scored_chunks[:TOP_K_CHUNKS])

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
                (row for row in scored_chunks[TOP_K_CHUNKS:]
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# Called by main.py orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def fetch(fact: str) -> dict:
    """
    Executes the full retrieval pipeline for a single atomic fact.
    Orchestrates NER query extraction, Wikipedia article fetching,
    chunking, scoring, and top-k selection into a single callable.

    Called by claim_extractor.grounded_verify() inside the AFEV loop.

    Execution Flow:
        [Atomic Fact]
            → ner.extract_queries()        # combined + per-entity queries
            → fetch_articles(q) per query  # Wikipedia API → (text, url) pairs,
                                           #   deduplicated across queries
            → chunk_articles(articles)     # sentence-aware → url-tagged chunks
            → score_chunks(fact, chunks)   # token overlap → ranked chunks
            → select_top_chunks(scored)    # top-k + rarest-token coverage
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

    chunks             = chunk_articles(articles)
    scored_chunks      = score_chunks(fact, chunks)
    top_chunks, source = select_top_chunks(
        scored_chunks, must_contain=rarest_fact_token(fact, chunks)
    )

    print(
        f"[Retriever] Retrieval complete. {len(queries)} queries, "
        f"{len(articles)} unique article(s), {len(top_chunks)} chunk(s) ready for verifier."
    )
    return {"chunks": top_chunks, "source_url": source, "query": queries[0]}