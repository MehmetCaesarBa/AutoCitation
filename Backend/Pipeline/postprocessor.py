import re
import urllib.parse
import requests

# ── Wikipedia URL config ──────────────────────────────────────────────────────
# WIKIPEDIA_API_URL: MediaWiki Action API endpoint used for page title
# resolution. We call this instead of constructing URLs by hand because
# the NER query (e.g., "Mehmed II Constantinople") is a free-text search
# string, not a guaranteed Wikipedia page title. The API's opensearch
# endpoint returns the canonical page title that Wikipedia itself would
# resolve the query to, which we then embed in the final URL.
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# WIKIPEDIA_PAGE_BASE: Base URL for constructing human-readable Wikipedia
# article links from a resolved page title.
# Format: WIKIPEDIA_PAGE_BASE + urllib.parse.quote(title.replace(" ", "_"))
WIKIPEDIA_PAGE_BASE = "https://en.wikipedia.org/wiki/"

# WIKIPEDIA_SEARCH_LIMIT: Number of candidate titles returned by the
# opensearch API per query. We only need the top result (index 0), but
# requesting 3 gives us fallback candidates if the top title is a
# disambiguation page (detectable by "(disambiguation)" in the title).
WIKIPEDIA_SEARCH_LIMIT = 3

# LABEL_COLORS: CSS class tokens used by the frontend to color-code each
# verification label. These map directly to classes in index.html.
# Defined here so postprocessor.py is the single source of truth for
# label presentation logic — frontend just reads the value.
LABEL_COLORS = {
    "SUPPORTS"        : "green",
    "REFUTES"         : "red",
    "NOT ENOUGH INFO" : "yellow",
}

# FALLBACK_URL: Returned as source_url when Wikipedia URL resolution fails
# entirely (network error, no results, disambiguation dead-end).
# An empty string would silently render as a broken link in the frontend;
# a None would require null-checks everywhere. A fallback search URL gives
# the user a recoverable path to investigate the claim themselves.
FALLBACK_URL = "https://en.wikipedia.org/wiki/Special:Search"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Wikipedia URL Resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_wikipedia_url(query: str) -> str:
    """
    Resolves a free-text NER search query to a canonical Wikipedia article URL.

    Why not construct the URL directly from the NER query?
    The NER query produced by ner.extract_query() is a multi-entity phrase
    (e.g., "Mehmed II Constantinople") optimized for Wikipedia's search
    engine, not for direct URL construction. Directly URL-encoding that
    phrase would produce a Special:Search URL rather than a real article
    page, which renders as a search results page in the browser — a poor
    citation experience.

    Instead, we hit the Wikipedia opensearch API (the same endpoint that
    powers the Wikipedia search bar) to resolve the query to its canonical
    page title, then construct a /wiki/<Title> URL from that title.

    Disambiguation handling:
    Wikipedia's opensearch can return disambiguation pages (e.g.,
    "Mehmed II (disambiguation)") as the top result when multiple articles
    share a name. We detect this by checking for "(disambiguation)" in the
    title and fall through to the next candidate if found. If all candidates
    are disambiguation pages or the list is exhausted, FALLBACK_URL is
    returned.

    Network resilience:
    Any requests exception (timeout, DNS failure, Wikipedia API outage)
    is caught and logged without crashing the pipeline. The verifier has
    already completed by the time this function is called, so a URL
    resolution failure should never block the main result from reaching
    the frontend.

    Args:
        query : free-text search string from ner.extract_query()
                (e.g., "Mehmed II Constantinople", "Python Guido van Rossum")

    Returns:
        Canonical Wikipedia article URL string (https://en.wikipedia.org/wiki/...)
        or FALLBACK_URL if resolution fails.

    Example:
        resolve_wikipedia_url("Mehmed II Constantinople")
        → "https://en.wikipedia.org/wiki/Mehmed_II"

        resolve_wikipedia_url("nonexistent topic xyz 99999")
        → "https://en.wikipedia.org/wiki/Special:Search"
    """
    print(f"[Postprocessor] Resolving Wikipedia URL for query: '{query}'")

    # Full-text search (action=query&list=search) instead of opensearch:
    # opensearch does *prefix* matching on page titles, so multi-entity NER
    # queries like "Gustave Eiffel Statue of Liberty" never matched any
    # title and every citation collapsed to the Special:Search fallback.
    # Full-text search ranks articles by content relevance — the same
    # engine behind Wikipedia's own search results page.
    params = {
        "action"   : "query",
        "list"     : "search",
        "srsearch" : query,
        "srlimit"  : WIKIPEDIA_SEARCH_LIMIT,
        "format"   : "json",
    }

    try:
        # Identifying User-Agent per Wikimedia policy — generic UAs get
        # blocked/rate-limited and return HTML instead of JSON (same failure
        # mode fixed in retriever.py).
        headers = {"User-Agent": "AutoCitation/0.1 (PoC fact-checker; contact: mehmet17b.b@gmail.com)"}
        response = requests.get(WIKIPEDIA_API_URL, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        hits = data.get("query", {}).get("search", [])

        for hit in hits:
            title = hit.get("title", "")

            # Skip disambiguation pages — they link to a list of articles,
            # not a specific factual source, which degrades citation quality.
            if "(disambiguation)" in title.lower():
                print(f"[Postprocessor] Skipping disambiguation page: '{title}'")
                continue

            url = WIKIPEDIA_PAGE_BASE + urllib.parse.quote(title.replace(" ", "_"))
            print(f"[Postprocessor] Resolved URL: {url}")
            return url

        # All candidates were disambiguation pages or list was empty
        print("[Postprocessor] No valid article found — using fallback URL.")
        return FALLBACK_URL

    except Exception as e:
        print(f"[Postprocessor] Wikipedia URL resolution failed: {e}")
        return FALLBACK_URL


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Label Normalization
# ─────────────────────────────────────────────────────────────────────────────
def normalize_label(raw_label: str) -> str:
    """
    Normalizes a raw verifier label string to one of the three canonical values.

    verifier.py's parse_verification_response() already enforces the label
    set {SUPPORTS, REFUTES, NOT ENOUGH INFO}. However, main.py may call
    postprocessor with results assembled from multiple stages, and edge cases
    exist where a label could arrive with inconsistent casing, extra
    whitespace, or a truncated value (e.g., "NOT ENOUGH" if the model cut
    its output mid-token).

    This function acts as a defensive normalization gate so that the final
    API response always contains a valid, frontend-renderable label string.

    Normalization rules (applied in order):
    1. Strip leading/trailing whitespace.
    2. Uppercase the entire string.
    3. Partial match: if the string starts with "NOT" → "NOT ENOUGH INFO"
       (handles "NOT ENOUGH", "NOT ENOUGH INFORMATION", etc.)
    4. Exact match against the canonical set.
    5. Default to "NOT ENOUGH INFO" for anything unrecognized.

    Args:
        raw_label : label string returned by verifier.verify()

    Returns:
        One of: "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO"

    Example:
        normalize_label("supports")        → "SUPPORTS"
        normalize_label("NOT ENOUGH")      → "NOT ENOUGH INFO"
        normalize_label("  Refutes  ")     → "REFUTES"
        normalize_label("INCONCLUSIVE")    → "NOT ENOUGH INFO"
    """
    cleaned = raw_label.strip().upper()

    # Partial prefix match for truncated "NOT ENOUGH ..." variants
    if cleaned.startswith("NOT"):
        return "NOT ENOUGH INFO"

    if cleaned in {"SUPPORTS", "REFUTES"}:
        return cleaned

    print(f"[Postprocessor] Unrecognized label '{raw_label}' — defaulting to NOT ENOUGH INFO.")
    return "NOT ENOUGH INFO"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Single Result Assembly
# ─────────────────────────────────────────────────────────────────────────────
def assemble_result(
    claim         : str,
    label         : str,
    rationale     : str,
    evidence      : str,
    ner_query     : str,
    retrieved_url : str = "",
    timings       : dict | None = None,
) -> dict:
    """
    Assembles one complete verification result dict for a single atomic fact.

    This is the function that transforms the raw pipeline outputs (strings
    from verifier.py) into the structured response object defined in the
    API contract. It is the only place where source_url is attached,
    label_color is derived, and evidence truncation is applied.

    Evidence truncation:
    verifier.py instructs the reasoning model to "copy the single most
    relevant evidence chunk exactly." In practice, qwen3:8b sometimes echoes
    the full chunk (up to 500 characters). The frontend results table renders
    this inline, so we truncate to 300 characters with an ellipsis to keep
    the table scannable without hiding the key evidence sentence.

    Label color:
    The color field maps the label to a CSS class token consumed by the
    frontend. Deriving it here (rather than in the frontend JS) keeps all
    label-to-presentation logic in one place and avoids duplicating the
    mapping in the HTML file.

    Args:
        claim     : original atomic claim string from claim_extractor
        label     : raw label string from verifier.verify() [pre-normalization]
        rationale : one-sentence explanation from verifier.verify()
        evidence  : the evidence chunk the verifier selected
        ner_query : the search query produced by ner.extract_query() for
                    this claim — used to resolve the Wikipedia source URL

    Returns:
        Dict matching the API response shape defined in the project spec:
        {
            "claim"      : str,
            "label"      : "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
            "label_color": "green" | "red" | "yellow",
            "rationale"  : str,
            "evidence"   : str   (truncated to 300 chars if needed),
            "source_url" : str   (canonical Wikipedia URL or fallback)
        }

    Example:
        assemble_result(
            claim     = "Constantinople fell in 1453",
            label     = "SUPPORTS",
            rationale = "The evidence confirms the date of the conquest.",
            evidence  = "Constantinople fell to Ottoman forces on May 29, 1453...",
            ner_query = "Constantinople Mehmed II"
        )
        → {
            "claim"      : "Constantinople fell in 1453",
            "label"      : "SUPPORTS",
            "label_color": "green",
            "rationale"  : "The evidence confirms the date of the conquest.",
            "evidence"   : "Constantinople fell to Ottoman forces on May 29, 1453...",
            "source_url" : "https://en.wikipedia.org/wiki/Fall_of_Constantinople"
          }
    """
    norm_label  = normalize_label(label)
    label_color = LABEL_COLORS.get(norm_label, "yellow")

    # Prefer the URL of the article the evidence chunk actually came from
    # (attached by retriever.fetch). Only fall back to search-based
    # resolution when retrieval produced no URL — this keeps the citation
    # and the evidence pointing at the same article.
    source_url = retrieved_url if retrieved_url else resolve_wikipedia_url(ner_query)

    # Truncate evidence for frontend display clarity.
    # Full chunk text is preserved for the label decision but truncated
    # here because the results table is not a document reader.
    MAX_EVIDENCE_DISPLAY = 300
    display_evidence = (
        evidence[:MAX_EVIDENCE_DISPLAY].rstrip() + "…"
        if len(evidence) > MAX_EVIDENCE_DISPLAY
        else evidence
    )

    result = {
        "claim"       : claim.strip(),
        "label"       : norm_label,
        "label_color" : label_color,
        "rationale"   : rationale.strip(),
        "evidence"    : display_evidence,
        "source_url"  : source_url,
        # Per-stage durations in seconds (extraction_s, retrieval_s,
        # verification_s), measured inside claim_extractor's loop.
        "timings"     : timings or {},
    }

    print(
        f"[Postprocessor] Assembled result — "
        f"label: {norm_label} | url: {source_url}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Aggregate Pipeline Results
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(raw_results: list[dict]) -> dict:
    """
    Aggregates a list of per-fact result dicts into the final API response body.

    Called by main.py after all atomic facts have been extracted, retrieved,
    and verified. Takes the flat list of assembled result dicts and wraps
    them in a top-level response envelope that includes:

    1. results      : the ordered list of per-fact dicts
    2. summary      : label frequency counts (how many SUPPORTS / REFUTES /
                      NOT ENOUGH INFO across all facts in the input)
    3. overall_label: a single top-level verdict for the entire input text,
                      derived from the majority label across all facts.
                      Tie-breaking order: REFUTES > NOT ENOUGH INFO > SUPPORTS,
                      because a single refuted claim is more significant than
                      uncertain or supported claims when assessing credibility.
    4. total_claims : total number of atomic facts that were processed

    Why include summary and overall_label?
    The frontend results table shows per-fact detail, but the user's primary
    question is "is my text well-supported?" A single top-level verdict
    answers that immediately without requiring the user to scan all rows.
    The summary counts provide the evidence behind that verdict.

    Tie-breaking rationale (REFUTES > NOT ENOUGH INFO > SUPPORTS):
    In a fact-checking context, a single refuted claim in an otherwise
    supported text signals a meaningful factual error. An "all supported"
    verdict despite one refuted claim would be actively misleading.
    The conservative ordering ensures the overall verdict never overstates
    confidence in the input text.

    Args:
        raw_results : list of dicts, each from assemble_result().
                      Expected keys per dict: claim, label, label_color,
                      rationale, evidence, source_url.
                      An empty list is valid (returns a zero-claim envelope).

    Returns:
        Top-level API response dict:
        {
            "total_claims"  : int,
            "overall_label" : "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
            "summary": {
                "SUPPORTS"        : int,
                "REFUTES"         : int,
                "NOT ENOUGH INFO" : int,
            },
            "results": [ { ...per-fact result dict... }, ... ]
        }

    Example:
        aggregate([
            {"label": "SUPPORTS", ...},
            {"label": "REFUTES",  ...},
            {"label": "SUPPORTS", ...},
        ])
        → {
            "total_claims"  : 3,
            "overall_label" : "REFUTES",
            "summary"       : {"SUPPORTS": 2, "REFUTES": 1, "NOT ENOUGH INFO": 0},
            "results"       : [...]
          }
    """
    summary = {
        "SUPPORTS"        : 0,
        "REFUTES"         : 0,
        "NOT ENOUGH INFO" : 0,
    }

    for result in raw_results:
        label = result.get("label", "NOT ENOUGH INFO")
        if label in summary:
            summary[label] += 1
        else:
            summary["NOT ENOUGH INFO"] += 1

    # Determine overall verdict using conservative tie-breaking order.
    # REFUTES takes precedence over all — even one refuted fact matters.
    if summary["REFUTES"] > 0:
        overall_label = "REFUTES"
    elif summary["NOT ENOUGH INFO"] > 0:
        overall_label = "NOT ENOUGH INFO"
    elif summary["SUPPORTS"] > 0:
        overall_label = "SUPPORTS"
    else:
        # No results at all — claim extraction yielded nothing
        overall_label = "NOT ENOUGH INFO"

    # Stage-time totals across all facts, for the frontend time bar.
    # total_s (wall clock for the whole request) is attached by main.py.
    stage_totals = {"extraction_s": 0.0, "retrieval_s": 0.0, "verification_s": 0.0}
    for result in raw_results:
        for key in stage_totals:
            stage_totals[key] += float(result.get("timings", {}).get(key, 0.0))
    stage_totals = {k: round(v, 2) for k, v in stage_totals.items()}

    response = {
        "total_claims"  : len(raw_results),
        "overall_label" : overall_label,
        "summary"       : summary,
        "results"       : raw_results,
        "timings"       : stage_totals,
    }

    print(
        f"[Postprocessor] Aggregation complete — "
        f"{len(raw_results)} claim(s) | overall: {overall_label} | "
        f"S:{summary['SUPPORTS']} R:{summary['REFUTES']} N:{summary['NOT ENOUGH INFO']}"
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Pipeline Integration Engine
# Called by main.py after all per-fact verify() calls are complete.
# ─────────────────────────────────────────────────────────────────────────────
def process(pipeline_outputs: list[dict]) -> dict:
    """
    Public entry point for the post-processing stage of the AutoCitation pipeline.

    Accepts a list of raw pipeline output dicts (one per atomic fact) from
    main.py, assembles each into a structured result via assemble_result(),
    and returns the complete aggregated API response via aggregate().

    Expected input format per dict (produced by main.py's orchestration loop):
    {
        "claim"     : str,   ← from claim_extractor
        "label"     : str,   ← from verifier.verify() return[0]
        "rationale" : str,   ← from verifier.verify() return[1]
        "evidence"  : str,   ← from verifier.verify() return[2]
        "ner_query" : str,   ← from ner.extract_query(), stored by main.py
    }

    Integration note for main.py:
        At the end of the per-fact loop, collect results into a list and call:

            import postprocessor
            final_response = postprocessor.process(pipeline_outputs)
            return final_response  # FastAPI serializes this dict to JSON

        main.py must pass "ner_query" in each dict so that postprocessor
        can resolve the correct Wikipedia URL per fact. The NER query is
        already computed during retrieval — main.py should store it
        alongside the verifier output rather than recomputing it here.

    Args:
        pipeline_outputs : list of raw per-fact result dicts from main.py

    Returns:
        Final API response dict ready for JSON serialization by FastAPI.
        See aggregate() docstring for the full response schema.

    Example:
        process([
            {
                "claim"     : "Constantinople fell in 1453",
                "label"     : "SUPPORTS",
                "rationale" : "Evidence confirms the date.",
                "evidence"  : "Constantinople fell on May 29, 1453...",
                "ner_query" : "Constantinople Mehmed II",
            },
            ...
        ])
        → {
            "total_claims"  : 1,
            "overall_label" : "SUPPORTS",
            "summary"       : {"SUPPORTS": 1, "REFUTES": 0, "NOT ENOUGH INFO": 0},
            "results"       : [
                {
                    "claim"       : "Constantinople fell in 1453",
                    "label"       : "SUPPORTS",
                    "label_color" : "green",
                    "rationale"   : "Evidence confirms the date.",
                    "evidence"    : "Constantinople fell on May 29, 1453...",
                    "source_url"  : "https://en.wikipedia.org/wiki/Fall_of_Constantinople"
                }
            ]
          }
    """
    print(f"\n[Postprocessor] Processing {len(pipeline_outputs)} pipeline result(s).")

    assembled = []
    for output in pipeline_outputs:
        result = assemble_result(
            claim         = output.get("claim",      ""),
            label         = output.get("label",      "NOT ENOUGH INFO"),
            rationale     = output.get("rationale",  ""),
            evidence      = output.get("evidence",   ""),
            ner_query     = output.get("ner_query",  ""),
            retrieved_url = output.get("source_url", ""),
            timings       = output.get("timings",    {}),
        )
        assembled.append(result)

    return aggregate(assembled)