import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import models.ollama_client as ollama_client
import Pipeline.claim_extractor as claim_extractor
import Pipeline.postprocessor as postprocessor
from config import MAX_INPUT_WORDS


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────
class CheckRequest(BaseModel):
    """
    Incoming request body for the POST /check endpoint.

    Fields:
        text : the user's input paragraph to fact-check.
                Validated here rather than inside claim_extractor so that
                FastAPI returns a structured 422 Unprocessable Entity with
                a clear message before any pipeline work begins.

    Validation:
        - Must be a non-empty string after stripping whitespace.
        - Word count must not exceed MAX_INPUT_WORDS (from config.py).
          Word count is checked here as a fast pre-flight guard; the
          deeper character-level validation (allowed characters, Turkish
          Unicode) is still performed inside claim_extractor.precondition()
          so that module remains independently testable.
    """
    text: str

    @field_validator("text")
    @classmethod
    def text_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Input text must not be empty.")

        word_count = len(v.split())
        if word_count > MAX_INPUT_WORDS:
            raise ValueError(
                f"Input exceeds the {MAX_INPUT_WORDS}-word limit "
                f"({word_count} words received). "
                f"Please shorten your text and try again."
            )
        return v


class FactResult(BaseModel):
    """
    Single per-fact verification result returned inside CheckResponse.

    All fields are populated by postprocessor.assemble_result() and passed
    through here unchanged. Defining the shape explicitly as a Pydantic model
    means FastAPI validates the postprocessor's output before serialization
    and generates accurate OpenAPI schema docs automatically.

    Fields:
        claim       : the atomic fact string extracted from the input
        label       : SUPPORTS | REFUTES | NOT ENOUGH INFO
        label_color : CSS class token for frontend color-coding (green/red/yellow)
        rationale   : one-sentence explanation from the reasoning model
        evidence    : truncated evidence chunk that drove the decision
        source_url  : canonical Wikipedia article URL (or fallback search URL)
    """
    claim       : str
    label       : str
    label_color : str
    rationale   : str
    evidence    : str
    source_url  : str
    # Per-stage durations in seconds: extraction_s, retrieval_s, verification_s
    timings     : dict[str, float] = {}


class SummaryBlock(BaseModel):
    """Label frequency counts across all facts in a single /check response."""
    SUPPORTS        : int
    REFUTES         : int
    NOT_ENOUGH_INFO : int = 0

    class Config:
        # Allow the postprocessor's "NOT ENOUGH INFO" key (with spaces)
        # to map to the Python-safe "NOT_ENOUGH_INFO" field name.
        populate_by_name = True


class CheckResponse(BaseModel):
    """
    Top-level response body for POST /check.

    Mirrors the aggregated output shape produced by postprocessor.aggregate()
    with the addition of Pydantic field validation for safe serialization.

    Fields:
        total_claims  : number of atomic facts processed
        overall_label : single top-level verdict for the full input text
        summary       : per-label counts across all facts
        results       : ordered list of per-fact FactResult objects
    """
    total_claims  : int
    overall_label : str
    summary       : dict[str, int]
    results       : list[FactResult]
    # Aggregated stage durations (extraction_s, retrieval_s, verification_s)
    # plus total_s — wall-clock time for the whole request.
    timings       : dict[str, float] = {}

    # Optional human-readable explanation, populated only when the outcome needs
    # one. A response with zero claims is otherwise indistinguishable from a
    # response the user should have got claims for, and "0 results" with no
    # reason is the least useful thing an API can say.
    message       : str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Application Lifespan (Startup Checks)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs startup logic before the server
    begins accepting requests, and teardown logic on shutdown.

    Startup checks (fail-fast philosophy):
    Both checks are informational rather than blocking — the server starts
    regardless of their outcome so that a developer can still hit /check
    and see a clear error message rather than a refused connection.
    Blocking startup on model availability would prevent the server from
    running even when Ollama is temporarily restarting mid-development.

    1. ollama_client.health_check()
       Pings localhost:11434 to confirm Ollama is running. Prints a
       "run: ollama serve" hint if unreachable.

    2. ollama_client.ensure_models_available()
       Queries /api/tags and PULLS anything missing rather than only
       reporting it. This replaced a check-and-warn call so that a fresh
       clone works without the reader downloading a 2 GB GGUF by hand or
       following the Colab notebook first — the first run fetches what is
       absent, later runs find it cached.

       Set AUTOCITATION_NO_AUTOPULL=1 to fall back to reporting only. CI
       wants that: a test run should fail loudly on a missing model, not
       quietly download several gigabytes.

    These checks surface the two most common local setup mistakes (Ollama
    not started, models not pulled) in the server log at startup rather
    than as cryptic ConnectionRefused errors during the first request.
    """
    print("\n[Main] AutoCitation pipeline starting up...")

    ollama_client.health_check()
    ollama_client.ensure_models_available(
        auto_pull = os.getenv("AUTOCITATION_NO_AUTOPULL") != "1"
    )

    print("[Main] Startup checks complete. Server ready.\n")
    yield
    print("[Main] AutoCitation pipeline shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AutoCitation API",
    description = (
        "Fact-checks a text paragraph by extracting atomic claims, "
        "retrieving Wikipedia evidence, and classifying each claim as "
        "SUPPORTS, REFUTES, or NOT ENOUGH INFO using local Ollama models."
    ),
    version  = "0.1.0",
    lifespan = lifespan,
)

# CORS middleware: permits the plain HTML/JS frontend (index.html opened
# directly from the file system as a file:// URL, or served on a different
# port during development) to call the FastAPI backend without browser
# cross-origin blocking. For PoC, all origins are allowed. In production
# this should be tightened to the specific frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["POST", "GET"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — (removed) Per-Fact Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────
# Retrieval + verification now happen inside claim_extractor's AFEV loop
# (grounded_verify) — the paper's intended design: each fact is verified
# against Wikipedia evidence the moment it is extracted, and that verified
# result doubles as the feedback signal for the next extraction iteration.
#
# The second verification pass that used to live here duplicated work and,
# worse, produced inconsistent verdicts: the loop's feedback came from
# phi3's world knowledge while the final label came from qwen against
# evidence — two different judgments of the same fact. The world-knowledge
# rationales were also the channel through which fabricated entities leaked
# into extracted claims.


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — POST /check Endpoint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/check", response_model=CheckResponse)
def check(request: CheckRequest) -> CheckResponse:
    """
    POST /check — Full AutoCitation pipeline endpoint.

    Accepts a paragraph of text, extracts atomic claims, retrieves Wikipedia
    evidence per claim, verifies each claim, and returns a structured
    verdict for the full input.

    Request body (JSON):
        { "text": "<paragraph up to 500 words>" }

    Response body (JSON):
        {
            "total_claims"  : int,
            "overall_label" : "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
            "summary"       : {"SUPPORTS": int, "REFUTES": int, "NOT ENOUGH INFO": int},
            "results": [
                {
                    "claim"       : str,
                    "label"       : "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
                    "label_color" : "green" | "red" | "yellow",
                    "rationale"   : str,
                    "evidence"    : str,
                    "source_url"  : str
                },
                ...
            ]
        }

    Error responses:
        422 Unprocessable Entity : input failed Pydantic validation
                                   (empty text, exceeds word limit)
        503 Service Unavailable  : claim extraction returned zero facts
                                   (precondition failure: invalid characters,
                                   or model returned no extractable claims)
        500 Internal Server Error: unexpected exception during pipeline
                                   (surfaced with detail message for debugging)

    Pipeline flow:
        CheckRequest.text
            → claim_extractor.run()          # iterative atomic fact extraction
            → [for each fact]
                → run_fact_pipeline(fact)    # NER + retrieval + verification
            → postprocessor.process()        # assemble + aggregate → response
    """
    print(f"\n[Main] POST /check — received {len(request.text.split())} words.")
    t_start = time.perf_counter()

    # REFUSED INPUT IS NOT AN EMPTY RESULT.
    #
    # precondition() returns None on rejection and run() then returns [], which
    # this endpoint reported as "no verifiable claims could be extracted". An
    # observed request containing "C++" was refused by a character allowlist and
    # answered in 0.00s with that message — describing text the pipeline had
    # never read. The user is told their writing is unverifiable when in fact it
    # was never examined.
    #
    # 400 because the caller can act on it; the zero-claim path below stays 200
    # because there the caller did nothing wrong.
    rejection = claim_extractor.validate_input(request.text)
    if rejection is not None:
        print(f"[Main] Input rejected — {rejection}.")
        raise HTTPException(
            status_code = 400,
            detail      = f"Input could not be processed because {rejection}.",
        )

    # ── Stage 1+2: Claim Extraction with Grounded Verification ───────────────
    # claim_extractor.run() returns fully verified per-fact result dicts
    # (claim, label, rationale, evidence, ner_query, source_url) — retrieval
    # and verification run inside the AFEV loop; see the STEP 4 note above.
    try:
        pipeline_outputs = claim_extractor.run(request.text)
    except Exception as e:
        print(f"[Main] Claim extraction raised an unexpected exception: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Claim extraction failed: {e}"
        )

    if not pipeline_outputs:
        # "Nothing checkable here" is a RESULT, not a server fault.
        #
        # This used to raise 503 Service Unavailable, which tells the client the
        # backend is down and makes the frontend render an outage. But zero
        # facts is a perfectly ordinary outcome: text that is entirely opinion,
        # or whose only claims the extractor could not decontextualize, is
        # correctly answered with "no verifiable claims found". Returning 200
        # with an empty claim list lets the UI say that, and keeps genuine
        # 5xx codes meaningful for genuine outages.
        # THE KEYS HERE MUST MATCH CheckResponse. They did not, and FastAPI
        # answered a perfectly ordinary "nothing to check" outcome with a 500:
        #
        #   ResponseValidationError: 2 validation errors
        #     ('response', 'total_claims')  Field required
        #     ('response', 'results')       Field required
        #
        # This branch returned 'claims' and 'elapsed_s' where the model declares
        # 'results' and 'timings'. Nothing catches that mismatch until the branch
        # is actually taken, because response_model validation only runs on a
        # response that exists — and the happy path builds its dict in
        # postprocessor.aggregate(), which does use the right names. So the two
        # exits from this endpoint had drifted into different shapes and only one
        # of them was ever exercised.
        #
        # Hence test_empty_result_matches_the_response_model in Tests/main_test.py.
        elapsed = time.perf_counter() - t_start
        reason = (
            "No verifiable claims could be extracted from the input. This "
            "happens when the text is opinion rather than fact, when its "
            "statements depend on context the text does not supply, or when "
            "the input failed preprocessing — check the console for a "
            "'Validation Failed' line."
        )
        print(f"[Main] No verifiable claims extracted ({elapsed:.2f}s) — "
              f"returning empty result set.")
        return {
            "total_claims"  : 0,
            "overall_label" : "NO CLAIMS",
            "summary"       : {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0},
            "results"       : [],
            "timings"       : {"total_s": round(elapsed, 2)},
            "message"       : reason,
        }

    print(f"[Main] {len(pipeline_outputs)} verified fact(s) extracted.")

    # ── Stage 3: Post-Processing ──────────────────────────────────────────────
    try:
        final_response = postprocessor.process(pipeline_outputs)
    except Exception as e:
        print(f"[Main] Post-processing failed: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Post-processing failed: {e}"
        )

    # Attach total wall-clock time; per-stage totals were computed by
    # postprocessor.aggregate() from the per-fact timing data.
    final_response.setdefault("timings", {})["total_s"] = round(
        time.perf_counter() - t_start, 2
    )

    print(
        f"[Main] /check complete — "
        f"{final_response['total_claims']} claim(s) | "
        f"overall: {final_response['overall_label']} | "
        f"{final_response['timings']['total_s']}s total"
    )

    return final_response


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Direct Execution Entry Point
# ─────────────────────────────────────────────────────────────────────────────
# Allows starting the server with `python main.py` (from the Backend folder)
# as an alternative to `uvicorn main:app --port 8000`. Port 8000 matches the
# hardcoded API_BASE in Frontend/index.html.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)