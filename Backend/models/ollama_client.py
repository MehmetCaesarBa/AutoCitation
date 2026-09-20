import json
import time

import requests

import config

# ── Ollama REST API endpoints ─────────────────────────────────────────────────
# GENERATE_URL: The primary completions endpoint. Both pipeline models
# (phi3:mini and qwen3:8b) are called through this single endpoint —
# the target model is selected per-request via the "model" field in the
# JSON payload, not via separate endpoints.
GENERATE_URL = "http://localhost:11434/api/generate"

# HEALTH_URL: Ollama's root endpoint returns a plain "Ollama is running"
# string when the server is up. Used by health_check() at pipeline startup
# to fail fast with a clear error rather than hitting a connection refused
# mid-pipeline after claim extraction has already run.
HEALTH_URL = "http://localhost:11434"

# ── Model role registry ───────────────────────────────────────────────────────
# MODEL_ROLES: Maps semantic role names to their Ollama model identifiers.
#
# Why role names instead of model name strings scattered across modules?
# claim_extractor.py and verifier.py currently hardcode "phi3:mini" and
# "qwen3:8b" directly inside their call_ollama() functions. When a model
# is swapped (e.g., phi3:mini → phi4:mini after a benchmark), every file
# that hardcodes the string must be found and edited. With role dispatch,
# only this dict needs to change — all callers use "fast" or "reasoning"
# and get the updated model automatically.
#
# "fast"      → Lightweight extraction model. Low VRAM, high token/s.
#               Used by claim_extractor.py for iterative fact decomposition
#               where many LLM calls are made per input paragraph.
# "reasoning" → Larger reasoning model. Deeper chain-of-thought capability.
#               Used by verifier.py for SUPPORTS/REFUTES/NOT ENOUGH INFO
#               classification where accuracy matters more than speed.
MODEL_ROLES: dict[str, str] = {
    "fast"      : "phi3:mini",
    "reasoning" : "qwen3:8b",
}

# ── Registry sources for automatic pulling ────────────────────────────────────
# Maps a LOCAL model name to the REGISTRY name it can be downloaded from.
#
# Most entries are identity mappings: "qwen3:8b" is published under that exact
# name on ollama.com. The fine-tuned extractor is the exception — locally it is
# registered as "autocitation-extractor" by `ollama create`, but nobody else has
# run that command, so a fresh clone must fetch it from a namespace instead.
#
# Publishing it is a one-time step, and it distributes far more than the
# weights: `ollama push` bundles the Modelfile's TEMPLATE, stop tokens and
# parameters with the GGUF. That matters here specifically — finetune/Modelfile
# documents that Ollama otherwise guesses the chat template from GGUF metadata
# and picked "zephyr", wrapping every prompt in a format the fine-tune never
# saw and producing generic, off-task replies. Shipping a bare .gguf leaves
# every user one step away from reproducing that failure and concluding the
# model is bad.
#
#     ollama signin
#     ollama create <namespace>/autocitation-extractor -f Backend/finetune/Modelfile
#     ollama push   <namespace>/autocitation-extractor
#
# Replace the namespace below with your own once published.
MODEL_REGISTRY_SOURCES: dict[str, str] = {
    "autocitation-extractor": "mehmetba/autocitation-extractor",
}

# Pulling an 8B model is a multi-gigabyte download; it must not inherit the
# per-request inference timeout.
PULL_TIMEOUT_SECONDS = 3600

# ── Request config ────────────────────────────────────────────────────────────
# REQUEST_TIMEOUT: Per-request HTTP timeout in seconds.
# Local Ollama inference on a 7-8B model at 25+ tokens/s (the project's
# success threshold) should complete a ~500-token prompt in under 20s.
# Setting 60s gives 3x headroom for occasional slower batches without
# hanging the pipeline indefinitely on a stalled model process.
REQUEST_TIMEOUT = 60

# RETRY_ATTEMPTS: Number of times to retry a failed Ollama call before
# raising. Transient failures (Ollama briefly overloaded, single dropped
# request) are common when running local inference alongside other processes.
# 3 attempts covers the vast majority of transient cases without masking
# genuine model unavailability.
RETRY_ATTEMPTS = 3

# RETRY_DELAY: Seconds to wait between retry attempts.
# Exponential back-off is not used here because local Ollama failures are
# almost always caused by brief CPU/GPU contention rather than rate limits.
# A flat 2-second pause is sufficient for the process scheduler to free
# resources between attempts.
RETRY_DELAY = 2.0

# STREAM: Whether to use Ollama's streaming mode.
# Streaming (True) returns tokens incrementally via chunked HTTP — useful
# for a live frontend typing effect but requires a streaming response reader.
# Non-streaming (False) returns the full response in one JSON payload, which
# is simpler to parse and sufficient for the current batch-oriented pipeline.
# Set to False until a streaming frontend is implemented in index.html.
STREAM = False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Model Role Resolution
# ─────────────────────────────────────────────────────────────────────────────
def resolve_model(role: str) -> str:
    """
    Resolves a semantic role name to its registered Ollama model identifier.

    Callers (claim_extractor.py, verifier.py) pass a role string rather than
    a model name so that model substitutions require only a single change in
    MODEL_ROLES rather than edits across every pipeline module.

    Validation:
    An unrecognized role raises a ValueError immediately rather than
    silently falling back to a default model. A silent fallback would mean
    a typo in a caller (e.g., "fastt") silently routes to the wrong model,
    producing subtly wrong outputs that are hard to trace. A loud failure
    at call time is easier to debug.

    Args:
        role : semantic role string — must be a key in MODEL_ROLES.
               Currently valid values: "fast", "reasoning"

    Returns:
        Ollama model identifier string (e.g., "phi3:mini", "qwen3:8b")

    Raises:
        ValueError : if role is not registered in MODEL_ROLES

    Example:
        resolve_model("fast")      → "phi3:mini"
        resolve_model("reasoning") → "qwen3:8b"
        resolve_model("unknown")   → raises ValueError
    """
    if role not in MODEL_ROLES:
        valid = ", ".join(f'"{k}"' for k in MODEL_ROLES)
        raise ValueError(
            f"[OllamaClient] Unknown model role '{role}'. "
            f"Valid roles are: {valid}. "
            f"To add a new role, update MODEL_ROLES in ollama_client.py."
        )

    model = MODEL_ROLES[role]
    print(f"[OllamaClient] Role '{role}' resolved to model '{model}'.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Payload Construction
# ─────────────────────────────────────────────────────────────────────────────
def build_payload(model: str, prompt: str) -> dict:
    """
    Constructs the JSON payload for a single Ollama /api/generate request.

    Centralizing payload construction here ensures that every call to Ollama
    uses consistent options (stream mode, timeout behavior) regardless of
    which pipeline module initiated the request. If a new Ollama option is
    needed (e.g., temperature, num_ctx, seed for reproducibility), it is
    added once here rather than in each caller.

    Ollama /api/generate payload fields used:
    - model  : the model identifier string (e.g., "phi3:mini")
    - prompt : the full prompt string to generate a completion for
    - stream : False → return complete response in one JSON object.
               True  → stream tokens as newline-delimited JSON (not used here).

    Args:
        model  : resolved Ollama model identifier from resolve_model()
        prompt : complete prompt string to send to the model

    Returns:
        Dict ready for json= parameter of requests.post()

    Example:
        build_payload("phi3:mini", "Extract a fact from: ...")
        → {"model": "phi3:mini", "prompt": "Extract a fact from: ...", "stream": False}
    """
    return {
        "model"  : model,
        "prompt" : prompt,
        "stream" : STREAM,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Response Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_response(raw: requests.Response) -> str:
    """
    Extracts the model's text output from a raw Ollama HTTP response.

    Ollama's non-streaming /api/generate response is a JSON object with a
    top-level "response" key containing the model's generated text. All
    other fields (model, created_at, done, context, total_duration, etc.)
    are metadata used for monitoring and are not needed by pipeline modules.

    Error handling:
    - Missing "response" key: can occur if Ollama returns a partial error
      JSON (e.g., {"error": "model not found"}). Returns empty string and
      logs the full response body so the issue is visible without a crash.
    - Non-JSON body: can occur on rare Ollama internal errors that return
      plain text. Returns empty string rather than raising, allowing the
      retry loop in generate() to attempt recovery.

    Args:
        raw : completed requests.Response object from the Ollama endpoint

    Returns:
        Stripped model response string.
        Empty string if the "response" key is absent or body is not JSON.

    Example:
        raw.json() == {"model": "phi3:mini", "response": "Fact_1: ...", "done": True}
        → "Fact_1: ..."
    """
    try:
        body = raw.json()
    except ValueError:
        print(f"[OllamaClient] Response body was not valid JSON: {raw.text[:200]}")
        return ""

    if "response" not in body:
        print(f"[OllamaClient] 'response' key missing from Ollama output: {body}")
        return ""

    return body["response"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Retry-Wrapped Generate Call
# ─────────────────────────────────────────────────────────────────────────────
def generate(role: str, prompt: str) -> str:
    """
    Sends a prompt to the specified model role and returns the response string.

    This is the primary function called by claim_extractor.py and verifier.py.
    It replaces both of their local call_ollama() functions with a single
    shared implementation that adds:

    1. Role dispatch       : "fast" / "reasoning" → model identifier
    2. Retry logic         : up to RETRY_ATTEMPTS attempts on failure
    3. Timeout enforcement : REQUEST_TIMEOUT seconds per attempt
    4. Structured logging  : per-attempt status for pipeline debugging

    Retry behavior:
    Retries are triggered by any of:
    - requests.exceptions.Timeout     : model took longer than REQUEST_TIMEOUT
    - requests.exceptions.ConnectionError : Ollama process not reachable
    - response.raise_for_status()     : HTTP 4xx/5xx from the Ollama server
    - Empty string from parse_response(): Ollama returned malformed JSON

    An empty-string response after successful HTTP is treated as a soft
    failure and retried, because Ollama occasionally returns an empty
    "response" field on the first token if the model is still loading
    its KV cache for a new prompt format.

    After RETRY_ATTEMPTS exhausted, raises the last captured exception
    rather than returning an empty string, because a persistent failure
    indicates a model availability problem that the caller cannot
    silently recover from.

    Args:
        role   : semantic role — "fast" for extraction, "reasoning" for verification
        prompt : complete prompt string

    Returns:
        Model response string (stripped, non-empty on success)

    Raises:
        requests.exceptions.RequestException : if all retry attempts fail
        ValueError                            : if role is not in MODEL_ROLES

    Example:
        generate("fast", "Extract Fact_1 from: 'Python was created in 1991...'")
        → "Fact_1: Python was created in 1991 by Guido van Rossum."

        generate("reasoning", "LABEL: SUPPORTS or REFUTES? Claim: ...")
        → "LABEL: SUPPORTS\\nEVIDENCE: ...\\nRATIONALE: ..."
    """
    model      = resolve_model(role)
    payload    = build_payload(model, prompt)
    last_error = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            print(f"[OllamaClient] Attempt {attempt}/{RETRY_ATTEMPTS} — model: '{model}'")

            response = requests.post(
                GENERATE_URL,
                json    = payload,
                timeout = REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            text = parse_response(response)

            if not text:
                # Treat empty response as a soft failure — worth retrying
                # in case the model was mid-load on the first attempt.
                raise ValueError(
                    f"[OllamaClient] Empty response string from model '{model}' "
                    f"on attempt {attempt}."
                )

            print(f"[OllamaClient] Response received ({len(text)} chars).")
            return text

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"[OllamaClient] Network error on attempt {attempt}: {e}")
            last_error = e

        except requests.exceptions.HTTPError as e:
            print(f"[OllamaClient] HTTP error on attempt {attempt}: {e}")
            last_error = e

        except ValueError as e:
            # Covers both empty-response and JSON parse failures
            print(f"[OllamaClient] Parse/empty error on attempt {attempt}: {e}")
            last_error = e

        if attempt < RETRY_ATTEMPTS:
            print(f"[OllamaClient] Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    # All attempts exhausted — raise to surface the failure to main.py
    print(f"[OllamaClient] All {RETRY_ATTEMPTS} attempts failed for model '{model}'.")
    raise last_error


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Server Health Check
# ─────────────────────────────────────────────────────────────────────────────
def health_check() -> bool:
    """
    Verifies that the local Ollama server is reachable before the pipeline runs.

    Called once by main.py at FastAPI startup (via a lifespan event or the
    /check endpoint's pre-flight guard) to fail fast with a clear error
    message if Ollama is not running, rather than allowing the first
    claim_extractor call to raise an unhandled ConnectionError mid-pipeline.

    Why check at startup rather than per-request?
    Per-request health checks add ~5ms of latency per LLM call (one extra
    HTTP round-trip). A startup check pays that cost once and avoids it on
    every subsequent call. If Ollama goes down mid-run, the retry logic in
    generate() handles it with structured logging.

    Ollama's root endpoint (HEALTH_URL = "http://localhost:11434") returns
    "Ollama is running" as plain text with HTTP 200 when the server is up.
    We check for HTTP 200 only — we do not validate the response body text
    because Ollama's startup message is not part of its public API contract
    and may change across versions.

    Args:
        None

    Returns:
        True  : Ollama server responded with HTTP 200
        False : Server unreachable, timed out, or returned a non-200 status

    Example:
        health_check()  # Ollama running
        → True

        health_check()  # Ollama not started
        → False  (prints: "[OllamaClient] Health check failed: ...")
    """
    print(f"[OllamaClient] Running health check at {HEALTH_URL} ...")

    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            print("[OllamaClient] Ollama is reachable. Health check passed.")
            return True
        else:
            print(
                f"[OllamaClient] Health check returned unexpected status "
                f"{response.status_code}."
            )
            return False

    except requests.exceptions.RequestException as e:
        print(f"[OllamaClient] Health check failed: {e}")
        print(
            "[OllamaClient] Ensure Ollama is running with: ollama serve"
        )
        return False


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Model Availability Check
# ─────────────────────────────────────────────────────────────────────────────
def check_models() -> dict[str, bool]:
    """
    Verifies that all registered model roles are available in the local Ollama
    instance by querying the /api/tags endpoint.

    Called once at startup alongside health_check() to catch missing model
    pulls early. If a model listed in MODEL_ROLES has not been pulled with
    `ollama pull <model>`, the first generate() call will fail with a cryptic
    HTTP 404. This function surfaces that problem at startup with a clear
    per-model status message and actionable pull instructions.

    Ollama's /api/tags endpoint returns a JSON object:
        {"models": [{"name": "phi3:mini", ...}, {"name": "qwen3:8b", ...}]}

    We extract the "name" field from each entry and check whether each
    MODEL_ROLES value appears as a prefix match (Ollama sometimes appends
    ":latest" or digest suffixes to model names in the tag list).

    Args:
        None

    Returns:
        Dict mapping each role name to a boolean availability flag.
        Example: {"fast": True, "reasoning": False}

    Example:
        check_models()
        # phi3:mini pulled, qwen3:8b not pulled
        → {"fast": True, "reasoning": False}
        # Prints: "[OllamaClient] Model 'qwen3:8b' (role: 'reasoning') NOT FOUND.
        #          Run: ollama pull qwen3:8b"
    """
    tags_url = "http://localhost:11434/api/tags"
    print("[OllamaClient] Checking registered model availability...")

    try:
        response = requests.get(tags_url, timeout=5)
        response.raise_for_status()
        available_names: list[str] = [
            m["name"] for m in response.json().get("models", [])
        ]
    except requests.exceptions.RequestException as e:
        print(f"[OllamaClient] Could not reach /api/tags for model check: {e}")
        # Return all roles as unknown-unavailable rather than crashing
        return {role: False for role in MODEL_ROLES}

    status: dict[str, bool] = {}

    for role, model_id in MODEL_ROLES.items():
        # Prefix match: "phi3:mini" matches "phi3:mini:latest" etc.
        found = any(name.startswith(model_id) for name in available_names)
        status[role] = found

        if found:
            print(f"[OllamaClient] Model '{model_id}' (role: '{role}') — available.")
        else:
            print(
                f"[OllamaClient] Model '{model_id}' (role: '{role}') — NOT FOUND. "
                f"Run: ollama pull {model_id}"
            )

    return status


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Automatic model provisioning
# ─────────────────────────────────────────────────────────────────────────────
def pull_model(registry_name: str) -> bool:
    """
    Download a model into the local Ollama instance, streaming progress.

    Returns True on success. Never raises: a failed pull is reported and the
    server still starts, matching the fail-soft policy of the other startup
    checks — a developer with a flaky connection should still be able to run
    the API against whatever models they already have.

    /api/pull streams newline-delimited JSON status objects. Consuming the
    stream rather than blocking on stream=False matters for a multi-gigabyte
    download: without progress output the process looks frozen for several
    minutes and people kill it.
    """
    print(f"[OllamaClient] Pulling '{registry_name}' — this may take a while.")

    try:
        with requests.post(
            f"{config.OLLAMA_BASE_URL}/api/pull",
            json={"model": registry_name, "stream": True},
            stream=True,
            timeout=PULL_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()

            last_percent = -1
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("error"):
                    print(f"[OllamaClient] Pull failed: {event['error']}")
                    return False

                total, completed = event.get("total"), event.get("completed")
                if total:
                    percent = int(completed / total * 100)
                    # Only print each 10% step — iter_lines yields hundreds of
                    # events per second and would otherwise flood the log.
                    if percent >= last_percent + 10:
                        last_percent = percent
                        print(f"[OllamaClient]   {percent:3d}%  {event.get('status', '')}")

    except requests.exceptions.RequestException as e:
        print(f"[OllamaClient] Pull of '{registry_name}' failed: {e}")
        return False

    print(f"[OllamaClient] '{registry_name}' is ready.")
    return True


def ensure_models_available(auto_pull: bool = True) -> dict[str, bool]:
    """
    Make every model the pipeline needs present locally, pulling what is missing.

    This is what lets a fresh clone work without the reader following a notebook
    or downloading a 2 GB file by hand: the first run fetches whatever is absent,
    every run afterwards finds it cached and starts instantly.

    NOTE ON WHICH MODELS ARE CHECKED: the names come from MODEL_ROLES, which is
    the registry of what this client dispatches. claim_extractor.py and
    verifier.py currently hardcode their own model names instead of routing
    through here, so if you change one of those, change MODEL_ROLES too or the
    provisioning will check for a model nothing uses.

    auto_pull=False downgrades this to the reporting behaviour of
    check_models() — useful in CI, where downloading gigabytes is not wanted.
    """
    status = check_models()
    missing = [MODEL_ROLES[role] for role, ok in status.items() if not ok]

    if not missing:
        return status

    if not auto_pull:
        print(f"[OllamaClient] {len(missing)} model(s) missing; auto-pull disabled.")
        return status

    for local_name in missing:
        # A locally-created model (from `ollama create`) is not downloadable
        # under that name — it has to come from a published namespace.
        registry_name = MODEL_REGISTRY_SOURCES.get(local_name, local_name)

        if pull_model(registry_name) and registry_name != local_name:
            print(
                f"[OllamaClient] NOTE: pulled as '{registry_name}'. The pipeline "
                f"asks for '{local_name}' — alias it once with:\n"
                f"    ollama cp {registry_name} {local_name}"
            )

    return check_models()