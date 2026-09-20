import json

import requests

import config

# ── Ollama REST API endpoints ─────────────────────────────────────────────────
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