"""
backends.py — per-request selection of which Ollama the pipeline talks to.

The frontend offers a Local / RunPod toggle; its choice rides in the /check
request body, and main.py calls set_backend() once at the top of the request.
Every generation and embedding call then reads the active base URL live through
generate_url() / embed_url(), so one toggle moves the whole pipeline — verifier,
program Question, claim extraction, the query rewrite, and dense embedding — onto
the chosen server without changing a line of their logic.

WHY A ContextVar, NOT A MODULE GLOBAL. /check is a sync endpoint, so FastAPI runs
each request in its own threadpool thread with its own copied context. A
ContextVar is therefore isolated per request: two requests selecting different
backends cannot clobber each other's choice, where a plain global could. The
default is local, so anything that runs outside a request (startup, tests,
scripts) behaves exactly as before.

NOT the provisioning path. ollama_client still pulls and health-checks against
config.OLLAMA_BASE_URL (local) at startup — you provision the local instance
here and pull the pod's models on the pod. This module only routes inference.
"""
import contextvars

import config

# Toggle value (from the UI) -> base URL. Unknown values fall back to local.
BACKENDS: dict[str, str] = {
    "local":  config.OLLAMA_LOCAL_URL,
    "runpod": config.OLLAMA_RUNPOD_URL,
}
VALID_BACKENDS = frozenset(BACKENDS)

_active: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ollama_base_url", default=config.OLLAMA_LOCAL_URL
)


def set_backend(name: str) -> str:
    """Point this request's inference at `name`. Returns the resolved base URL."""
    base = BACKENDS.get(name, config.OLLAMA_LOCAL_URL)
    _active.set(base)
    return base


def base_url() -> str:
    return _active.get()


def generate_url() -> str:
    return f"{base_url()}/api/generate"


def embed_url() -> str:
    return f"{base_url()}/api/embed"
