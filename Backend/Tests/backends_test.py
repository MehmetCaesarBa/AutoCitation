"""
Tests for backends.py — the per-request Local/RunPod inference switch.

Pure: no network, no app. Covers URL resolution, the unknown-value fallback,
and the ContextVar isolation that makes concurrent requests safe.
"""
import contextvars

import config
import backends


def test_default_is_local():
    # A fresh context has never called set_backend → the module default.
    assert contextvars.Context().run(backends.base_url) == config.OLLAMA_LOCAL_URL


def test_set_backend_routes_generate_and_embed():
    backends.set_backend("runpod")
    assert backends.base_url()     == config.OLLAMA_RUNPOD_URL
    assert backends.generate_url() == config.OLLAMA_RUNPOD_URL + "/api/generate"
    assert backends.embed_url()    == config.OLLAMA_RUNPOD_URL + "/api/embed"

    backends.set_backend("local")
    assert backends.generate_url() == config.OLLAMA_LOCAL_URL + "/api/generate"


def test_unknown_backend_falls_back_to_local():
    assert backends.set_backend("nonsense") == config.OLLAMA_LOCAL_URL
    assert backends.base_url() == config.OLLAMA_LOCAL_URL


def test_valid_backends_set():
    assert backends.VALID_BACKENDS == frozenset({"local", "runpod"})


def test_requests_are_isolated_across_contexts():
    """
    /check is a sync endpoint, so each request runs in its own copied context.
    One request selecting runpod must not change what another request sees — a
    plain module global could not guarantee this; a ContextVar does.
    """
    def pick(name):
        backends.set_backend(name)
        return backends.base_url()

    ctx_a = contextvars.Context()
    ctx_b = contextvars.Context()
    assert ctx_a.run(pick, "runpod") == config.OLLAMA_RUNPOD_URL
    assert ctx_b.run(pick, "local")  == config.OLLAMA_LOCAL_URL
    # ctx_a's choice did not leak into ctx_b, nor into a third fresh context.
    assert contextvars.Context().run(backends.base_url) == config.OLLAMA_LOCAL_URL
