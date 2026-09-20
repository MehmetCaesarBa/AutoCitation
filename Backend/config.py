# ─────────────────────────────────────────────────────────────────────────────
# config.py — Shared Pipeline Configuration
#
# Scope: constants that are referenced by more than one module, or that
# represent a tunable project-level decision (model choice, word limit).
#
# What does NOT belong here:
# Pipeline-specific constants that are only ever read by one module
# (e.g., CHUNK_SIZE and TOP_K_CHUNKS in retriever.py, ENTITY_PRIORITY in
# ner.py) stay in their own files. Centralising them here would create an
# invisible coupling: a reader of retriever.py would have to open config.py
# to understand chunking behavior, and a change to config.py could silently
# affect a module the editor didn't intend to touch. Single-module constants
# belong with their module.
#
# Import pattern for all pipeline modules:
#     from config import OLLAMA_BASE_URL, FAST_MODEL, REASONING_MODEL, MAX_INPUT_WORDS
# ─────────────────────────────────────────────────────────────────────────────


# ── Ollama server ─────────────────────────────────────────────────────────────
# OLLAMA_BASE_URL: Root URL of the local Ollama REST server.
# All pipeline modules that talk to Ollama (ollama_client.py) derive their
# endpoint URLs from this base rather than hardcoding the host/port
# individually. Changing the port or moving Ollama to a remote host during
# development requires editing exactly one line here.
OLLAMA_BASE_URL: str = "http://localhost:11434"

# Derived endpoint URLs built from the base so callers never construct paths.
OLLAMA_GENERATE_URL: str = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL: str     = f"{OLLAMA_BASE_URL}/api/tags"


# ── Model identifiers ─────────────────────────────────────────────────────────
# FAST_MODEL: Lightweight extraction model used by claim_extractor.py.
# Selected for high token/s throughput — the extraction loop calls the
# model up to MAX_ITERATIONS times per input paragraph, so inference
# speed directly determines end-to-end latency more than accuracy does
# at this stage.
FAST_MODEL: str = "autocitation-extractor"

# REASONING_MODEL: Larger reasoning model used by verifier.py.
# Selected for deeper chain-of-thought capability needed to classify a
# claim as SUPPORTS / REFUTES / NOT ENOUGH INFO against evidence chunks.
# Accuracy matters more than throughput here — the verifier is called
# once per fact, not in a tight loop.
REASONING_MODEL: str = "qwen3:8b"


# ── Input validation ──────────────────────────────────────────────────────────
# MAX_INPUT_WORDS: Hard ceiling on user input length.
# Enforced by claim_extractor.precondition() before any LLM calls are made.
# Rationale: local 7-8B models have context windows of 4096-8192 tokens.
# A 500-word input (~650 tokens) leaves ample room for the system prompt,
# few-shot examples, and the model's generated output within one context
# window. Inputs beyond this limit would require chunking the input itself,
# which is out of scope for the PoC.
MAX_INPUT_WORDS: int = 500

