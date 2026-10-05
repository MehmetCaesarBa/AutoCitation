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
# MODEL IDENTIFIERS ARE NOT HERE, DELIBERATELY. This file used to declare
# FAST_MODEL and REASONING_MODEL, and nothing ever read them: every module
# that calls Ollama names its model inline (claim_extractor.FAST_MODEL,
# program.ANSWER_MODEL, verifier.REASONING_MODEL), and ollama_client keeps
# the startup-check roster in its own MODEL_ROLES. The unread copies had
# drifted — this file claimed the extractor was "autocitation-extractor"
# while every run used phi3:mini — so they were worse than redundant: a
# reader consulting config.py would have learned the wrong model. If these
# ever come back, they have to be the values actually passed to Ollama, not
# a parallel declaration of intent.
# ─────────────────────────────────────────────────────────────────────────────


# ── Ollama server ─────────────────────────────────────────────────────────────
# OLLAMA_BASE_URL: Root URL of the local Ollama REST server.
# Read by ollama_client.pull_model() to build the /api/pull endpoint.
# Changing the port or moving Ollama to a remote host during development
# requires editing exactly one line here.
OLLAMA_BASE_URL: str = "http://localhost:11434"


# ── Input validation ──────────────────────────────────────────────────────────
# MAX_INPUT_WORDS: Hard ceiling on user input length.
# Enforced by claim_extractor.precondition() before any LLM calls are made.
# Rationale: local 7-8B models have context windows of 4096-8192 tokens.
# A 500-word input (~650 tokens) leaves ample room for the system prompt,
# few-shot examples, and the model's generated output within one context
# window. Inputs beyond this limit would require chunking the input itself,
# which is out of scope for the PoC.
MAX_INPUT_WORDS: int = 500


# ── Retrieval scoring ─────────────────────────────────────────────────────────
# EMBED_MODEL: Ollama embedding model for dense retrieval scoring.
# Read by retriever.embed_texts(). A small (~137M) embedding model, so it runs
# on CPU in a few seconds even for the 30-90 chunks a claim retrieves — no GPU
# and no PyTorch/transformers; it is reached over the same localhost Ollama HTTP
# API the generative models use.
EMBED_MODEL: str = "nomic-embed-text"

# RETRIEVAL_SCORING: which scorer retriever.fetch() ranks chunks with.
#   "idf"    — IDF-weighted exact-token overlap (the original, lexical only).
#   "dense"  — cosine similarity of nomic-embed-text embeddings (paraphrase).
#   "hybrid" — Reciprocal Rank Fusion of the two: a chunk must rank well on
#              BOTH the lexical and the dense scorer to surface.
#
# DEFAULT IS "idf" ON PURPOSE. Dense and hybrid add a per-claim call to the
# embedding endpoint; leaving the default lexical keeps behaviour, latency and
# startup provisioning identical to before until this is explicitly flipped.
# ollama_client.MODEL_ROLES reads this value and only enrolls EMBED_MODEL in
# startup auto-pull when it is not "idf", so an "idf" run pulls nothing new.
RETRIEVAL_SCORING: str = "hybrid"    # "idf" | "dense" | "hybrid"
