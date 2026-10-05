"""
llm.py — the one place that sends a generation request to Ollama.

Every module that generates text used to build its own /api/generate request:
verifier.call_ollama, program._call_answer_model, claim_extractor.call_ollama
and verifier_probe.call_raw. Each hardcoded the URL instead of reading
config.OLLAMA_BASE_URL, and each re-implemented the same <think> strip. Four
copies of plumbing is four places for one of them to drift — verifier_probe's
docstring already promised it "cannot drift from what the pipeline sends" while
sending a different temperature.

WHAT IS SHARED AND WHAT IS NOT. Only the request mechanics live here. Model
choice and sampling options stay in the calling module, beside the reasoning
that justifies them (see config.py on why model names are not centralised).

DELIBERATELY SEPARATE FROM ollama_client. That module provisions models —
health check, startup roster, pulls — and program.py was once broken by a
helper that a revert removed from it. This module is small, has one job, and
should change rarely; keep it that way.
"""

import re
from dataclasses import dataclass

import requests

import config

GENERATE_URL = f"{config.OLLAMA_BASE_URL}/api/generate"

_THINK_BLOCK = re.compile(r'<think>.*?</think>', re.DOTALL)


@dataclass(frozen=True)
class Generation:
    """
    One completed generation.

    `text` has any inline <think> block removed. `thinking` is the reasoning
    Ollama returns in its own field when `think` is set — empty otherwise.

    `truncated` is Ollama's own report that generation stopped at num_predict
    (done_reason == "length"). Callers previously had to infer truncation from
    what was missing in the text; this is the direct signal.
    """
    text: str
    thinking: str
    truncated: bool
    eval_count: int
    eval_seconds: float


def strip_think(text: str) -> str:
    """
    Remove inline <think>...</think> blocks.

    Defensive: with `think` set, current Ollama returns reasoning in a separate
    field, but an Ollama version that ignores `think` puts it inline, where it
    could be mistaken for the answer.
    """
    return _THINK_BLOCK.sub('', text).strip()


def log_stats(body: dict, tag: str) -> None:
    """Timing line for one inference. Missing fields must not raise."""
    ns = body.get("eval_duration") or 0
    tokens = body.get("eval_count") or 0
    seconds = ns / 1e9
    rate = tokens / seconds if seconds else 0.0
    print(f"[Inference] {tag:12} output {tokens:5d} tok in {seconds:6.1f}s ({rate:.1f} tok/s)")


def generate(model: str, prompt: str, *,
             options: dict,
             think: bool | str | None = None,
             keep_alive: str | None = None,
             timeout: float | None = None,
             log: bool = True) -> Generation:
    """
    POST one non-streaming generation and return it.

    `think=None` OMITS the field rather than sending False. Models without a
    reasoning mode (phi3:mini) never received it before this helper existed,
    and the request each caller sends should not change by being routed here.

    Raises requests exceptions unchanged; deciding whether a failed call is
    fatal belongs to the caller.
    """
    payload = {"model": model, "prompt": prompt, "stream": False, "options": options}
    if think is not None:
        # Top-level field, NOT an option — see Ollama's /api/generate schema.
        payload["think"] = think
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    response = requests.post(GENERATE_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()

    if log:
        log_stats(body, model)

    truncated = body.get("done_reason") == "length"
    if truncated:
        print(f"[Inference] {model}: generation hit num_predict="
              f"{options.get('num_predict')} and was cut off.")

    return Generation(
        text=strip_think(body.get("response", "")),
        thinking=body.get("thinking", "") or "",
        truncated=truncated,
        eval_count=body.get("eval_count") or 0,
        eval_seconds=(body.get("eval_duration") or 0) / 1e9,
    )
