# ─────────────────────────────────────────────────────────────────────────────
# build_dataset.py — Distillation dataset builder for the extractor fine-tune
#
# Runs LOCALLY (not in Colab). Uses your local Ollama model as the "teacher":
# it decomposes source paragraphs into atomic facts, the script filters out
# ungrounded/malformed outputs, and converts each decomposition into training
# examples in the EXACT prompt format the production pipeline uses
# (claim_extractor.build_extraction_prompt) — so the fine-tuned model is
# trained on precisely what it will see at inference time.
#
# Usage:
#   cd Backend
#   python finetune/build_dataset.py --num-paragraphs 300
#   python finetune/build_dataset.py --source-dir my_texts/ --num-paragraphs 0
#
# Output:
#   finetune/data/train.jsonl   (one {"prompt": ..., "completion": ...} per line)
#   finetune/data/val.jsonl
#
# Sources:
#   - Random Wikipedia paragraphs (default), AND/OR
#   - --source-dir: a folder of .txt files (use this when you move to more
#     reliable domain sources — sample paragraphs from THOSE domains so the
#     extractor trains on the text style it will actually serve).
#
# Expect ~1000 paragraphs to take several hours on a laptop (teacher model
# inference is the bottleneck). Run overnight; the script saves incrementally
# and can be re-run — it appends only new paragraphs.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

# Make Backend/ importable so we reuse the production prompt builder.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from Pipeline.claim_extractor import build_extraction_prompt  # noqa: E402
from Pipeline.ner import NLP  # noqa: E402  — reuse the already-loaded spaCy model

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
TEACHER_MODEL = "qwen3:8b"        # strongest model you have locally
MIN_WORDS     = 40                # paragraph size window
MAX_WORDS     = 120
MIN_FACTS     = 2                 # reject decompositions outside this range
MAX_FACTS     = 6
VAL_FRACTION  = 0.08
OUT_DIR       = Path(__file__).resolve().parent / "data"

DECOMPOSE_PROMPT = """You are an expert atomic fact extractor. Decompose the text into atomic facts.

Rules:
- Each fact is ONE plain declarative sentence expressing a single subject-relation-object relationship.
- Use ONLY information stated in the text. Copy names, dates, and places exactly as written. NEVER add details from your own knowledge.
- Resolve pronouns to the entity names given in the text.
- Output ONLY numbered lines in this exact format, nothing else:
Fact_1: <fact>
Fact_2: <fact>
...

Text:
\"\"\"{paragraph}\"\"\"

Facts:"""


# ── Teacher call ──────────────────────────────────────────────────────────────
def call_teacher(prompt: str) -> str:
    payload = {
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        # Disable qwen3's <think> reasoning blocks: they multiply generation
        # time on CPU and get stripped anyway. (Ignored by models/Ollama
        # versions that don't support it — the strip regex below still runs.)
        "think": False,
        "options": {"num_ctx": 4096, "temperature": 0.2},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    raw = r.json()["response"]
    # Strip qwen3 <think> blocks
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


# ── Source paragraphs ─────────────────────────────────────────────────────────
def wikipedia_paragraphs(n: int) -> list[str]:
    """Random Wikipedia article summaries, cut to paragraph-sized pieces."""
    import wikipedia
    wikipedia.set_rate_limiting(True)

    paragraphs, seen = [], set()
    while len(paragraphs) < n:
        try:
            title = wikipedia.random(1)
            if title in seen:
                continue
            seen.add(title)
            summary = wikipedia.summary(title, auto_suggest=False)
        except Exception:
            continue
        for para in summary.split("\n"):
            wc = len(para.split())
            if MIN_WORDS <= wc <= MAX_WORDS:
                paragraphs.append(para.strip())
                print(f"  [{len(paragraphs)}/{n}] {title}")
                break
    return paragraphs


def dir_paragraphs(source_dir: Path) -> list[str]:
    """Paragraphs from user-provided .txt files (your future reliable sources)."""
    paragraphs = []
    for f in sorted(source_dir.glob("*.txt")):
        for para in f.read_text(encoding="utf-8").split("\n\n"):
            para = " ".join(para.split())
            if MIN_WORDS <= len(para.split()) <= MAX_WORDS:
                paragraphs.append(para)
    return paragraphs


# ── Decomposition + filtering ────────────────────────────────────────────────
def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zçğışöü0-9]+", text.lower()) if len(w) > 3}


# Running counters for the atomicity filter, printed by main() at the end so
# you can judge its real-world hit rate (see finetune/test_atomicity_filter.py
# for the pre-flight accuracy check on labeled examples).
STATS = {"facts_seen": 0, "compound_rejected": 0}


# Subordinating markers that, when they introduce a clause with its OWN
# subject and finite verb, typically join two independent statements about
# (often) two different things rather than adding circumstantial detail to
# one statement — e.g. "while", "although", "whereas". "since" is excluded:
# it's ambiguous between temporal ("since 1990") and causal use, and the
# temporal reading is common enough in factual text that including it would
# risk over-rejecting good atomic facts — a false positive here silently
# discards good training data, which is worse than one missed catch.
SUBORDINATING_MARKERS = {"while", "although", "whereas", "though"}


def is_compound_fact(fact: str) -> bool:
    """
    Structural atomicity check — one spaCy dependency parse, no LLM call.

    Per the atomic-claim literature (FActScore; AFEV; "single relation/
    predicate" definitions used across claim-decomposition work), atomicity
    is defined by predicate count, not sentence length. Two patterns are
    treated as non-atomic (two independent predicates), covering both how
    they're commonly joined:

    1. COORDINATION — a second verb coordinated onto the main verb via
       and/but/or, e.g. "Eiffel designed the tower and built the statue's
       frame." Structurally: a token with dep_ == "conj", POS VERB/AUX.

    2. SUBORDINATION with its own subject — a clause introduced by while/
       although/whereas/though that has its own subject and finite verb,
       e.g. "The bridge was built in 1930 while the tunnel was completed
       in 1935." (two different subjects, two independent facts).
       Structurally: a token with dep_ == "advcl", POS VERB/AUX, whose
       children include both a "mark" matching SUBORDINATING_MARKERS and
       a subject (nsubj/nsubjpass). Requiring the subject is what keeps
       elliptical adverbial phrases ("operates while stationary" — no
       second subject/verb) from being flagged.

    Deliberately NOT flagged (these keep a single relation despite 'and',
    a subordinator, or extra clauses, so a length- or keyword-based filter
    would over-reject them):
        - compound subjects  : "Eiffel and his company built the tower."
        - compound objects   : "The museum contains paintings and sculptures."
        - relative clauses   : "Curie, who was born in Warsaw, won two prizes."
        - participial phrases: "The bridge, built in 1973, connects two continents."
        - elliptical adverbials: "The device operates while stationary."
    """
    doc = NLP(fact)

    for tok in doc:
        # 1. Coordination
        if tok.dep_ == "conj" and tok.pos_ in ("VERB", "AUX"):
            return True

        # 2. Subordination with its own subject
        if tok.dep_ == "advcl" and tok.pos_ in ("VERB", "AUX"):
            children = list(tok.children)
            marker = next((c for c in children if c.dep_ == "mark"), None)
            has_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in children)
            if marker is not None and marker.text.lower() in SUBORDINATING_MARKERS and has_subject:
                return True

    return False


def decompose(paragraph: str) -> list[str] | None:
    """Teacher decomposition with the same quality gates as the pipeline."""
    response = call_teacher(DECOMPOSE_PROMPT.format(paragraph=paragraph))

    raw_facts = []
    for m in re.finditer(r"(?im)^\s*Fact[_\s\-]?\d+\s*[:\-]\s*(.+?)\s*$", response):
        fact = re.sub(r"\s*\([^()]*\)[\s.]*$", "", m.group(1).strip()).strip()
        if fact:
            raw_facts.append(fact)

    # Atomicity gate: drop individual compound facts (cheap, no extra LLM
    # call) rather than rejecting the whole paragraph — one bad fact
    # shouldn't cost the other well-formed ones in the same decomposition.
    facts = []
    for fact in raw_facts:
        STATS["facts_seen"] += 1
        if is_compound_fact(fact):
            STATS["compound_rejected"] += 1
            print(f"  [Atomicity] rejected compound fact: '{fact}'")
            continue
        facts.append(fact)

    if not (MIN_FACTS <= len(facts) <= MAX_FACTS):
        return None

    # Grounding gate: every fact must share content words with the source.
    text_words = content_words(paragraph)
    for fact in facts:
        fw = content_words(fact)
        if fw and text_words and not (fw & text_words):
            return None  # one hallucinated fact rejects the whole decomposition

    return facts


def find_support_sentence(fact: str, paragraph: str) -> str:
    """Pseudo-rationale: the source sentence sharing most content words."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    fw = content_words(fact)
    best = max(sentences, key=lambda s: len(fw & content_words(s)), default=paragraph)
    return best.strip()


# ── Convert to production-format training examples ──────────────────────────
def to_examples(paragraph: str, facts: list[str]) -> list[dict]:
    """
    One decomposition becomes len(facts)+1 examples, each using the REAL
    production prompt (build_extraction_prompt):
        step k   : history of facts 1..k-1  → completion "Fact_k: <fact>"
        final    : history of all facts     → completion "Terminate"
    History labels/rationales are synthesized (SUPPORTS + supporting source
    sentence) — the extractor only uses them as context for coreference.
    """
    examples = []
    labels     = ["SUPPORTS"] * len(facts)
    rationales = [find_support_sentence(f, paragraph) for f in facts]

    for k in range(len(facts) + 1):
        prompt = build_extraction_prompt(
            paragraph, facts[:k], labels[:k], rationales[:k]
        )
        completion = f"Fact_{k + 1}: {facts[k]}" if k < len(facts) else "Terminate"
        examples.append({"prompt": prompt, "completion": completion})

    return examples


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-paragraphs", type=int, default=200,
                    help="random Wikipedia paragraphs to distill (0 to skip)")
    ap.add_argument("--source-dir", type=Path, default=None,
                    help="folder of .txt files to distill in addition/instead")
    args = ap.parse_args()

    # Preflight: fail fast with a clear message instead of a raw traceback
    # if Ollama isn't running yet — this is the #1 cause of an immediate crash.
    try:
        requests.get("http://localhost:11434/api/tags", timeout=3).raise_for_status()
    except Exception:
        print(
            "[Dataset] ERROR: Cannot reach Ollama at localhost:11434.\n"
            "  Start it in another terminal first:  ollama serve\n"
            "  Then rerun this exact command — nothing already saved is lost."
        )
        sys.exit(1)

    paragraphs = []
    if args.source_dir:
        paragraphs += dir_paragraphs(args.source_dir)
        print(f"[Dataset] {len(paragraphs)} paragraph(s) from {args.source_dir}")
    if args.num_paragraphs > 0:
        print(f"[Dataset] Sampling {args.num_paragraphs} Wikipedia paragraphs...")
        paragraphs += wikipedia_paragraphs(args.num_paragraphs)

    random.shuffle(paragraphs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_f = open(OUT_DIR / "train.jsonl", "a", encoding="utf-8")
    val_f   = open(OUT_DIR / "val.jsonl",   "a", encoding="utf-8")

    kept = 0
    for i, para in enumerate(paragraphs, 1):
        print(f"[Dataset] Distilling {i}/{len(paragraphs)}...")
        try:
            facts = decompose(para)
        except Exception as e:
            print(f"  teacher call failed: {e}")
            time.sleep(5)
            continue
        if facts is None:
            print("  rejected (fact count / grounding gate)")
            continue

        out = val_f if random.random() < VAL_FRACTION else train_f
        for ex in to_examples(para, facts):
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")
        out.flush()
        kept += 1
        print(f"  kept ({len(facts)} facts → {len(facts) + 1} examples)")

    train_f.close(); val_f.close()
    print(f"\n[Dataset] Done. {kept}/{len(paragraphs)} paragraphs kept.")
    if STATS["facts_seen"]:
        pct = STATS["compound_rejected"] / STATS["facts_seen"]
        print(
            f"[Dataset] Atomicity filter: {STATS['compound_rejected']}/{STATS['facts_seen']} "
            f"candidate facts rejected as compound ({pct:.1%})."
        )
    print(f"[Dataset] Files: {OUT_DIR / 'train.jsonl'}, {OUT_DIR / 'val.jsonl'}")
    print("[Dataset] Next: upload train.jsonl + val.jsonl to Google Drive "
          "(folder 'AutoCitation_finetune') and open the Colab notebook.")


if __name__ == "__main__":
    main()
