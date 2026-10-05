import re
import time

import Pipeline.ner as ner
import Pipeline.verification as verification
from models import llm

# ── Ollama config ─────────────────────────────────────────────────────────────
FAST_MODEL   = "phi3:mini"   # swap with your chosen fast model name in Ollama

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 10   # safety ceiling so the loop never runs forever

# How many consecutive rejected/malformed responses to tolerate before giving
# up on the current fact slot. Under greedy decoding the model is deterministic
# given a prompt, so an unchanged prompt cannot produce a new answer; we vary
# the prompt by feeding the rejection back, but if it still will not comply
# after being shown its own failures this many times, it will not comply at all.
# Lowered from 3: two unchanged rejections already show the feedback isn't
# changing the model's output, so a third attempt was just spending an
# iteration to confirm what the second one already showed.
MAX_CONSECUTIVE_REJECTIONS = 2

# Per-sentence iteration ceiling. A single sentence holds far fewer atomic
# facts than a document, so this can be much tighter than MAX_ITERATIONS —
# which now bounds total work across the whole input instead.
MAX_ITERATIONS_PER_SENTENCE = 5

# Whether the extraction history carries the verifier's Answer/Rationale lines
# (AFEV Eq. 2) or only the bare list of covered facts.
#
# Set False by default on the evidence of an observed run: after a REFUTES
# verdict, the extractor's very next output was the verifier's own correction
# restated as a fact ("The Bosporus forms the continental boundary between Asia
# and Europe"), which the faithfulness gate then had to reject. Eq. 2 was meant
# to aid coreference, but rule 4 now resolves references from the source
# document instead, leaving the rationales with no extraction purpose and a
# demonstrated cost. Flip to True to reproduce the paper's configuration and
# measure the difference rather than taking this on faith.
INCLUDE_RATIONALES_IN_HISTORY = False

# Function words the faithfulness gate tolerates in an extracted fact even when
# they do not occur in the source text. Only closed-class items belong here:
# auxiliaries, determiners, relativisers, connectives. Anything carrying
# referential content (a name, place, number, date) must come from the input.
# Words of 3 characters or fewer are filtered out before the check, so short
# items such as 'is', 'was', 'the', 'and', 'of' need no entry.
FUNCTION_WORDS = {
    "were", "been", "being", "have", "having", "does", "will", "would",
    "that", "this", "these", "those", "which", "whose", "there", "then",
    "they", "them", "their", "with", "within", "from", "into", "also",
    "than", "when", "while", "where", "both", "each", "such", "same",
    "some", "more", "most", "other", "another", "after", "before",
    "between", "during", "about", "only", "very", "much", "many",
}


_ARTICLES = {"the", "a", "an"}


def _normalize_claim(fact: str) -> str:
    """
    Canonical form used ONLY for duplicate detection — never for display,
    retrieval, or verification, all of which need the claim verbatim.

    Beyond lowercasing and collapsing whitespace, this drops articles and
    punctuation: differences that carry no propositional content. An observed
    run verified "...with Yavuz Sultan Selim Bridge." and "...with the Yavuz
    Sultan Selim Bridge." as two separate claims, at 118s and 130s, because
    the previous normalizer compared raw lowercased strings.

    Deliberately exact-match rather than similarity-based. A threshold would
    also catch paraphrases ("Bosporus is a strait located between X and Y"),
    but claims that differ in one content word can be genuine opposites —
    "between Africa and Europe" vs "between Asia and Europe" overlap almost
    entirely — and silently merging those would defeat the whole pipeline.
    """
    words = re.findall(r"[a-z0-9çğışöü]+", fact.lower())
    return " ".join(w for w in words if w not in _ARTICLES)


# ── Tunable parameters of the faithfulness vocabulary ─────────────────────────
# These were magic numbers inside _content_words. Naming them matters for more
# than tidiness: a threshold buried in a comprehension is a decision nobody
# made, nobody documents, and nobody can sweep. You cannot tune what you cannot
# name — and you cannot cite it either, which is the immediate problem: two
# comments elsewhere in this file, and ner.check_negation's docstring, already
# refer to MIN_CONTENT_WORD_LENGTH by name while it existed only as an inline
# `len(w) <= 3`.
#
# MIN_CONTENT_WORD_LENGTH excludes short function words so that 'the', 'and',
# 'of' never count as material the claim borrowed. It also has a consequence
# that was invisible while it was inline:
#
#   'not' IS THREE CHARACTERS, so the faithfulness gate cannot see negation.
#
# An observed run turned "It is universally acknowledged as longer than the
# Nile" into "The Nile is NOT universally acknowledged as the world's longest
# river" — a complete reversal — and this gate passed it, because every
# surviving word does appear in the source and the inserted 'not' was filtered
# out before comparison. LOWERING THIS IS NOT THE FIX: it would flood the set
# with articles and prepositions. Negation needed its own check, which is
# ner.check_negation, and this constant is named here so the limitation is
# visible at the point where it is created.
MIN_CONTENT_WORD_LENGTH = 4

# Suffixes stripped so that inflectional variants introduced by rewriting a
# clause ('separates' vs 'separated') do not register as foreign material.
# Order matters — longest first, first match wins.
_STEM_SUFFIXES = ("ing", "es", "ed", "s")

# A stem shorter than this is too mangled to compare, so the suffix is kept.
# Without it, 'axes' would stem to 'ax' and collide with unrelated words.
MIN_STEM_LENGTH = 4

# ── How content words are normalised ──────────────────────────────────────────
# "lemma_keep_propn"  spaCy lemma for ordinary words, proper nouns untouched.
#                     Won every column of benchmarks/stemmer_bench.py.
# "stem"              the previous behaviour — the four-suffix stripper above.
#                     Kept as the comparison baseline: flip this back and re-run
#                     the benchmark to reproduce the numbers rather than
#                     trusting them.
#
# The stemming constants above apply in "stem" mode only.
CONTENT_WORD_MODE = "lemma_keep_propn"


def _content_words_lemma(text: str) -> set[str]:
    """
    Lemmatise ordinary words; leave proper nouns exactly as written.

    WHY PART OF SPEECH IS THE MISSING INFORMATION. A stemmer is string rewriting
    with no concept of a name, so any rule that strips a terminal 's' turns
    every place name ending in -s into a plural: Paris->pari, Athens->athen,
    Wales->wale. Fact-checking prose is dense with such names, and a mangled
    name silently stops matching its own mention elsewhere.

    No pure stemmer can do better — once the string is lowercased, 'Paris' and
    'parries' are indistinguishable. The POS tag is what separates them, and
    spaCy already computes it for this text.

    The second gain is the opposite direction: lemmatisation collapses
    inflectional variants a suffix stripper misses entirely, including
    irregulars ('held'->'hold', 'built'->'build', 'went'->'go'), which is what
    the faithfulness gate needs when the extractor rewrites a clause. Fewer
    false rejections, so fewer wasted retries.

    Cost is a parse (~1ms for a claim, ~20ms for a document) instead of a regex.
    Against ~10-20s extraction calls that is not measurable, and
    extract_atomic_facts parses the document once per request, not per claim.
    """
    doc = ner.NLP(text)

    out: set[str] = set()
    for token in doc:
        if token.is_punct or token.is_space:
            continue

        surface = token.text.lower()
        if len(surface) < MIN_CONTENT_WORD_LENGTH:
            continue

        # PROPN passes through untouched. This is the whole point: a name is not
        # an inflected form of anything, so any normalisation of it is damage.
        out.add(surface if token.pos_ == "PROPN" else token.lemma_.lower())

    return out


def _content_words(text: str) -> set[str]:
    """
    Lowercase content words of `text`, crudely stemmed.

    Stemming is deliberately naive — it exists so that inflectional variants
    introduced by rewriting a clause into a standalone sentence ('separates'
    vs 'separated', 'prizes' vs 'prize') do not register as foreign material.
    It is not meant to be linguistically correct; it only needs to leave
    proper nouns and numbers untouched, which it does.

    See MIN_CONTENT_WORD_LENGTH above for why this function is blind to
    negation — a limitation, not an oversight, and one that has its own gate
    (ner.check_negation) rather than a different threshold here.

    CONTENT_WORD_MODE selects the implementation. The stemming path below is the
    baseline the lemma path was measured against, not dead code: it is what
    benchmarks/stemmer_bench.py compares, and deleting it would make the
    comparison unreproducible.
    """
    if CONTENT_WORD_MODE == "lemma_keep_propn":
        return _content_words_lemma(text)

    words = re.findall(r"[a-zçğışöü0-9]+", text.lower())
    stemmed = set()
    for w in words:
        if len(w) < MIN_CONTENT_WORD_LENGTH:
            continue
        for suffix in _STEM_SUFFIXES:
            if len(w) - len(suffix) >= MIN_STEM_LENGTH and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        stemmed.add(w)
    return stemmed

# ── Text you are testing with ─────────────────────────────────────────────────
text_ba = "Here is a 500-word text..."   # keep your original text here

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Precondition (your original function, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
# Maximum input length. A genuine resource guard: every sentence costs one or
# more LLM calls, so an unbounded input is an unbounded request.
MAX_INPUT_WORDS = 500

# CONTROL CHARACTERS ONLY — a denylist, not an allowlist.
#
# This was an ALLOWLIST of "acceptable" characters:
#
#     r'^[a-zA-ZçğışöüÇĞİŞÖÜ0-9\s.,!?\-\'\"()]+$'
#
# and it silently rejected this input:
#
#     "Both Python and C++ are widely used object-oriented programming
#      languages..."
#
# because of the '+' in C++. The request returned in 0.00s having read nothing,
# and the API reported "no verifiable claims could be extracted" — an
# explanation that was not merely unhelpful but FALSE, since the text was never
# examined.
#
# The allowlist was unfixable in principle, not just incomplete. It also
# rejected percentages, ratios written with '/', colons, semicolons, en-dashes,
# curly quotes from any word processor, chemical formulae, mathematical notation
# and every non-Latin script. Enumerating the characters that may appear in a
# true statement about the world is not a finite task.
#
# Nothing downstream needed the protection either. The text goes to an LLM,
# which reads arbitrary characters, and to Wikipedia search, where
# ner._sanitize_query already strips the characters that break query parsers.
# What genuinely does not belong in a claim is a C0 control character — a NUL,
# a bell, an escape — which cannot appear in prose and can corrupt a log or a
# terminal. That is a short, closed list, so it is the one worth writing down.
_CONTROL_CHARACTERS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def validate_input(text: str) -> str | None:
    """
    Why this text cannot be processed, or None when it can.

    SPLIT OUT SO THE CALLER CAN TELL THE DIFFERENCE. precondition() returns None
    on rejection, run() then returns [], and main.py reported that as "no
    verifiable claims could be extracted" — conflating "this input was refused"
    with "this input was read and contained nothing checkable". Those are
    opposite situations: one is the user's to fix, the other is a finding about
    their text.
    """
    if not text or not text.strip():
        return "the input is empty"

    if len(text.split()) > MAX_INPUT_WORDS:
        return (f"the input is {len(text.split())} words, over the "
                f"{MAX_INPUT_WORDS}-word limit")

    if _CONTROL_CHARACTERS.search(text):
        return "the input contains control characters"

    return None


def precondition(text):
    text = re.sub(r'\s+', ' ', text).strip()

    reason = validate_input(text)
    if reason is not None:
        print(f"Validation Failed: {reason}.")
        return None

    print(f"Preprocessed text: {text}")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Call Ollama (shared helper)
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(prompt: str) -> str:
    """Send a prompt to the fast model and return the raw response string."""
    return llm.generate(
        FAST_MODEL, prompt,
        # No `think` field: phi3 has no reasoning mode, and the request
        # never carried one. Not logged: this runs many times per input.
        log=False,
        options={
            # Explicit context window to prevent 500s on long iterative prompts.
            "num_ctx": 4096,
            # Greedy decoding. Ollama defaults to temperature 0.8 / top_p 0.9,
            # which is actively harmful here: extraction is a copying task, so
            # the faithful span is the argmax and every unit of sampling noise
            # is an opportunity for the model to drift toward the fluent,
            # world-consistent paraphrase it prefers over the literal claim.
            # Fixing the seed also makes runs reproducible, without which no
            # prompt change can be evaluated.
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "repeat_penalty": 1.0,
            "seed": 0,
        },
    ).text


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Parse numbered list from model output (STRICT format contract)
# ─────────────────────────────────────────────────────────────────────────────
# Sentinel distinguishing "model signaled Terminate" (stop the loop) from
# "response was malformed" (reject this iteration, but keep looping).
# The old parser conflated both into None, and its loose regex matched the
# word 'Fact' anywhere in the response — which let leaked chain-of-thought
# (e.g. '...we proceed with Fact-2 extraction') be accepted as a claim.
TERMINATE = "__TERMINATE__"

def parse_fact_from_response(response: str, expected_idx: int) -> str | None:
    """
    Strictly parse one atomic fact from model output.

    Accepted formats (nothing else):
        - A line that is exactly 'Terminate' (optional trailing punctuation)
          → returns TERMINATE sentinel.
        - A line starting with 'Fact_<expected_idx>:' (underscore/dash/space
          separators tolerated) → returns the fact text after the colon.

    Anything else → None (malformed; caller rejects the iteration).
    Requiring the *expected* index prevents the model from re-emitting an
    earlier fact or inventing out-of-sequence ones.
    """
    # PRECEDENCE: a valid Fact_N line wins over Terminate. phi3 sometimes
    # emits both in one response ("Fact_2: ... \n ... \n Terminate");
    # checking Terminate first threw the fact away and closed the loop
    # with the input only partially covered.
    match = re.search(
        rf'(?im)^\s*Fact[_\s\-]?{expected_idx}\s*[:\-]\s*(.+?)\s*$',
        response,
    )
    if match:
        fact = match.group(1).strip()

        # Strip trailing parenthetical meta-commentary phi3 likes to append,
        # e.g. "... in 1889. (The entity ... extracted from the text.)" —
        # it pollutes the NER query and the verifier prompt.
        fact = re.sub(r'\s*\([^()]*\)[\s.]*$', '', fact).strip()

        if fact:
            return fact

    # Terminate must be a standalone line — not merely the substring
    # 'terminate' anywhere, which false-positived on claims containing it.
    if re.search(r'(?im)^\s*terminate[.!]?\s*$', response):
        print("[Extractor] Termination signaled. Loop closed successfully.")
        return TERMINATE

    return None  # malformed output — no fallback acceptance


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build extraction prompt  (mirrors Figure 3 in the AFEV paper)
# ─────────────────────────────────────────────────────────────────────────────
def build_extraction_prompt(
    original_text: str,
    extracted_facts: list[str],
    verification_results: list[str],
    verification_rationales: list[str],  # Crucial addition from Eq. 2
    rejected_attempts: list[tuple[str, str, str]] | None = None,
    target_sentence: str | None = None,
) -> str:
    """
    Dynamic iterative extraction prompt that mirrors Section 4.1 & 5.7 of AFEV.

    `rejected_attempts` carries (attempt_text, foreign_words) pairs for outputs
    the faithfulness gate refused this round. It exists because the extractor
    now runs at temperature 0: greedy decoding is a deterministic function of
    the prompt, so re-sending an unchanged prompt after a rejection reproduces
    the rejected output *exactly*, forever. One observed run burned iterations
    2 through 10 re-emitting the same sentence verbatim. Feeding the rejection
    back is what makes the retry a different question rather than a replay —
    and it turns the gate from a silent filter into a correction signal.

    `target_sentence` narrows what may be extracted while leaving the full
    document visible. The two roles are deliberately distinct:

        original_text   — CONTEXT. Read-only. Exists so pronouns resolve.
        target_sentence — TARGET.  The only source of new facts this round.

    Both are needed. Scoping to a sentence bounds the decomposition problem and
    stops one uncooperative sentence from consuming the whole budget, but a
    sentence read in isolation is often unverifiable: "You can walk between
    them with Yavuz Sultan Selim Bridge" has no recoverable meaning without the
    sentence before it. Keeping the document in view is what lets "them"
    resolve while the extraction stays scoped.
    """
    # Each rejection carries the KIND of defect, because the advice that fixes
    # one makes the others worse. An earlier version rendered every rejection
    # with the faithfulness wording, so a claim rejected for an unresolved
    # pronoun was told "these words do not appear in the text: 'its' refers to
    # something outside the claim" — an assertion that is both false ('its' is
    # in the text) and self-contradictory. phi3 responded by re-emitting the
    # identical sentence three times, which under greedy decoding is exactly
    # what incoherent feedback should be expected to produce.
    _ADVICE = {
        "faithfulness": "Rewrite using ONLY wording that appears in the document.",
        "decontextualization": "Replace the reference with the name it stands for, "
                               "taken from the document.",
        "fluency": "Write one complete, grammatical declarative sentence.",
        "negation": "Keep the polarity of the source. If the text states something "
                    "positively, state it positively; if it states something "
                    "negatively, keep the negation. Never insert 'not', 'no' or "
                    "'never', and never remove one.",
        "malformed": "Respond with exactly one line: 'Fact_N: <sentence>' or 'Terminate'.",
    }

    rejection_block = ""
    if rejected_attempts:
        lines = []
        for attempt, kind, detail in rejected_attempts:
            if kind == "faithfulness":
                reason = f"These words are not in the document: {detail}."
            elif kind == "malformed":
                reason = "This was not in the required output format."
            else:
                reason = detail if detail.endswith(".") else f"{detail}."
            lines.append(f'- You wrote: "{attempt}"\n  REJECTED. {reason}')

        # Only advise on defects actually seen, so the model is not handed four
        # competing instructions when it made one mistake.
        advice = [_ADVICE[k] for k in dict.fromkeys(k for _, k, _ in rejected_attempts)]

        rejection_block = (
            "\nFAILED ATTEMPTS — do not repeat these:\n"
            + "\n".join(lines)
            + "\n" + " ".join(advice)
            + " If nothing extractable remains, output Terminate.\n"
        )

    # Presentation of the source material. When scoped to a sentence the
    # document is shown first and explicitly demoted to context, so the model
    # can resolve references without treating the whole document as fair game.
    scoped = bool(target_sentence) and target_sentence.strip() != original_text.strip()
    if scoped:
        text_block = (
            'Full document — CONTEXT ONLY. Use it to resolve pronouns and vague\n'
            'references. Do NOT extract facts from it.\n'
            f'\"\"\"{original_text}\"\"\"\n\n'
            'TARGET SENTENCE — extract only from this sentence:\n'
            f'\"\"\"{target_sentence}\"\"\"'
        )
        scope_rule = (
            "- SCOPE: Extract only what the TARGET SENTENCE asserts. If it contains a pronoun "
            "or vague reference, replace it with the matching wording from the full document.\n"
        )
    else:
        text_block = f'Text:\n\"\"\"{original_text}\"\"\"'
        scope_rule = ""

    # 1. Base Strategy: First Iteration
    if not extracted_facts:
        return f"""You are an expert Atomic Fact Extractor. Your task is to extract the VERY FIRST single, verifiable factual unit from the text.

Rules:
- Isolate one checkable fact expressing a single subject-relation-object relationship.
- Write the fact as ONE plain declarative sentence. Do NOT use arrow notation (->), quotation marks, or parenthetical commentary. Do NOT explain your reasoning.
- FAITHFULNESS — MOST IMPORTANT RULE: Use ONLY information stated in the text. Copy names, dates, and places exactly as they are written. NEVER add dates, names, numbers, or details from your own memory, even if you are certain they are true. If the text says "in 1889", write "in 1889" — never a more specific date.
- DECONTEXTUALIZATION: the fact must stand alone. Replace every backward-pointing reference (it, they, them, he, she, this, that city, there) with the name it stands for, exactly as that name appears in the text. Generic "you" or "one" refers to nobody in particular — leave it as it is. If a reference has no antecedent in the text, leave it unchanged rather than inventing one.
{scope_rule}- Output exactly ONE line in this format and nothing else: Fact_1: <extracted fact>

Example of a CORRECT extraction:
Text: \"\"\"Marie Curie, who was born in Warsaw, won two Nobel Prizes.\"\"\"
Fact_1: Marie Curie was born in Warsaw.

Example of an INCORRECT extraction — never do this:
Text: \"\"\"Marie Curie, who was born in Warsaw, won two Nobel Prizes.\"\"\"
Fact_1: Marie Curie was born on 7 November 1867 in Warsaw, Poland.
This is wrong because the exact date and the country are NOT in the text — they were added from memory.

{text_block}
{rejection_block}
Extract Fact_1:"""

    # 2. Dynamic Strategy: Subsequent Iterations (The Feedback Loop)
    #
    # Built with a list comprehension + str.join instead of repeated `+=`.
    # Every `+=` on a str allocates a whole new string (str is immutable), so
    # the old loop was quadratic in the history length; join walks the parts
    # once and allocates the result buffer exactly once.
    #
    # strict=True makes a length mismatch between the three parallel lists a
    # loud ValueError instead of a silent truncation: plain zip() stops at the
    # shortest input, so one missing rationale would quietly drop a verified
    # fact out of the history the extractor reasons over.
    #
    # INCLUDE_RATIONALES_IN_HISTORY gates the Answer/Rationale lines. With them
    # on, an observed run answered a REFUTES rationale by proposing the
    # verifier's correction as the next "fact" — the rationale is the single
    # strongest pull away from faithful extraction in the system, because it
    # sits near the end of the prompt in fluent prose that contradicts the
    # source. With them off, the extractor still sees which parts of the text
    # are covered, which is all it needs to decide what remains.
    if INCLUDE_RATIONALES_IN_HISTORY:
        history_block = "\n\n".join(
            f"Fact_{i}: {fact}\nAnswer_{i}: {label}\nRationale_{i}: {rationale}"
            for i, (fact, label, rationale) in enumerate(
                zip(
                    extracted_facts,
                    verification_results,
                    verification_rationales,
                    strict=True,
                ),
                1,
            )
        )
    else:
        history_block = "\n".join(
            f"Fact_{i}: {fact}" for i, fact in enumerate(extracted_facts, 1)
        )

    next_idx = len(extracted_facts) + 1
    unit = "TARGET SENTENCE" if scoped else "Original Text"

    prompt = f"""You are an adaptive, iterative Atomic Fact Extractor (AFEV Framework).

{text_block}

Already Extracted (do not repeat these):
{history_block}

Instructions for Step-by-Step Contextual Extraction:
1. Analyze whether the 'Facts' listed above completely cover the full semantic meaning and all multi-hop dependencies of the {unit}.
2. If the {unit} is fully accounted for, output exactly: Terminate
3. If uncovered details remain, extract the next atomic fact (Fact_{next_idx}) FROM THE {unit}.
4. DECONTEXTUALIZATION — the claim must be understandable BY ITSELF, with no access to the document. Replace every backward-pointing reference (it, they, them, he, she, this, that club, the city, there, then) with the name it stands for, AS THAT NAME APPEARS IN THE DOCUMENT ABOVE. Also make time and place absolute: "last year" or "there" must become the actual year or place if the document states it. The replacement must be findable in the document — do NOT take names, identities, dates, or corrections from anywhere else. Generic "you" or "one" refers to nobody in particular and should be left as it is.
5. FAITHFULNESS — MOST IMPORTANT RULE: Every word of the fact must be traceable to the document. NEVER add dates, names, numbers, timeframes, or details from your own memory, even if you are certain they are true. The single allowed exception is rule 4, and its replacement text must itself come from the document.
6. OUTPUT FORMAT — STRICT: Respond with exactly ONE line. Either "Fact_{next_idx}: <one plain declarative sentence>" or "Terminate". NEVER output both. No Answer lines, no Rationale lines, no arrow notation (->), no parentheses, no explanations of any kind.
{rejection_block}
Extract Fact_{next_idx} or Terminate:"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Iterative extractor  (core logic from AFEV Section 4.1)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_from_sentence(
    document: str,
    sentence: str,
    doc_words: set[str],
    seen_claims: set[str],
    source_is_fluent: bool = True,
) -> list[dict]:
    """
    Run the AFEV decomposition loop over ONE sentence.

    `document` is passed through to the prompt as read-only context so
    references resolve across sentence boundaries, and `doc_words` — content
    words of the whole document, not just this sentence — is what the
    faithfulness gate checks against. That combination is deliberate: it lets
    "You can walk between them" become "...between Africa and Europe" while
    still rejecting anything the document never says.

    `seen_claims` is shared across sentences so a fact already extracted
    earlier is not re-verified at 150s a piece.
    """
    results: list[dict] = []

    # Extraction-LLM time accumulated since the last accepted fact —
    # rejected/malformed iterations are billed to the next accepted fact.
    pending_extraction_s = 0.0

    text_words = doc_words

    # Rejections for the CURRENT fact slot, as (attempt, foreign_words) pairs.
    # Fed back into the prompt so each retry asks a different question, and
    # cleared as soon as a fact is accepted.
    rejected_attempts: list[tuple[str, str, str]] = []   # (attempt, kind, detail)

    for iteration in range(MAX_ITERATIONS_PER_SENTENCE):
        print(f"\n[Extractor]   Iteration {iteration + 1}")

        if len(rejected_attempts) >= MAX_CONSECUTIVE_REJECTIONS:
            print(
                f"[Extractor]   {len(rejected_attempts)} consecutive rejections — "
                f"giving up on this sentence and moving to the next."
            )
            break

        # Build prompt using the full historical feedback loop
        prompt = build_extraction_prompt(
            document,
            [r["claim"] for r in results],
            [r["label"] for r in results],
            [r["rationale"] for r in results],
            rejected_attempts,
            target_sentence=sentence,
        )

        t_llm = time.perf_counter()
        response = call_ollama(prompt)
        pending_extraction_s += time.perf_counter() - t_llm
        print(f"[Extractor] Model response: {response}")

        fact = parse_fact_from_response(response, len(results) + 1)

        if fact == TERMINATE:
            print("[Extractor]   Sentence fully covered.")
            break

        if fact is None:
            # Malformed output (no strict 'Fact_N:' line, no 'Terminate').
            # Reject this iteration rather than accepting garbage as a claim;
            # the MAX_ITERATIONS ceiling still bounds the loop.
            print("[Extractor]   Rejected malformed response — retrying iteration.")
            rejected_attempts.append(
                (response.strip()[:200], "malformed", "no 'Fact_N:' line")
            )
            continue

        # Duplicate gate: a re-emitted fact means the model believes coverage
        # is complete but failed to output 'Terminate'. Treat it as termination
        # instead of burning the remaining iterations on repeats (the Eiffel
        # test run emitted the same fabricated claim twice, verified both).
        # Checked against every claim seen so far in the DOCUMENT, not just
        # this sentence — two sentences restating the same fact should not cost
        # two full retrieval+verification cycles.
        normalized = _normalize_claim(fact)
        if normalized in seen_claims:
            print(f"[Extractor]   Duplicate fact re-emitted — treating as termination: '{fact}'")
            break

        # Faithfulness gate — containment, not overlap.
        #
        # The previous version rejected a fact only when it shared ZERO content
        # words with the input, which is far too weak to catch the failure mode
        # that matters: the extractor silently *correcting* the claim. Given
        # "The Bosporus separates Africa and Europe", phi3 returned "The
        # Bosporus separates Asia Minor and Thrace" — it substituted what it
        # knows to be true for what the text actually asserts. That fact still
        # shared 'bosporus' and 'separates' with the input, so the old gate
        # passed it. Verification then checked the corrected claim, returned
        # SUPPORTS, and the pipeline lost the very error it exists to detect.
        #
        # So the requirement is now containment: every content word of the fact
        # must trace back to the source (modulo function words and inflection).
        # Any residue is material the model brought in from its own knowledge.
        # False rejections are cheap — the iteration is retried — while a false
        # acceptance corrupts the verdict, so the asymmetry favours strictness.
        # Fluency gate (Ullrich et al. 2025, §4.1) — a garbled claim produces a
        # garbled retrieval query and an uninterpretable verdict. Rejecting
        # costs one extraction call (~19s); admitting costs ~152s downstream.
        #
        # The escape hatch matters: if the SOURCE sentence is itself
        # ungrammatical (typo, missing verb), then a faithful extraction is
        # ungrammatical too. Without this, the gate would reject it, the model
        # would faithfully reproduce it, and after three rounds the sentence
        # would be silently dropped — making sloppily written input
        # uncheckable. Faithfulness outranks fluency.
        if source_is_fluent:
            defect = ner.check_fluency(fact)
            if defect:
                print(f"[Extractor]   Rejected disfluent claim — {defect} | '{fact}'")
                rejected_attempts.append((fact, "fluency", defect))
                continue

        # Decontextualization gate (AIDA 'Independent'). Runs BEFORE the
        # faithfulness gate on purpose: resolving 'them' to 'Africa and Europe'
        # pulls words from the document, which the faithfulness check then
        # validates. Reversing the order would let an unresolved claim through
        # simply because it borrowed nothing.
        dangling = ner.check_decontextualized(fact)
        if dangling:
            print(f"[Extractor]   Rejected context-dependent claim — {dangling} | '{fact}'")
            rejected_attempts.append((fact, "decontextualization", dangling))
            continue

        foreign = _content_words(fact) - text_words - FUNCTION_WORDS
        if foreign:
            print(
                f"[Extractor]   Rejected unfaithful fact — words absent from the "
                f"document: {sorted(foreign)} | '{fact}'"
            )
            rejected_attempts.append((fact, "faithfulness", ", ".join(sorted(foreign))))
            continue

        # POLARITY GATE. Runs immediately after containment because it checks
        # the half of faithfulness that containment structurally cannot see:
        # _content_words drops every token shorter than
        # MIN_CONTENT_WORD_LENGTH (4), and 'not' is three characters.
        #
        # Observed failure on this very input:
        #
        #   sentence: "It is universally acknowledged as longer than the Nile
        #              by all international cartographers."      ('It' = Amazon)
        #   claim:    "The Nile is not universally acknowledged as the world's
        #              longest river."
        #
        # Wrong subject, inserted negation, meaning reversed — and every gate
        # passed it, because each surviving content word does appear in the
        # document and the 'not' was filtered out before comparison. The
        # verifier then REFUTED the fabrication, and the run reported a
        # confident REFUTES on a claim the input never made. The system did not
        # merely fail to catch an error; it manufactured one.
        #
        # Compared against the SENTENCE, not the document. A negation elsewhere
        # in the text licenses nothing here — that licence is exactly what the
        # failure above took.
        flipped = ner.check_negation(sentence, fact)
        if flipped:
            print(f"[Extractor]   Rejected polarity change — {flipped} | '{fact}'")
            rejected_attempts.append((fact, "negation", flipped))
            continue

        # Check-worthiness gate. Unlike the two above this does NOT retry:
        # re-extracting from an opinion yields another opinion. The claim is
        # recorded so the user sees it was considered and why, marked so it is
        # distinguishable from a claim that was checked and found wanting.
        #
        # Kept as nei_kind rather than a new top-level label, because
        # postprocessor.normalize_label() coerces anything outside
        # {SUPPORTS, REFUTES, NOT ENOUGH INFO} back to NOT ENOUGH INFO and the
        # frontend has no CSS class for a fourth value. Promoting UNVERIFIABLE
        # to a first-class label is a small change in both files — worth doing,
        # but it touches the UI, so it is not done unilaterally here.
        unworthy = ner.check_worthy(fact)
        if unworthy:
            print(f"[Extractor]   Skipping unverifiable claim — {unworthy} | '{fact}'")
            results.append({
                "claim"      : fact,
                "label"      : "NOT ENOUGH INFO",
                "nei_kind"   : "UNVERIFIABLE",
                "rationale"  : f"Not sent for verification because {unworthy}.",
                "evidence"   : "",
                "ner_query"  : "",
                "source_url" : "",
                "source_sentence": sentence,
                "timings"    : {
                    "retrieval_s"    : 0.0,
                    "verification_s" : 0.0,
                    "extraction_s"   : round(pending_extraction_s, 2),
                },
            })
            pending_extraction_s = 0.0
            seen_claims.add(normalized)
            rejected_attempts.clear()
            continue

        # Evidence-grounded verification (replaces the world-knowledge mock).
        # The verified label/rationale feed the next iteration's history AND
        # constitute the final result for this fact.
        result = verification.verify_claim(fact)
        result["timings"]["extraction_s"] = round(pending_extraction_s, 2)
        pending_extraction_s = 0.0
        # Provenance: which input sentence this verdict came from, so the UI
        # can highlight the offending sentence rather than the whole passage.
        result["source_sentence"] = sentence
        results.append(result)
        seen_claims.add(normalized)
        rejected_attempts.clear()   # fresh slate for the next fact slot

        print(f"[Extractor]   Accepted: {fact}")
        print(f"[Extractor]   Grounded feedback -> {result['label']} | {result['rationale']}")

    return results


def extract_atomic_facts(text: str) -> list[dict]:
    """
    Per-sentence AFEV extraction with evidence-grounded verification.

    Each sentence gets its own decomposition budget. Previously one loop ran
    over the whole document, which meant a sentence the model would not
    decompose faithfully consumed every remaining iteration: an observed run
    spent all its retries trying to restate sentence 1 as "between Asia and
    Europe" and never examined sentence 2 at all. Scoping per sentence turns
    that from a document-wide failure into a local one.

    Returns a flat list of per-fact result dicts (see results.FactResult), each
    tagged with the sentence it came from.
    """
    sentences = ner.split_sentences(text)
    if not sentences:
        return []

    doc_words = _content_words(text)
    seen_claims: set[str] = set()
    results: list[dict] = []

    for idx, sentence in enumerate(sentences, 1):
        print(f"\n[Extractor] Sentence {idx}/{len(sentences)}: '{sentence}'")

        # MAX_ITERATIONS is now a whole-document ceiling rather than the only
        # bound; without it, a long input could still run unboundedly long.
        if len(results) >= MAX_ITERATIONS:
            print("[Extractor] Document-wide fact ceiling reached — stopping.")
            break

        # If the input sentence is itself malformed, disable the fluency gate
        # for claims drawn from it — see the escape hatch note in the loop.
        source_defect = ner.check_fluency(sentence)
        if source_defect:
            print(f"[Extractor] Source sentence is not well-formed ({source_defect}); "
                  f"fluency gate disabled for it.")

        sentence_results = _extract_from_sentence(
            text, sentence, doc_words, seen_claims,
            source_is_fluent=source_defect is None,
        )
        print(f"[Extractor] Sentence {idx} yielded {len(sentence_results)} fact(s).")

        # A SENTENCE THAT YIELDS NOTHING IS A FAILURE, NOT AN ABSENCE.
        #
        # Without this branch the sentence simply vanishes: extraction prints
        # "yielded 0 fact(s)", moves on, and the request reports an overall
        # verdict computed from whatever the other sentences produced. In the
        # output there is no difference at all between
        #
        #     "this sentence contains no checkable claim"   — fine
        #     "extraction broke and the claim was lost"      — the founding bug
        #
        # Observed on:
        #
        #     "His forces single-handedly defeated Napoleon's French Army and
        #      ended the Napoleonic Wars."
        #
        # phi3 returned "His forces defeated..." three times, the
        # decontextualisation gate correctly rejected the dangling 'His' each
        # time, and the sentence was dropped in silence. That sentence carried
        # 'single-handedly' — the only false word in the input — and the request
        # answered `overall: SUPPORTS` from the one sentence that survived.
        #
        # This does not repair the extraction failure. It makes it VISIBLE,
        # which has to come first: a bug that leaves no trace in the output
        # cannot be measured, and a fix for it cannot be evaluated.
        #
        # Labelled NOT ENOUGH INFO because postprocessor.normalize_label()
        # coerces anything outside the three FEVER labels back to it anyway;
        # nei_kind carries the real distinction, as UNVERIFIABLE already does.
        if not sentence_results:
            print(
                f"[Extractor] EXTRACTION FAILURE on sentence {idx} — no claim was "
                f"produced. Recorded, not discarded: the sentence may have carried "
                f"an assertion that never reached verification."
            )
            results.append({
                "claim"      : sentence,
                "label"      : "NOT ENOUGH INFO",
                "nei_kind"   : "EXTRACTION_FAILURE",
                "rationale"  : (
                    "No atomic claim could be extracted from this sentence, so it "
                    "was never verified. The verdict describes the extractor, not "
                    "the sentence — treat this as unchecked, not as unsupported."
                ),
                "evidence"   : "",
                "ner_query"  : "",
                "source_url" : "",
                "source_sentence": sentence,
                "timings"    : {
                    "retrieval_s": 0.0, "verification_s": 0.0, "extraction_s": 0.0,
                },
            })
            continue

        results.extend(sentence_results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run(text: str) -> list[dict]:
    """
    Full claim extraction + verification pipeline:
    precondition → iterative extraction with evidence-grounded feedback
    → list of verified per-fact result dicts for postprocessor.process().
    """
    clean_text = precondition(text)
    if clean_text is None:
        return []

    results = extract_atomic_facts(clean_text)

    print(f"\n[Extractor] Done. {len(results)} verified atomic fact(s):")
    for i, r in enumerate(results, 1):
        suffix = ""
        if r.get("nei_kind") == "RETRIEVAL_FAILURE":
            suffix = "  <- evidence never mentioned the subject; verdict not reached"
        elif r.get("nei_kind") == "GENUINE":
            suffix = "  <- relevant evidence retrieved but inconclusive"
        elif r.get("nei_kind") == "UNVERIFIABLE":
            suffix = "  <- not check-worthy; never sent for verification"
        elif r.get("nei_kind") == "EXTRACTION_FAILURE":
            suffix = "  <- EXTRACTION FAILED; sentence never became a claim"
        print(f"  {i}. [{r['label']}] {r['claim']}{suffix}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(text_ba)