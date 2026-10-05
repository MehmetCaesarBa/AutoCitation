"""
program.py — program-guided verification for claims the single-shot verifier
cannot decide.

INSPIRED BY, NOT FAITHFUL TO, ProgramFC (Pan et al., ACL 2023,
arXiv:2305.12744). The paper generates reasoning programs with an LLM using
in-context demonstrations. Here the programs are SYNTHESISED FROM THE PARSE,
deterministically, with no extra inference. That is a deliberate divergence and
the thesis should describe it as such.

    the paper's way   general, handles claims nobody anticipated, needs a
                      capable LLM to emit valid programs, one extra call per
                      claim (30-120s on this hardware), and a parser plus a
                      fallback for malformed output.

    this way          covers exactly the operator classes that appear in the
                      failure logs, costs no inference, and is unit-testable
                      offline against a fixed parse.

WHY A PROGRAM AT ALL. The single-shot verifier is asked "does this evidence
support this claim?" — one question, one answer. For two claim types that
question is the wrong one, and no prompt wording repairs it:

  SUPERLATIVE   "Jamestown is the earliest European settlement in the US."
                Evidence about Jamestown cannot refute this, because the
                refutation is a passage about SAINT AUGUSTINE, which does not
                contain the word 'Jamestown' and therefore never ranks. Three
                verifier prompt edits produced three different verdicts on
                identical evidence — the signature of an under-determined
                question rather than a wording problem.

                The program asks a different question: WHO ACTUALLY WAS FIRST?
                Then it compares that answer to the claim's subject. Retrieval
                is aimed at the category, not the subject, so the evidence that
                settles it is reachable.

  COMPARATIVE   "Mount Everest is taller than K2."
                Needs two numbers from two articles and an arithmetic
                comparison. The pipeline has no stage where two facts meet, and
                an 8B model comparing 8849 against 8611 is unreliable because a
                tokenizer splits numbers into arbitrary pieces. The program
                retrieves each side, extracts the quantity, and compares in
                Python — where comparison is exact and the working is citable.

WHAT IT DELIBERATELY DOES NOT DO. Return None for anything it does not
recognise. Most claims carry no operator, the existing path handles them, and a
program that fires on everything would be a rewrite of the verifier rather than
an addition to it. Every uncertain branch resolves to NOT ENOUGH INFO rather
than to a guess: this module can only ever move a verdict when it is confident,
which keeps its failure mode "no help" instead of "new wrong answers".
"""

import functools
import re
from dataclasses import dataclass, field

import Pipeline.ner as ner
import Pipeline.retriever as retriever
from models import llm
from Pipeline.results import VerificationResult

# ── Config ────────────────────────────────────────────────────────────────────
# Master switch. False restores single-shot verification for every claim, which
# is the comparison baseline — the claim "programs beat single-shot on
# superlatives" should be reproducible, not asserted.
PROGRAM_MODE = True

ANSWER_MODEL = "qwen3:8b"
KEEP_ALIVE = "30m"

# The Question() handler was on phi3:mini, on the reasoning that "which name does
# this passage give for X?" is a copying task the fast model can do. Live runs
# said otherwise: with the right articles retrieved it answered 'Natchitoches'
# for the earliest European settlement in the US, and 'Perucetus colossus' then
# 'Bruhathkayosaurus' for the most gigantic creature in history — three wrong
# picks in three answers. Choosing WHICH name answers the question is a reading
# comprehension judgement, not copying, so it now runs on the verifier's model.
# Thinking is disabled (see _call_answer_model): the answer is one name, and a
# <think> block would spend the whole ANSWER_NUM_PREDICT budget before it.
#
# The safety net is not the model's care, it is the containment check in
# _answer_is_grounded: an answer whose words do not appear in the evidence is
# discarded. That makes hallucination a deterministic rejection rather than a
# probabilistic hope.
ANSWER_NUM_PREDICT = 48

# How many chunks the Question handler reads. Higher than the verifier's budget
# because this is a lookup over a category rather than a judgement about one
# sentence — the answer may sit outside the top two.
QUESTION_TOP_K = 4

# The kind check (_answer_is_right_kind) is an NLI cross-encoder, not an Ollama
# model: it classifies (evidence sentence, hypothesis) pairs as entailment /
# neutral / contradiction and generates nothing. Trained on MNLI, FEVER and
# ANLI — FEVER is itself Wikipedia fact-checking, which is this pipeline's task.
# Runs on CPU through transformers; downloaded from Hugging Face on first use.
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Minimum P(entailment) for an evidence window to count as stating the
# hypothesis. Below it, the answer is rejected and the claim goes to NOT ENOUGH
# INFO — the safe direction for this module's failure mode.
ENTAILMENT_THRESHOLD = 0.5


# ── Operator detection ────────────────────────────────────────────────────────
# Comparative markers. JJR/RBR are the comparative tags ('taller', 'longer',
# 'more'). A comparative alone is not enough — "the river is longer now than in
# 1970" compares one thing to itself across time — so a comparison also requires
# a 'than' with a nominal object to compare AGAINST.
_COMPARATIVE_TAGS = {"JJR", "RBR"}

# Dimension lexicon: which quantity a comparative adjective is about, and the
# family of units that expresses it. Without this, "taller" and "longer" are
# indistinguishable strings and the extractor cannot tell which of an article's
# many numbers is the relevant one.
#
# Deliberately small. Each entry is a claim about what a word measures, and a
# wrong entry produces a confident comparison of unrelated quantities — far
# worse than declining. Ambiguous adjectives ('larger', 'bigger', 'greater')
# are EXCLUDED for exactly that reason: larger by area, population, or volume
# is not decidable from the adjective.
_DIMENSIONS = {
    "tall":     ("height", "length"),
    "high":     ("height", "length"),
    "long":     ("length", "length"),
    "deep":     ("depth", "length"),
    "wide":     ("width", "length"),
    "populous": ("population", "count"),
    "old":      ("age", "year"),
}

# Unit conversion to a canonical base. Length -> metres, count -> units,
# year -> year. Only units that actually appear in encyclopedia prose.
_UNIT_TO_BASE = {
    # length family -> metres
    "mm": 0.001, "cm": 0.01, "m": 1.0, "metre": 1.0, "metres": 1.0,
    "meter": 1.0, "meters": 1.0, "km": 1000.0, "kilometre": 1000.0,
    "kilometres": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}

_UNIT_FAMILY = {u: "length" for u in _UNIT_TO_BASE}


# ── Program representation ────────────────────────────────────────────────────
@dataclass(frozen=True)
class Step:
    """One instruction. `op` names the handler, `args` are its inputs."""
    var: str
    op: str
    args: tuple


@dataclass
class Program:
    """
    A synthesised reasoning program plus the bindings produced by executing it.

    `trace` is not decoration. ProgramFC's stated advantage over end-to-end
    verification is that the reasoning is inspectable and debuggable, and a
    program whose intermediate answers are invisible gives that up. The trace
    becomes part of the rationale shown to the user, so a verdict can be argued
    with rather than merely accepted.
    """
    kind: str
    steps: list[Step]
    bindings: dict = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    source_url: str = ""
    query: str = ""
    # The category the superlative ranks within, read from the CLAIM's parse at
    # synthesis time. Carried on the program because execute() has the claim and
    # _handle_question does not — see ner.superlative_type_word for why deriving
    # it from `query` instead is unreliable.
    type_word: str = ""


@dataclass(frozen=True)
class ProgramOutcome:
    """
    What a program run produces: a verdict, plus where it came from.

    Returned instead of a finished per-fact record. Building that record, and
    timing the work, is verification.verify_claim's job for BOTH paths — this
    module used to build its own copy "shaped like grounded_verify's" and fill
    the timings with zeros for the caller to overwrite.
    """
    verdict: VerificationResult
    kind: str
    query: str
    source_url: str


# ── Synthesis ─────────────────────────────────────────────────────────────────
def synthesize(claim: str) -> Program | None:
    """
    Build a program for `claim`, or None when no operator is recognised.

    Superlative is tested BEFORE comparative. "The Amazon is the longest river"
    contains 'longest' (JJS) and would also satisfy a loose comparative test;
    the superlative reading is the correct one, and running the comparative
    handler on it would look for a second entity that does not exist.
    """
    doc = ner.NLP(claim)

    superlative = _find_superlative(doc)
    if superlative is not None:
        return _superlative_program(claim, doc, superlative)

    comparative = _find_comparative(doc)
    if comparative is not None:
        return _comparative_program(claim, doc, *comparative)

    return None


def _find_superlative(doc):
    """
    The superlative or ordinal token, if the claim has one.

    SHARES ner.is_superlative RATHER THAN REPEATING ITS TEST. The two used the
    same two conditions written out twice, which is how they came to disagree:
    'First' inside "The First Transcontinental Railroad" is PROPN, and the fix
    that taught ner.build_predicate_query to ignore it would have left this
    function still routing the claim down the program path — synthesising a
    superlative program for a claim whose predicate query had just been refused.
    One definition, one behaviour.
    """
    for token in doc:
        if ner.is_superlative(token):
            return token
    return None


def _find_comparative(doc):
    """
    (comparative_token, compared_entity) — or None.

    Requires BOTH a comparative tag and a 'than' phrase with a nominal object.
    "longer than the Nile" qualifies; "the river is longer now" does not, and
    treating it as a comparison would send the extractor looking for a second
    entity the claim never names.
    """
    comp = next((t for t in doc if t.tag_ in _COMPARATIVE_TAGS), None)
    if comp is None:
        return None

    than = next((t for t in doc if t.text.lower() == "than"), None)
    if than is None:
        return None

    # The compared entity is the nominal governed by 'than'. Taking its whole
    # subtree keeps multi-word names intact ('the Nile', 'Mount Kilimanjaro').
    for token in doc:
        if token.head.i == than.i or (token.i > than.i and token.pos_ in ("PROPN", "NOUN")):
            span = _noun_span(token)
            if span:
                return comp, span
    return None


def _noun_span(token) -> str:
    """Contiguous noun phrase around `token`, articles stripped."""
    doc = token.doc
    idx = sorted(
        t.i for t in token.subtree
        if t.pos_ in ("PROPN", "NOUN", "ADJ", "NUM") and not t.is_stop
    )
    if not idx:
        return ""
    text = doc[idx[0]: idx[-1] + 1].text
    return re.sub(r'^(the|a|an)\s+', '', text, flags=re.I).strip()


def _superlative_program(claim: str, doc, superlative) -> Program | None:
    """
    answer_1 = Question(<category query>)
    label    = Match(answer_1, subject)

    The whole point is in the first line: the query names the CATEGORY the
    superlative ranks within, not the claim's subject. Asking "what was the
    earliest European settlement in the US?" reaches Saint Augustine; asking
    about Jamestown never can.
    """
    subject = ner.extract_subject(claim)
    category = ner.build_predicate_query(claim)

    # Both are required. Without a subject there is nothing to compare the
    # answer against; without a category query the retrieval would fall back to
    # the subject and reproduce the failure this exists to fix.
    if not subject or not category:
        return None

    return Program(
        kind="superlative",
        steps=[
            Step("answer_1", "Question", (category,)),
            Step("label", "Match", ("answer_1", subject)),
        ],
        query=category,
        # Read from `claim`, not from `category`. The category query has had its
        # prepositions stripped, which is exactly the information needed to tell
        # the ranked type from the scope it is ranked within.
        type_word=ner.superlative_type_word(claim) or "",
    )


def _comparative_program(claim: str, doc, comp, other: str) -> Program | None:
    """
    q_1   = Quantity(subject, dimension)
    q_2   = Quantity(other,   dimension)
    label = Compare(q_1, q_2)

    Arithmetic happens in Python. A tokenizer splits '8849' into pieces that
    carry no magnitude, so asking an 8B model whether 8849 exceeds 8611 is a
    text-pattern question wearing a number's clothes. Python does not have that
    problem, and the two source values can be shown in the rationale.
    """
    subject = ner.extract_subject(claim)
    if not subject:
        return None

    dimension = _DIMENSIONS.get(comp.lemma_.lower())
    if dimension is None:
        # An unmapped comparative ('larger', 'better', 'more important') names
        # no measurable dimension, or an ambiguous one. Decline rather than
        # compare whatever numbers happen to be nearby.
        return None

    name, family = dimension
    return Program(
        kind="comparative",
        steps=[
            Step("q_1", "Quantity", (subject, name, family)),
            Step("q_2", "Quantity", (other, name, family)),
            Step("label", "Compare", ("q_1", "q_2")),
        ],
        query=f"{subject} {other}",
    )


# ── Handlers ──────────────────────────────────────────────────────────────────
def _retrieve(query: str, top_k: int,
              rank_against: str | None = None) -> tuple[list[str], str]:
    """
    Chunks for a free-text query, ranked by retriever.rank_chunks — the same
    scorer and selection the verifier path uses.

    `query` selects the ARTICLES. `rank_against` selects the CHUNKS within them,
    defaulting to `query` so every existing caller is unaffected.

    They differ only on _handle_question's fallback path, where the subject name
    fetches the right article but ranks its chunks near-uniformly — the subject
    appears in most of them, so it discriminates nothing. The chunk that answers
    is the one matching the CATEGORY, so fetch by subject and rank by category.
    """
    articles = retriever.fetch_articles(query)
    if not articles:
        return [], ""
    chunks = retriever.chunk_articles(articles)
    # Ranked exactly as the verifier path ranks its evidence — see
    # retriever.rank_chunks for why this must never be a separate scorer again.
    return retriever.rank_chunks(rank_against or query, chunks, top_k=top_k)


def _call_answer_model(prompt: str) -> str:
    """
    One Question() inference through the shared llm.generate.

    THE OLD REASON FOR NOT SHARING CODE HERE. The first version of this module
    called ollama_client.log_inference_stats, which a revert removed, and the
    AttributeError took down every program run. llm.py exists to be the small,
    stable dependency that ollama_client was not, and
    test_every_cross_module_call_actually_exists now walks references into it
    as well, so a renamed helper fails a test rather than a 700-second run.
    """
    return llm.generate(
        ANSWER_MODEL, prompt,
        # The answer is one name; a <think> block would spend the whole
        # ANSWER_NUM_PREDICT budget before it.
        think=False,
        keep_alive=KEEP_ALIVE,
        options={
            "num_ctx": 4096,
            "num_predict": ANSWER_NUM_PREDICT,
            "temperature": 0, "top_p": 1, "top_k": 1,
            "repeat_penalty": 1.0, "seed": 0,
        },
    ).text


def _answer_is_grounded(answer: str, chunks: list[str]) -> bool:
    """
    Does every content word of the answer appear in the evidence?

    THE ONLY THING STANDING BETWEEN THIS MODULE AND A FABRICATED VERDICT. The
    Question handler runs on an LLM, which knows a great deal about early
    American settlements and would happily answer from memory. An answer drawn
    from weights rather than from the retrieved text would then be compared
    against the claim's subject and could produce a confident REFUTES with a
    citation to a passage that never said it.

    Containment makes that a deterministic rejection. Same principle as the
    extractor's faithfulness gate, same reason.
    """
    haystack = " ".join(chunks).lower()
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) >= 4]
    if not words:
        return False
    return all(w in haystack for w in words)


def _handle_question(query: str,
                     fallback_queries: tuple[str, ...] = (),
                     type_word: str = "",
                     claim: str = "",
                     subject: str = ""
                     ) -> tuple[str | None, list[str], str, bool]:
    """
    Question(query), retried against `fallback_queries` only if it finds nothing.

    Fourth return value is `from_fallback`: True when the answer came from a
    fallback query rather than from `query` itself.

    A FALLBACK, DELIBERATELY NOT A POOL. ner.build_predicate_query's docstring
    says the predicate query is "an ADDITIONAL query, never a replacement", and
    ner.build_query does append it beside the entity queries — so the verifier
    path already works that way. _superlative_program is the path that does not:
    it sets query=category and _retrieve runs one fetch, so the subject's own
    article is never read except by _handle_corroborate on the REFUTES branch.

    Wiring the subject query in as documented is right. Wiring it in by POOLING
    both into this step's evidence is not, and would undo the reason the
    predicate query exists. That query's whole job is to reach the RIVAL —
    Saint Augustine for the Jamestown claim — and it works BECAUSE the subject's
    article is absent. Pool it back in and the model reads Jamestown-centric
    prose, answers 'Jamestown', and _handle_match returns SUPPORTS: the precise
    false SUPPORTS this module was built to prevent, readmitted through the back
    door.

    The asymmetry is the whole point. When a superlative claim is TRUE the
    subject usually IS the record holder, so its own article legitimately
    answers. When it is FALSE the subject's article structurally cannot name the
    rival. Pooling therefore helps true claims and pushes false ones toward
    SUPPORTS — one label restrained and not the other, which is bias rather
    than caution, and the same trap rule 6 of the verifier prompt was rewritten
    to escape.

    Running the fallback ONLY when the primary attempt yields nothing keeps both
    properties. A claim whose category query retrieves and answers never reaches
    this path, so no currently-correct verdict can move; a claim whose category
    query was too malformed to retrieve anything gets a second chance instead of
    an automatic NOT ENOUGH INFO.

    from_fallback is returned rather than merely logged because a SUPPORTS
    reached this way rests on the subject asserting its own record — weaker than
    an independent search for the holder — and execute() says so in the
    rationale instead of presenting the two as equivalent.
    """
    answer, chunks, url = _question_once(query, type_word=type_word,
                                         claim=claim, subject=subject)
    if answer is not None:
        return answer, chunks, url, False

    for fallback in fallback_queries:
        if not fallback or fallback == query:
            continue
        print(f"[Program] Question('{query}') found nothing — retrying with "
              f"fallback query '{fallback}'.")
        fb_answer, fb_chunks, fb_url = _question_once(
            fallback, rank_against=query, type_word=type_word,
            claim=claim, subject=subject,
        )
        if fb_answer is not None:
            return fb_answer, fb_chunks, fb_url, True
        # Keep whatever evidence was seen, so a NOT ENOUGH INFO still carries
        # chunks and a citation rather than coming back empty.
        chunks = chunks or fb_chunks
        url = url or fb_url

    return None, chunks, url, False


def _question_once(query: str,
                   rank_against: str | None = None,
                   type_word: str = "",
                   claim: str = "",
                   subject: str = "") -> tuple[str | None, list[str], str]:
    """
    One Question attempt: retrieve, ask, validate. Returns the entity or None.

    `rank_against` is passed through to _retrieve; the prompt and the kind check
    both use it in preference to `query`, so the model is still asked the
    CATEGORY question even when the articles were fetched by subject name.

    `type_word` is supplied by the caller from the claim's own parse. It falls
    back to _type_word(query) only when absent, which keeps this function usable
    standalone and keeps its existing unit tests meaningful.

    Returns None rather than guessing whenever the evidence does not clearly
    answer, because None becomes NOT ENOUGH INFO downstream and a wrong name
    becomes a wrong verdict.

    THE PROMPT USED TO SAY "answer with the name of the place, person or
    thing" — every kind of name it might need, and also every kind it must
    not pick. Asked "what is the earliest European permanent settlement
    United States?", the evidence named two things in one sentence: the
    settlement itself ('Caparra, Puerto Rico') and the man who founded it
    ('Juan Ponce de León'). Nothing in the prompt said which of the two kinds
    on offer — place or person — this question wanted, so the model was free
    to pick either, and it picked the founder.

    _answer_is_right_kind's 'by'-agent check catches that ONE surface pattern
    after the fact. It does not stop the model from reaching for some other
    wrong name near the type word — an unrelated place in the same sentence, a
    sponsoring organization, a date spelled out as words — because the prompt
    itself never narrowed what kind of answer is wanted. That has to happen
    here, before generation, not in a filter afterward.

    The query already carries the answer to "what kind": _type_word extracts
    it ('settlement', 'building', 'river') from the same predicate query this
    function is answering. Telling the model directly — answer with the
    {type_word}'s own name, not a person connected to it — removes the
    ambiguity that let the wrong pick happen, rather than catching it once it
    already has. The downstream check stays; a prompt instruction is a
    preference an 8B-class model can still miss, not a guarantee, and
    defense in depth costs nothing extra here.
    """
    asked = rank_against or query

    chunks, url = _retrieve(query, QUESTION_TOP_K, rank_against=rank_against)
    if not chunks:
        print(f"[Program] Question('{query}') — no evidence retrieved.")
        return None, [], ""

    type_word = type_word or _type_word(asked) or ""
    if type_word:
        kind_rule = (
            f"- The answer is the {type_word} itself — its own name. Do NOT answer "
            f"with the name of a person connected to it (who founded, built, "
            f"discovered, led, or named it) — that is a different entity from "
            f"the {type_word} being asked about."
        )
    else:
        kind_rule = "- Answer with the name of the place, person or thing, and nothing else."

    evidence = "\n".join(f"Evidence_{i}: {c}" for i, c in enumerate(chunks, 1))
    prompt = f"""Read the evidence and answer the question with a NAME ONLY.

{evidence}

Question: What is the {asked}?

Rules:
{kind_rule}
- Copy the name exactly as it appears in the evidence.
- If the evidence does not clearly answer the question, reply exactly: UNKNOWN

Answer:"""

    raw = _call_answer_model(prompt)
    answer = raw.splitlines()[0].strip().strip('."') if raw else ""

    if not answer or answer.upper().startswith("UNKNOWN"):
        print(f"[Program] Question('{query}') -> UNKNOWN")
        return None, chunks, url

    if not _answer_is_grounded(answer, chunks):
        print(f"[Program] Question('{query}') -> '{answer}' REJECTED — not in the "
              f"evidence, so it came from the model's own knowledge.")
        return None, chunks, url

    if not _answer_is_right_kind(answer, asked, chunks, claim=claim, subject=subject):
        return None, chunks, url

    print(f"[Program] Question('{query}') -> '{answer}'")
    return answer, chunks, url


class NLIUnavailable(RuntimeError):
    """The NLI model could not be loaded: a setup defect, not a runtime condition."""


@functools.lru_cache(maxsize=1)
def _nli():
    """
    (tokenizer, model, entailment_index), loaded once per process.

    Imported lazily so that importing this module, and every test that does not
    reach the kind check, costs nothing and needs no torch. A missing package or
    a failed download raises NLIUnavailable, which try_verify reports loudly
    instead of letting it look like an ordinary retrieval failure.
    """
    try:
        import torch  # noqa: F401  (transformers needs it; fail here, not deep inside)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise NLIUnavailable(
            f"{e}. Install the NLI dependencies: pip install -r requirements.txt"
        ) from e

    try:
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    except OSError as e:
        raise NLIUnavailable(f"could not load '{NLI_MODEL}': {e}") from e

    # Read the label order from the model rather than hard-coding it: NLI
    # checkpoints disagree on whether entailment is index 0 or 2.
    labels = {name.lower(): i for i, name in model.config.id2label.items()}
    return tokenizer, model, labels["entailment"]


def _entailment_scores(premises: list[str], hypothesis: str) -> list[float]:
    """P(entailment) of `hypothesis` given each premise, in one batched pass."""
    import torch

    tokenizer, model, entail = _nli()
    inputs = tokenizer(premises, [hypothesis] * len(premises),
                       truncation="only_first", max_length=512,
                       padding=True, return_tensors="pt")
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1)
    return probs[:, entail].tolist()


def _hypothesis(answer: str, asked: str, claim: str = "", subject: str = "") -> str:
    """
    The claim restated about `answer` instead of its subject.

        claim   'The blue whale is the most gigantic creature in history.'
        subject 'blue whale',  answer 'Antarctic blue whale'
        ->      'The Antarctic blue whale is the most gigantic creature in history.'

    Falls back to "{answer} is the {asked}." when the subject cannot be found in
    the claim, or when called without one (standalone use, unit tests).
    """
    answer = re.sub(r'^(the|a|an)\s+', '', answer.strip(), flags=re.I)
    if claim and subject:
        pattern = re.compile(re.escape(subject), re.I)
        if pattern.search(claim):
            # A function replacement, so a backslash in the answer is literal.
            return pattern.sub(lambda _m: answer, claim, count=1)
    return f"{answer} is the {asked}."


def _answer_is_right_kind(answer: str, query: str, chunks: list[str],
                          claim: str = "", subject: str = "") -> bool:
    """
    Does the evidence actually STATE that the answer holds the claim's record?

    Asked by NLI: the claim is restated with the answer in place of its subject
    (see _hypothesis), and an evidence window must ENTAIL that restatement with
    P(entailment) >= ENTAILMENT_THRESHOLD.

        premise     "The blue whale is the largest animal known to have ever
                     existed."
        hypothesis  "The blue whale is the most gigantic creature in history."
                     -> entailment

    THE FAILURE THIS CATCHES. Asked for the earliest European permanent
    settlement in the United States, the answer model returned 'Juan Ponce de
    León' — an explorer who does appear in the evidence, so grounding passed.
    Match then returned REFUTES for a person compared against a settlement, and
    it would have done the same for ANY grounded string: "Virginia Company",
    "1565", "the Atlantic Ocean". "Juan Ponce de León is the earliest European
    permanent settlement in the United States" is entailed by nothing, so the
    answer is rejected here and the claim falls to NOT ENOUGH INFO instead.

    WHAT IT REPLACED, AND WHY. The previous check was positional: accept the
    answer if it appeared in the same sentence as a TYPE WORD ('settlement')
    read off the query. Every part of that was a string rule standing in for a
    judgement about meaning, and each broke on live input:

      - the type word itself: _type_word returned 'states' (the scope) and then
        'land' (a modifier) before being fixed, and still returns the scope
        noun when the scope is a common noun ('planet', 'history');
      - synonyms: evidence saying 'largest ANIMAL' failed a query about the
        'most gigantic CREATURE', rejecting a correct answer;
      - substrings: 'land' matched inside 'island';
      - co-occurrence: Ponce de León passed whenever a sentence named him
        alongside the word 'settlement' ("founded ... by Ponce de León");
      - sentence splitting: 'St. Augustine' split at 'St.' and could never
        co-occur with anything.

    NLI needs no type word, reads synonyms and paraphrase, and tests the whole
    predicate rather than one noun — so it also checks that the answer holds the
    RECORD, not merely that it is the right kind of thing.

    WINDOWS OF TWO SENTENCES. Each premise is a sentence plus the one before it,
    so "It is the largest animal..." still has its referent, and a name split
    across a bad sentence boundary ('St.' | 'Augustine, Florida') is rejoined.

    FAILS CLOSED. No entailing window means rejection -> NOT ENOUGH INFO: this
    module may only move a verdict when the evidence states the answer, never
    on a guess. A model that cannot be loaded raises NLIUnavailable rather than
    silently accepting or rejecting everything.
    """
    # A BARE NUMBER CANNOT NAME A NAMED SUBJECT'S RIVAL — and NLI accepts one.
    # Measured on this model: "1565 is the earliest European permanent
    # settlement in the United States" is entailed at p=0.98 by "...a permanent
    # settlement ... at Saint Augustine, Florida (1565)", which reads the
    # parenthesised year as another name for the place. Rejected on form, not
    # meaning, and only when the subject itself contains a letter, so a claim
    # about a year ("2016 was the hottest year") is unaffected.
    if re.search(r'[a-z]', subject, re.I) and not re.search(r'[a-z]', answer, re.I):
        print(f"[Program] Question('{query}') -> '{answer}' REJECTED — a bare "
              f"number cannot be the rival of '{subject}'.")
        return False

    hypothesis = _hypothesis(answer, query, claim, subject)

    premises = []
    for chunk in chunks:
        sents = _sentences(chunk)
        premises += [" ".join(sents[max(0, i - 1):i + 1]) for i in range(len(sents))]
    if not premises:
        return False

    scores = _entailment_scores(premises, hypothesis)
    best = max(range(len(scores)), key=scores.__getitem__)

    if scores[best] >= ENTAILMENT_THRESHOLD:
        print(f"[Program] Kind check: '{hypothesis}' entailed "
              f"(p={scores[best]:.2f}) by: \"{premises[best]}\"")
        return True

    print(f"[Program] Question('{query}') -> '{answer}' REJECTED — no evidence "
          f"entails '{hypothesis}' (best p={scores[best]:.2f}).")
    return False


def _type_word(query: str) -> str | None:
    """
    The noun in the predicate query that names what is being ranked.

    'earliest European permanent settlement United States' -> 'settlement'

    Taken from the parse rather than by position: the last token is 'States'
    here, which names the SCOPE of the superlative, not its TYPE.

    THAT COMMENT WAS ASPIRATIONAL, NOT MEASURED. build_predicate_query's output
    is a keyword fragment, not a sentence, and en_core_web_sm parses this exact
    string as ONE noun chunk — 'European permanent settlement United States' —
    headed by 'States', the same rightward-compounding rule that makes 'leg'
    the head of 'kitchen table leg'. The first loop below used to accept a
    PROPN root as well as a NOUN one, so it took that chunk's root and returned
    'states' without ever reaching the token scan beneath it, which does find
    'settlement' (tagged NOUN, sitting right there in the parse).

    THIS WAS NOT A LATENT RISK. Run live against the real Jamestown claim with
    real Wikipedia retrieval (not verifier_probe's fixed evidence): the answer
    model returned 'Juan Ponce de León' for the settlement question, and this
    function — checked at that moment — returned 'states', which 'Juan Ponce
    de León' co-occurs with in any sentence mentioning "a United States
    territory". _answer_is_right_kind passed a person as a settlement and
    _handle_match returned a confident REFUTES, sourced to a sentence that
    never called him a settlement. This is the exact failure the function's
    own docstring says it exists to catch — reproduced against the case that
    motivated writing it.

    THE SCOPE ENTITY IS ALWAYS PROPN, BY CONSTRUCTION. build_predicate_query
    keeps NOUN/PROPN/ADJ/NUM content words after the subject is stripped, and
    the trailing scope ('United States', 'East Asia') is always the proper
    noun at the end; the type word ('settlement', 'building', 'river') is
    always the common noun before it. So the fix is not a better parse of the
    fragment, it is to stop trusting the chunk's root at all: restrict the
    first loop to NOUN only. A PROPN-rooted chunk falls straight through to
    the second loop, which was already correct and is now reachable.

    THE FALLBACK LOOP HAD THE SAME ONE-NOUN ASSUMPTION AS THE FIRST. Live
    query 'largest terrestrial land animal Earth' (from "...the largest
    living terrestrial land animal on Earth" with 'on' stripped): spaCy again
    roots the whole span at 'Earth' (PROPN), so the first loop correctly
    declines. The fallback then scanned tokens in TEXTUAL ORDER and returned
    the first NOUN, which is 'land' — a compound MODIFIER of 'animal', not
    the type being asked about. Every prior example ('settlement United
    States') only ever had one NOUN before the trailing proper-noun run, so
    this branch was never exercised against a multi-noun compound until now.

    Fed downstream, type_word='land' produced the instruction "the answer is
    the land itself... not a person connected to it" — nonsense for a
    question about an animal — and phi3:mini answered UNKNOWN despite the
    evidence stating the answer plainly. Confirmed with
    `python -c "import Pipeline.program as p; print(p._type_word('largest
    terrestrial land animal Earth'))"` -> 'land', before and after fix.

    English compounds are right-headed ('land animal' is a kind of animal,
    not a kind of land), so the LAST noun before the trailing proper-noun run
    is the type word, not the first. Taking the last NOUN still returns
    'settlement' for the single-noun cases above.
    """
    doc = ner.NLP(query)
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "NOUN":
            return chunk.root.text.lower()
    nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
    return nouns[-1] if nouns else None


def _sentences(text: str) -> list[str]:
    """Cheap sentence split. No spaCy: this runs over long evidence chunks."""
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _head_and_modifiers(name: str) -> tuple[str | None, set[str]]:
    """
    ('African bush elephant') -> ('elephant', {'african', 'bush'})

    The HEAD is what the name is a kind of; the MODIFIERS say which one. English
    compounds are right-headed — 'land animal' is a kind of animal, not a kind
    of land — so the head is the last content word. Same rule _type_word relies
    on, for the same reason.
    """
    doc = ner.NLP(name)
    content = [t for t in doc
               if t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM") and not t.is_stop]
    if not content:
        return None, set()
    return content[-1].text.lower(), {t.text.lower() for t in content[:-1]}


def _handle_match(answer: str | None, subject: str) -> tuple[str, str]:
    """
    Match(answer, subject) -> (label, rationale).

    THREE TIERS, because "these two strings differ" and "these two strings name
    different things" are not the same statement, and the single substring test
    that used to be here could not tell them apart.

      1. CONTAINMENT. 'Jamestown' against 'the Jamestown settlement', 'Everest'
         against 'Mount Everest' — the same thing written at two lengths. Runs
         first because these have DIFFERENT head nouns ('jamestown' vs
         'settlement') and tier 2 would wrongly split them.

      2. SHARED HEAD, COMPARED MODIFIERS. The tier that exists because of an
         observed near-miss: asked for the largest land animal, the evidence
         said 'African elephant' while the claim said 'African bush elephant'.
         Neither contains the other, so tier 1 failed and the old code returned
         a confident REFUTES on a true claim — saved only by
         _answer_is_right_kind rejecting the answer first and downgrading to
         NEI. The distinction that settles it is set containment, not string
         similarity:

             African elephant       head=elephant  mods={african}
             African bush elephant  head=elephant  mods={african, bush}
                 {african} ⊂ {african, bush}  -> a narrower name for one animal

             Asian elephant         head=elephant  mods={asian}
             African elephant       head=elephant  mods={african}
                 neither contains the other   -> two different animals

         WHY NOT A SIMILARITY SCORE. Measured on en_core_web_lg, GloVe rates
         'Asian elephant'/'African elephant' (different animals) at 0.830 and
         'Jamestown'/'the Jamestown settlement' (same place) at 0.667 — the
         wrong pair scores higher, so no threshold classifies both correctly.
         Plain edit distance inverts them too (0.867 vs 0.545). Subset-vs-
         disjoint is categorical and needs no threshold to tune.

      3. DIFFERENT HEADS. A genuine rival: 'Saint Augustine' against
         'Jamestown'. Keeps the previous REFUTES, which _handle_corroborate
         then requires a second independent source to sustain.

    KNOWN GAP: an alias sharing no words with the subject ('Loxodonta
    africana') reaches tier 3 and is reported as a rival. Corroboration will
    usually sustain it, since the subject's own article does mention its
    scientific name. Resolving that needs canonical titles from Wikipedia
    redirects, not a cleverer string rule.
    """
    if answer is None:
        return ("NOT ENOUGH INFO",
                "The evidence does not name which entity actually holds this "
                "distinction, so the claim could not be checked against it.")

    a, s = answer.lower(), subject.lower()
    if a in s or s in a:
        return ("SUPPORTS",
                f"The evidence names '{answer}' as holding this distinction, "
                f"which matches the claim's subject '{subject}'.")

    head_a, mods_a = _head_and_modifiers(answer)
    head_b, mods_b = _head_and_modifiers(subject)

    if head_a is not None and head_a == head_b:
        if mods_a <= mods_b or mods_b <= mods_a:
            return ("SUPPORTS",
                    f"The evidence names '{answer}' as holding this "
                    f"distinction. That is the same {head_a} as the claim's "
                    f"subject '{subject}', named less specifically.")

        return ("REFUTES",
                f"The evidence names '{answer}' as holding this distinction. "
                f"That is a different {head_a} from the claim's subject "
                f"'{subject}'.")

    return ("REFUTES",
            f"The evidence names '{answer}' as holding this distinction, not "
            f"'{subject}' as the claim asserts.")


def _handle_corroborate(answer: str, subject: str) -> tuple[bool, list[str], str]:
    """
    Does the SUBJECT'S OWN article back up the rival Question() found?

    WHY THIS EXISTS. Question()'s retrieval is aimed at the CATEGORY
    ('earliest European permanent settlement United States'), not the
    subject, which is the whole point of the mechanism — it is how Saint
    Augustine gets found at all. But that same aim is the risk: Wikipedia's
    search on an artificial category phrase is not always good. Measured on
    two claims sharing this exact machinery: 'earliest European permanent
    settlement United States' scored 0.72 and reached the right article;
    'earliest permanent European settlement East Asia' scored 0.27 and
    reached 'Indo-European migrations' — a keyword collision, not an answer.
    A single low-confidence retrieval is standing between REFUTES and a
    citation that does not actually support it.

    So a REFUTES verdict gets a second, independent source before it is
    trusted: retrieve the SUBJECT's own article (the same way the ordinary
    verifier path already does for every claim) and check whether the rival
    name Question() found appears there too. Two retrieval calls landing on
    the same rival, aimed at two different queries, is a much stronger signal
    than either one alone.

    WHAT A MISS MEANS, AND WHAT IT DOES NOT. If the subject's own article
    never mentions the rival, that is NOT evidence the rival is wrong — most
    articles do not enumerate every earlier claimant, which is exactly why
    Question() had to search the category in the first place. A miss here
    means only "not corroborated," and the caller downgrades to NOT ENOUGH
    INFO rather than treating the absence as a refutation of the refutation.
    Silence is not confirmation in either direction — see verifier.py's rule 6
    for the same principle applied to the single-shot prompt.

    COST. One more retrieval call (~5-15s), spent only on the REFUTES branch,
    where the failure this guards against — a confident wrong verdict with a
    citation attached — is the one this whole project exists to prevent.
    """
    chunks, url = _retrieve(subject, QUESTION_TOP_K)
    if not chunks:
        return False, [], ""

    answer_low = answer.lower()
    found = any(answer_low in chunk.lower() for chunk in chunks)
    return found, chunks, url


_NUM = r'(\d[\d,]*(?:\.\d+)?)'
_UNIT = r'(' + '|'.join(sorted(_UNIT_TO_BASE, key=len, reverse=True)) + r')'
_QUANTITY_RE = re.compile(_NUM + r'\s*' + _UNIT + r'\b', re.I)


def _handle_quantity(entity: str, dimension: str, family: str):
    """
    Quantity(entity, dimension) -> (value_in_base_units, display, chunks, url)

    THE NUMBER MUST SHARE A SENTENCE WITH THE DIMENSION WORD.

    The first version scored candidates by character distance to the nearest
    dimension keyword anywhere in the chunk. On the Nile that produced:

        Quantity('Nile', length) -> 20 ft        (the Nile is 6,650 km)

    — a measurement of something else entirely, sitting a few hundred characters
    from the word 'length'. Character distance across a 500-character chunk is
    not a relation between a number and a word; it is a coincidence with a
    threshold. A shared sentence is an actual relation:

        "The Nile is about 6,650 km (4,130 mi) long."

    Among candidates that clear that bar, one that appears in a sentence NAMING
    THE ENTITY wins, because an article about the Nile discusses many things'
    lengths and only some sentences are about the Nile's.

    The final tiebreak prefers the LARGEST value, which is a heuristic and worth
    knowing about: in a sentence giving a figure twice in different units
    ("6,650 km (4,130 mi)") both convert to nearly the same base value, so the
    choice is harmless; where a sentence mixes a feature's length with a smaller
    secondary measurement, the larger is usually the headline figure. It is the
    weakest rule here and the first thing to suspect if a comparison looks odd.
    """
    chunks, url = _retrieve(f"{entity} {dimension}", QUESTION_TOP_K)
    if not chunks:
        return None, "", [], ""

    keywords = {dimension} | {k for k, v in _DIMENSIONS.items() if v[0] == dimension}
    entity_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", entity) if len(t) >= 4]

    best = None
    for chunk in chunks:
        for sentence in _sentences(chunk):
            low = sentence.lower()
            if not any(k in low for k in keywords):
                continue
            names_entity = any(t in low for t in entity_tokens) if entity_tokens else False

            for m in _QUANTITY_RE.finditer(sentence):
                unit = m.group(2).lower()
                if _UNIT_FAMILY.get(unit) != family:
                    continue
                value = float(m.group(1).replace(",", "")) * _UNIT_TO_BASE[unit]
                # Lower sorts first: entity-naming sentences beat others, then
                # larger values beat smaller ones.
                rank = (0 if names_entity else 1, -value)
                if best is None or rank < best[0]:
                    best = (rank, value, m.group(0), sentence)

    if best is None:
        print(f"[Program] Quantity('{entity}', {dimension}) — no value found in any "
              f"sentence mentioning {sorted(keywords)}.")
        return None, "", chunks, url

    _, value, display, sentence = best
    print(f"[Program] Quantity('{entity}', {dimension}) -> {display} ({value:g} base)")
    print(f"[Program]   from: {sentence.strip()[:110]}")
    return value, display, chunks, url


def _handle_compare(a, b, subject: str, other: str, dimension: str,
                    disp_a: str, disp_b: str) -> tuple[str, str]:
    """Compare(q_1, q_2) -> (label, rationale). Arithmetic, in Python."""
    if a is None or b is None:
        missing = subject if a is None else other
        return ("NOT ENOUGH INFO",
                f"The {dimension} of '{missing}' could not be read from the "
                f"retrieved evidence, so the comparison could not be made.")

    # IMPLAUSIBLE RATIOS ARE EXTRACTION ERRORS, NOT FINDINGS.
    #
    # Two things compared on the same dimension are, in any claim a person
    # would bother writing, the same order of magnitude — nobody asks whether a
    # river is longer than a shoelace. An observed run read the Nile's length as
    # '20 ft' (6 m) against a genuine 6,650 km: a ratio of a million, which is
    # not a fact about rivers but a fact about the extractor.
    #
    # 1000x is deliberately loose. It passes every real comparison (Everest is
    # 1.03x K2; the Burj is 1.9x the Empire State) and catches the failures,
    # which are wrong by several orders of magnitude rather than by a little.
    ratio = max(a, b) / min(a, b) if min(a, b) > 0 else float("inf")
    if ratio > 1000:
        return ("NOT ENOUGH INFO",
                f"The values read from the evidence — {subject} {disp_a}, "
                f"{other} {disp_b} — differ by a factor of {ratio:.0f}, which "
                f"means at least one was misread. No comparison was made.")

    if a > b:
        return ("SUPPORTS",
                f"The evidence gives {subject} as {disp_a} and {other} as "
                f"{disp_b}, so the claim holds.")
    if a < b:
        return ("REFUTES",
                f"The evidence gives {subject} as {disp_a} and {other} as "
                f"{disp_b}, so the claim is false.")
    return ("NOT ENOUGH INFO",
            f"The evidence gives both {subject} and {other} as {disp_a}, which "
            f"does not settle the comparison.")


# ── Execution ─────────────────────────────────────────────────────────────────
def execute(program: Program, claim: str) -> ProgramOutcome | None:
    """
    Run a program and return its verdict with provenance.

    Returns None if execution could not proceed at all, which sends the caller
    back to single-shot verification rather than leaving the claim unchecked.
    """
    if program.kind == "superlative":
        category = program.steps[0].args[0]
        subject = program.steps[1].args[1]

        answer, chunks, url, from_fallback = _handle_question(
            category,
            fallback_queries=(subject,),
            type_word=program.type_word,
            claim=claim,
            subject=subject,
        )
        label, rationale = _handle_match(answer, subject)

        program.trace = [
            f"answer_1 = Question('{category}'"
            + (f", fallback='{subject}'" if from_fallback else "")
            + f") -> {answer or 'UNKNOWN'}",
            f"label = Match(answer_1, '{subject}') -> {label}",
        ]

        # A SUPPORTS that came from the subject's own article is not independent
        # evidence of a ranking — it is the subject asserting its own record.
        # Keep the label; say where it came from.
        if from_fallback and label == "SUPPORTS":
            rationale += (
                f" Note: the category search for '{category}' returned no usable "
                f"evidence, so this rests on '{subject}''s own article rather "
                f"than an independent search for the record holder."
            )

        # REFUTES rests on ONE retrieval call aimed at an artificial category
        # phrase — see _handle_corroborate for why that is not always trusted
        # alone. Require a second, independent source before keeping it.
        if label == "REFUTES":
            corroborated, subj_chunks, subj_url = _handle_corroborate(answer, subject)
            program.trace.append(
                f"corroborate = SubjectArticle('{subject}') mentions '{answer}' "
                f"-> {corroborated}"
            )
            if corroborated:
                chunks = chunks + subj_chunks
            else:
                label = "NOT ENOUGH INFO"
                rationale = (
                    f"A search for '{category}' named '{answer}' as holding this "
                    f"distinction, but '{subject}''s own article does not "
                    f"corroborate that, so the rival finding could not be "
                    f"confirmed from a second source."
                )
                if subj_chunks:
                    chunks = chunks + subj_chunks
                    url = url or subj_url

        program.chunks, program.source_url = chunks, url

    elif program.kind == "comparative":
        subject, dimension, family = program.steps[0].args
        other = program.steps[1].args[0]

        a, disp_a, chunks_a, url_a = _handle_quantity(subject, dimension, family)
        b, disp_b, chunks_b, _ = _handle_quantity(other, dimension, family)

        label, rationale = _handle_compare(
            a, b, subject, other, dimension, disp_a, disp_b
        )

        program.trace = [
            f"q_1 = Quantity('{subject}', {dimension}) -> {disp_a or 'UNKNOWN'}",
            f"q_2 = Quantity('{other}', {dimension}) -> {disp_b or 'UNKNOWN'}",
            f"label = Compare(q_1, q_2) -> {label}",
        ]
        program.chunks = chunks_a + chunks_b
        program.source_url = url_a

    else:
        return None

    for line in program.trace:
        print(f"[Program]   {line}")

    return ProgramOutcome(
        verdict=VerificationResult(
            label=label,
            evidence=program.chunks[0] if program.chunks else "",
            # The trace travels with the verdict. A program-guided answer that
            # cannot show its steps has thrown away the reason for using one.
            rationale=rationale + "  [program: " + " ; ".join(program.trace) + "]",
        ),
        kind=program.kind,
        query=program.query,
        source_url=program.source_url,
    )


def try_verify(claim: str) -> ProgramOutcome | None:
    """
    Entry point. Returns a ProgramOutcome, or None when no program applies and
    the caller should fall back to single-shot verification.
    """
    if not PROGRAM_MODE:
        return None

    program = synthesize(claim)
    if program is None:
        return None

    print(f"\n[Program] {program.kind.upper()} claim — running program:")
    for step in program.steps:
        print(f"[Program]   {step.var} = {step.op}{step.args}")

    try:
        return execute(program, claim)

    except NLIUnavailable as e:
        # A setup defect, reported as loudly as a code defect below and for the
        # same reason: the fallback keeps the claim, but the program path is off.
        print(f"[Program] *** NLI MODEL UNAVAILABLE: {e}")
        print(f"[Program] *** The claim falls back to the verifier, but the "
              f"program path is NOT running until this is fixed.")
        return None

    except (AttributeError, TypeError, NameError, KeyError, IndexError) as e:
        # A BUG, NOT A RUNTIME CONDITION — and the two must not look alike.
        #
        # The first release of this module called a helper that exists in some
        # versions of ollama_client and not others. Every program run raised
        # AttributeError, the blanket handler caught it, and the log line was
        # indistinguishable from "no program applied". The feature was dead for
        # a full 700-second run and the only trace was one quiet line.
        #
        # Falling back is still right — losing the claim would be worse — but
        # these exception types mean the code is broken, so say so loudly.
        print(f"[Program] *** BUG: {type(e).__name__}: {e}")
        print(f"[Program] *** This is a defect in program.py, not a retrieval "
              f"failure. The claim falls back to the verifier, but the program "
              f"path is NOT running.")
        return None

    except Exception as e:
        # Expected runtime trouble: a network timeout, a Wikipedia error page,
        # a malformed response. Fall through quietly; the verifier can still
        # answer from its own retrieval.
        print(f"[Program] Execution failed ({type(e).__name__}: {e}) — "
              f"falling back to the verifier.")
        return None
