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

ONE FRAME, NOT TWO CLAIM TYPES. Every program is built from a ComparisonFrame
(subject, dimension, direction, rank, category, scope, reference), so a
superlative with an ordinal ("the second-highest") and a comparative that ranks
("longer than any other river") are both expressible. The frame comes from the
parse by default; FRAME_PARSER = "llm" has the model fill it through a JSON
schema — slots, not a program, which is a middle path between the two above.
See ComparisonFrame and Documents/program_references.md.

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
import json
import re
from dataclasses import dataclass, field

import pint
import requests
from spacy.lang.en.stop_words import STOP_WORDS

import Pipeline.ner as ner
import Pipeline.retriever as retriever
import Pipeline.wikidata as wikidata
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
#
# WAS 0.5, WHICH ACCEPTED A DIFFERENT RECORD. Live: "Huascarán is the
# second-highest peak on Earth" passed at p=0.74 from a window about the
# FARTHEST POINT FROM EARTH'S CENTER ("...with Huascarán in Peru a very close
# second"). Measured on this model:
#
#     true statements of the record   0.977 - 0.987   (Everest, K2's own lead,
#                                                      Saint Augustine, the blue
#                                                      whale by synonym, the Nile)
#     false or off-record windows     0.001 - 0.739   (Huascarán 0.739, Mauna Kea
#                                                      0.327, Ponce de León 0.080)
#
# 0.9 sits in the gap with margin on both sides. A dimension-keyword test on the
# window was considered and rejected: the Huascarán window contains 'higher',
# so it would have passed that test too.
ENTAILMENT_THRESHOLD = 0.9


# How the comparison frame is built. "rules" reads it off the spaCy parse —
# deterministic, offline, and the baseline the tests pin. "llm" asks
# ANSWER_MODEL to fill the frame through Ollama's JSON-schema output, validates
# every slot against the claim, and falls back to the rules when the reply is
# unusable. Its value is phrasings no rule anticipates ("one of the two
# highest", "outranks every other"); its cost is one extra call per claim, which
# should be measured before it becomes the default.
FRAME_PARSER = "rules"

# ── Operator detection ────────────────────────────────────────────────────────
# Comparative markers. JJR/RBR are the comparative tags ('taller', 'longer',
# 'more'). A comparative alone is not enough — "the river is longer now than in
# 1970" compares one thing to itself across time — so a comparison also requires
# a 'than' with a nominal object to compare AGAINST.
_COMPARATIVE_TAGS = {"JJR", "RBR"}


@dataclass(frozen=True)
class Dimension:
    """
    A measurable quantity: what it is called, which unit family expresses it,
    and the words a sentence must contain for its numbers to be about it.

    `family` is a pint dimensionality ('length', 'area', 'mass') or 'count' for
    unit-less tallies such as population, which pint has nothing to say about.
    """
    name: str
    family: str
    keywords: tuple[str, ...]


_DIMENSION_SPECS = {
    "height":     Dimension("height", "length", ("height", "tall", "high", "elevation", "altitude")),
    "elevation":  Dimension("elevation", "length", ("elevation", "height", "altitude", "high")),
    "altitude":   Dimension("altitude", "length", ("altitude", "elevation", "height", "high")),
    "length":     Dimension("length", "length", ("length", "long")),
    "depth":      Dimension("depth", "length", ("depth", "deep")),
    "width":      Dimension("width", "length", ("width", "wide")),
    "area":       Dimension("area", "area", ("area",)),
    "weight":     Dimension("weight", "mass", ("weight", "weigh", "heavy", "mass")),
    "population": Dimension("population", "count", ("population", "populous", "inhabitants")),
}

# Dimension lexicon: which quantity an ADJECTIVE is about, and which way it
# points. 'max' — larger value wins ('taller', 'deepest'); 'min' — smaller
# value wins ('lower', 'shallowest').
#
# Deliberately small. Each entry is a claim about what a word measures, and a
# wrong entry produces a confident comparison of unrelated quantities — far
# worse than declining. Ambiguous adjectives are EXCLUDED for exactly that
# reason: 'larger', 'bigger', 'greater' (by area, population or volume?),
# 'shorter' (height or length?), 'lighter' (mass or colour?).
#
# 'old' WAS HERE, mapped to a 'year' family that no extractor ever read — so
# every "older than" claim silently went to NOT ENOUGH INFO. It stays out:
# age needs a reference event (founded, built, born, formed) that differs by
# category, and the year nearest a keyword is as often a renovation as a
# founding.
_DIMENSIONS = {
    "tall":     ("height", "max"),
    "high":     ("height", "max"),
    "low":      ("height", "min"),
    "long":     ("length", "max"),
    "deep":     ("depth", "max"),
    "shallow":  ("depth", "min"),
    "wide":     ("width", "max"),
    "narrow":   ("width", "min"),
    "heavy":    ("weight", "max"),
    "populous": ("population", "max"),
}

# _NOUN_DIMENSIONS: dimension keyed by the NOUN a generic comparative modifies.
#
# _DIMENSIONS above is keyed by the ADJECTIVE, which is correct only when the
# adjective itself names the dimension ("is TALLER", "is LONGER"). It is wrong
# for an attributive comparative like "higher VELOCITY" or "higher fatality
# RATE": there "higher" is a bare magnitude word and the dimension is the noun.
# Reading the adjective in that case mapped every "higher X" to 'high' -> height
# and produced, live, a fatality-rate claim "refuted" by comparing mountain
# heights. See _dimension_of for which parse shape selects which table.
#
# Only nouns whose unit family the Quantity handler can actually read are listed.
# A noun that is not here — velocity, rate, temperature — resolves to None and
# the claim declines to the verifier rather than inventing a comparison in the
# wrong units.
_NOUN_DIMENSIONS = {
    "height": "height", "elevation": "elevation", "altitude": "altitude",
    "length": "length", "depth": "depth", "width": "width", "area": "area",
    "weight": "weight", "mass": "weight", "population": "population",
}

# Markers whose direction is 'min' whatever they modify: "LESS populous",
# "the LEAST deep".
_DIMINISHING = {"less", "least", "little", "few", "fewer", "fewest"}

# Comparative -> superlative, for "longer than any other river", which is a
# superlative claim written as a comparison and needs a superlative question.
_IRREGULAR_SUPERLATIVE = {
    "better": "best", "worse": "worst", "more": "most", "less": "least",
    "farther": "farthest", "further": "furthest",
}

# Quantifiers that make a 'than' object a whole category, not an entity:
# "than ANY other river", "than ALL other rivers", "than EVERY other river".
_UNIVERSAL_DETERMINERS = {"any", "all", "every", "each"}

_ORDINAL_TEXT = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
                 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}


# ── Program representation ────────────────────────────────────────────────────
@dataclass(frozen=True)
class ComparisonFrame:
    """
    One typed reading of a claim that compares or ranks — the form every
    program is built from.

    WHY ONE FRAME AND NOT TWO CLAIM TYPES. The module used to sort claims into
    'superlative' or 'comparative' and give each its own path. Real claims do
    not respect that split:

        "K2 is the second-highest peak on Earth"             a superlative with a RANK
        "The Nile is longer than any other river in Africa"  a comparative that is
                                                             really a superlative

    The first was asked "what is the highest peak?"; the second was compared
    against an entity called 'river in Africa'. Both are one operation — order
    a set by a quantity, then check a position or a pair — which is how
    semantic parsing has represented them: lambda DCS's argmax with a rank
    argument (Liang 2013), TabFact's argmax/argmin program functions (Chen et
    al. 2020), KQA Pro's KoPL comparison operators (Cao et al. 2022). See
    Documents/program_references.md.

    rank       1.. for a ranking claim ("the second-highest" -> 2); None for a
               pairwise comparison against `reference`.
    marker     the superlative as a question would use it, ordinal included:
               'second highest', 'most populous', 'longest'.
    dimension  key into _DIMENSION_SPECS, or None when the marker names no
               quantity this module can read ('earliest', 'largest').
    direction  'max' when the larger value wins, 'min' when the smaller does.
    category   what is ranked ('peak', 'settlement'); scope is where ('Africa').
    query      the category question a ranking claim retrieves with.
    """
    subject: str
    marker: str
    dimension: str | None
    direction: str
    rank: int | None
    category: str | None = None
    scope: str | None = None
    reference: str | None = None
    query: str = ""
    type_word: str = ""
    source: str = "rules"
    # True when nothing but the marker narrows the category — see
    # _scope_and_bare. Defaults False: unknown means "may be narrowed".
    bare: bool = False

    @property
    def world_scope(self) -> bool:
        """Is the ranking over the whole class, not a place within it?"""
        return self.scope is None or self.scope.lower() in _WORLD_SCOPES

    @property
    def kind(self) -> str:
        # The two names the rest of the pipeline (results.program_kind, the
        # frontend) already knows. 'superlative' now covers every ranking claim.
        return "superlative" if self.rank is not None else "comparative"


@dataclass(frozen=True)
class Step:
    """One instruction. `op` names the handler, `args` are its inputs."""
    var: str
    op: str
    args: tuple


@dataclass
class Program:
    """
    A synthesised reasoning program: the frame it was built from, the steps it
    shows, and what executing it produced.

    The FRAME drives execution; `steps` is the plan as printed before running.
    kind, query and type_word are read from the frame rather than stored beside
    it, so the two can never disagree.

    `trace` is not decoration. ProgramFC's stated advantage over end-to-end
    verification is that the reasoning is inspectable and debuggable, and a
    program whose intermediate answers are invisible gives that up. The trace
    becomes part of the rationale shown to the user, so a verdict can be argued
    with rather than merely accepted.
    """
    frame: ComparisonFrame
    steps: list[Step]
    trace: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    source_url: str = ""

    @property
    def kind(self) -> str:
        return self.frame.kind

    @property
    def query(self) -> str:
        return self.frame.query

    @property
    def type_word(self) -> str:
        # The category the superlative ranks within, read from the CLAIM's
        # parse — see ner.superlative_type_word for why the query cannot say.
        return self.frame.type_word


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
    Build a program for `claim`, or None when the claim neither ranks nor compares.
    """
    frame = build_frame(claim)
    if frame is None:
        return None
    return _program_for(frame)


def build_frame(claim: str) -> ComparisonFrame | None:
    """
    The claim's comparison frame, from the model when FRAME_PARSER says so and
    its reply validates, otherwise from the parse.

    The rules always remain the fallback. A model reply that fails validation,
    times out, or says "no comparison" where the parse found one never
    withdraws a program the rules would have built.
    """
    rules = _frame_from_rules(claim)
    if FRAME_PARSER == "llm":
        frame = _frame_from_llm(claim, rules)
        if frame is not None:
            return frame
    return rules


def _frame_from_rules(claim: str) -> ComparisonFrame | None:
    """
    Superlative is tested BEFORE comparative. "The Amazon is the longest river"
    contains 'longest' (JJS) and would also satisfy a loose comparative test;
    the superlative reading is the correct one, and running the comparative
    handler on it would look for a second entity that does not exist.
    """
    doc = ner.NLP(claim)

    superlative = _find_superlative(doc)
    if superlative is not None:
        return _ranking_frame(claim, superlative)

    comparative = _find_comparative(doc)
    if comparative is None:
        return None
    comp, obj, other = comparative
    if obj is not None and _is_universal(obj):
        return _universal_frame(claim, comp, obj)
    return _pairwise_frame(claim, comp, other)


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
    (comparative_token, than_object_token | None, compared_text) — or None.

    Requires BOTH a comparative tag and a 'than' phrase with a nominal object.
    "longer than the Nile" qualifies; "the river is longer now" does not, and
    treating it as a comparison would send the extractor looking for a second
    entity the claim never names.

    The object TOKEN is returned beside its text so the caller can see its
    determiners: "than ANY other river" names a category, not an entity.
    """
    comp = next((t for t in doc if t.tag_ in _COMPARATIVE_TAGS), None)
    if comp is None:
        return None

    than = next((t for t in doc if t.text.lower() == "than"), None)
    if than is None:
        return None

    # The compared entity is the nominal governed by 'than'. Take its whole
    # subtree so multi-word names stay intact ('the Nile', 'Mount Everest').
    #
    # PREFER THE DIRECT DEPENDENT of 'than' (its pobj) over a doc-order scan.
    # The scan version matched the FIRST nominal after 'than', which for "than
    # Mount Everest" is the compound modifier 'Mount' (a leaf whose subtree is
    # just 'Mount'), truncating the entity. The pobj is 'Everest', and its
    # subtree is the full 'Mount Everest'. Only if the parse attaches nothing
    # to 'than' do we fall back to the positional scan.
    # No POS filter on the pobj: spaCy mis-tags entity-like tokens ('K2' comes
    # back ADJ, model numbers as NUM), but the DEPENDENCY 'pobj of than' is
    # reliable where the tag is not. _noun_span keeps PROPN/NOUN/ADJ/NUM, so it
    # recovers 'K2' and multi-word names from the chosen token's subtree.
    pobj = next((t for t in doc if t.head.i == than.i and not t.is_punct), None)
    if pobj is not None:
        span = _noun_span(pobj)
        if span:
            return comp, pobj, span

    for token in doc:
        if token.i > than.i and token.pos_ in ("PROPN", "NOUN"):
            span = _noun_span(token)
            if span:
                return comp, token, span
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


def _is_universal(obj) -> bool:
    """
    Does the 'than' object quantify over a whole category?

        "longer than ANY other river"   -> True   (a superlative in disguise)
        "longer than the Nile"          -> False  (a second entity)

    Read off the determiner, and only on a COMMON noun: "than all" with nothing
    after it ranks against nothing nameable.
    """
    return obj.pos_ == "NOUN" and any(
        c.dep_ in ("det", "predet") and c.lower_ in _UNIVERSAL_DETERMINERS
        for c in obj.children
    )


def _dimension_of(marker) -> tuple[str, str] | None:
    """
    (dimension_key, direction) for a comparative or superlative token, or None.

    THE DIMENSION COMES FROM THE PARSE SHAPE, NOT ALWAYS THE ADJECTIVE. spaCy's
    dependency label on the marker says where the dimension lives:

      advmod "MORE populous / LEAST deep"   -> the adjective it modifies names it
      amod   "higher VELOCITY / RATE"       -> the NOUN it modifies names it
      amod   "is the highest PEAK"          -> the adjective — see below
      acomp  "X is TALLER / LONGER"         -> the adjective IS the dimension

    Reading `marker.lemma_` unconditionally was correct only for the last case;
    for "higher velocity" it mapped 'high' -> height and compared the wrong
    quantity.

    THE COPULAR EXCEPTION. In "Everest is the highest PEAK" the noun is what the
    subject IS — a category, not an attribute — so it cannot name a dimension
    and the adjective does. In "K2 has the highest fatality RATE" the noun is
    the object of 'has', an attribute of the subject, and the claim declines.
    The dependency that separates them is `attr`: the complement of a copula.

    DIRECTION. 'less' / 'least' reverse whatever they modify: "LESS populous"
    is a population claim where the smaller value wins. Before this, "India is
    less populous than China" was compared as if it said MORE populous.
    """
    diminishing = marker.lemma_.lower() in _DIMINISHING

    def flip(direction: str) -> str:
        return {"max": "min", "min": "max"}[direction] if diminishing else direction

    if marker.dep_ == "advmod" and marker.head.pos_ in ("ADJ", "ADV"):
        entry = _DIMENSIONS.get(marker.head.lemma_.lower())
        return (entry[0], flip(entry[1])) if entry else None

    entry = _DIMENSIONS.get(marker.lemma_.lower())
    if marker.dep_ == "amod" and marker.head.pos_ == "NOUN":
        noun = marker.head
        key = _NOUN_DIMENSIONS.get(noun.lemma_.lower())
        if key is not None:
            # "higher ELEVATION", "lower POPULATION": the noun is the quantity
            # and the adjective only says which way. A magnitude word outside
            # the adjective table ('greater', 'larger') points up.
            if diminishing:
                return key, "min"
            return key, entry[1] if entry else "max"
        if noun.dep_ == "attr" and entry is not None:
            return entry
        return None

    return entry


def _superlative_form(comp) -> str:
    """'longer' -> 'longest', 'more populous' -> 'most populous'."""
    if comp.dep_ == "advmod" and comp.head.pos_ in ("ADJ", "ADV"):
        word = _IRREGULAR_SUPERLATIVE.get(comp.lower_, comp.lower_)
        return f"{word} {comp.head.lower_}"
    word = comp.lower_
    if word in _IRREGULAR_SUPERLATIVE:
        return _IRREGULAR_SUPERLATIVE[word]
    # 'longer' -> 'longest', 'bigger' -> 'biggest', 'heavier' -> 'heaviest'.
    return word[:-2] + "est" if word.endswith("er") else word


def _by_dimension(superlative) -> tuple[str, str] | None:
    """
    "the largest country BY AREA" — an ambiguous adjective whose dimension the
    claim states outright, in a 'by' phrase on the ranked noun. Without this,
    'largest' declines for being ambiguous even when the claim has resolved it.
    """
    noun = superlative
    while noun.pos_ != "NOUN" and noun.head is not noun:
        noun = noun.head
    for prep in noun.children:
        if prep.dep_ == "prep" and prep.lower_ == "by":
            for pobj in prep.children:
                key = _NOUN_DIMENSIONS.get(pobj.lemma_.lower())
                if pobj.dep_ == "pobj" and key is not None:
                    smaller = superlative.lemma_.lower() in _DIMINISHING | {"small"}
                    return key, "min" if smaller else "max"
    return None


# Prepositions that place a category somewhere: 'the highest mountain IN the
# world', 'the longest river IN Africa', 'the second-highest peak ON Earth'.
_SCOPE_PREPS = {"in", "on", "of", "within", "across", "throughout"}

# Scopes that restrict nothing. "The highest mountain on Earth" and "the
# highest mountain" rank the same set.
_WORLD_SCOPES = {"world", "earth", "planet", "globe"}


def _category_token(marker):
    """The noun a superlative or comparative ranks: walk up to the first NOUN."""
    node = marker
    while node.pos_ != "NOUN" and node.head is not node:
        node = node.head
    return node if node.pos_ == "NOUN" else None


def _scope_and_bare(noun, marker_ids: set[int]) -> tuple[str | None, bool, bool]:
    """
    (scope, bare, unread_by) for the ranked noun.

    scope      the place the ranking is within ('Africa'), or None.
    bare       True when nothing but the marker narrows the category. "the
               highest UNCLIMBED mountain", "the earliest EUROPEAN PERMANENT
               settlement", "the tallest mountain CLIMBED WITHOUT OXYGEN" rank
               a subset, and a rival from the whole class does not refute them
               — so whatever compares the subject against arbitrary members of
               the class (the Wikidata route) must know.
    unread_by  the claim names its own dimension in a 'by' phrase this module
               cannot read ('highest mountain BY PROMINENCE'). The adjective's
               dimension must then not be used: 'highest' would say elevation.

    'above sea level' is allowed: it says which height, the one Wikidata and
    every list of mountains already use.
    """
    scope, bare, unread_by = None, True, False
    for child in noun.children:
        if child.i in marker_ids or child.dep_ in ("det", "predet", "punct"):
            continue
        if child.dep_ == "prep":
            pobj = next((g for g in child.children if g.dep_ == "pobj"), None)
            if pobj is None:
                bare = False
            elif child.lower_ in _SCOPE_PREPS and scope is None:
                scope = _noun_span(pobj) or None
            elif child.lower_ in ("above", "below") and pobj.lemma_.lower() == "level":
                pass
            elif child.lower_ == "by" and pobj.lemma_.lower() in _NOUN_DIMENSIONS:
                pass
            else:
                bare = False
                unread_by = unread_by or child.lower_ == "by"
            continue
        bare = False                 # amod, compound, acl, relcl, nummod, poss ...
    return scope, bare, unread_by


def _ranking_frame(claim: str, superlative) -> ComparisonFrame | None:
    """
    answer_1 = Question(<category query>)
    label    = Match(answer_1, subject)

    The whole point is in the first line: the query names the CATEGORY the
    superlative ranks within, not the claim's subject. Asking "what was the
    earliest European settlement in the US?" reaches Saint Augustine; asking
    about Jamestown never can.

    The rank travels inside the query ('second highest peak Earth'), so the
    Question step asks the claim's own question rather than "who is first?".
    """
    subject = ner.extract_subject(claim)
    query = ner.build_predicate_query(claim)

    # Both are required. Without a subject there is nothing to compare the
    # answer against; without a category query the retrieval would fall back to
    # the subject and reproduce the failure this exists to fix.
    if not subject or not query:
        return None

    rank, ordinal = ner.superlative_rank(superlative)
    words = [ordinal.lower_] if ordinal is not None else []
    if superlative.tag_ == "RBS" and superlative.head.pos_ in ("ADJ", "ADV"):
        words += [superlative.lower_, superlative.head.lower_]
    else:
        words.append(superlative.lower_)

    # Read from `claim`, not from `query`. The category query has had its
    # prepositions stripped, which is exactly the information needed to tell
    # the ranked type from the scope it is ranked within.
    type_word = ner.superlative_type_word(claim) or ""

    marker_ids = {superlative.i} | ({ordinal.i} if ordinal is not None else set())
    if superlative.tag_ == "RBS":
        marker_ids.add(superlative.head.i)          # 'most POPULOUS'
    noun = _category_token(superlative)
    scope, bare, unread_by = (_scope_and_bare(noun, marker_ids) if noun is not None
                              else (None, False, False))

    dimension = None if unread_by else (_dimension_of(superlative)
                                        or _by_dimension(superlative))
    return ComparisonFrame(
        subject=subject,
        marker=" ".join(words),
        dimension=dimension[0] if dimension else None,
        direction=dimension[1] if dimension else "max",
        rank=rank,
        category=type_word or None,
        scope=scope,
        bare=bare,
        query=query,
        type_word=type_word,
    )


def _universal_frame(claim: str, comp, obj) -> ComparisonFrame | None:
    """
    "The Nile is longer than any other river in Africa" — rank 1, category
    'river', scope 'Africa'. Treated exactly like "the longest river in Africa",
    which is what it asserts. Compared pairwise, as it used to be, the second
    entity was 'river in Africa' and Quantity() went looking for its length.
    """
    subject = ner.extract_subject(claim)
    if not subject:
        return None

    modifiers = [t.text for t in obj.children
                 if t.dep_ in ("amod", "compound") and t.lower_ != "other"]
    category = " ".join(modifiers + [obj.lemma_.lower()])
    # 'other' and the universal determiner are the comparison's own machinery,
    # not narrowings of the category.
    scope, bare, _ = _scope_and_bare(obj, {t.i for t in obj.children
                                           if t.lower_ == "other"})
    marker = _superlative_form(comp)
    query = ner._sanitize_query(" ".join(w for w in (marker, category, scope) if w))

    dimension = _dimension_of(comp)
    return ComparisonFrame(
        subject=subject,
        marker=marker,
        dimension=dimension[0] if dimension else None,
        direction=dimension[1] if dimension else "max",
        rank=1,
        category=category,
        scope=scope,
        bare=bare,
        query=query,
        type_word=obj.lemma_.lower(),
    )


def _pairwise_frame(claim: str, comp, other: str) -> ComparisonFrame | None:
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

    dimension = _dimension_of(comp)
    if dimension is None:
        # No measurable dimension this program can read (an ambiguous adjective
        # like 'larger', or a noun like velocity/rate it has no units for).
        # Decline rather than compare whatever numbers happen to be nearby — the
        # claim falls through to the verifier.
        return None

    return ComparisonFrame(
        subject=subject,
        marker=comp.lower_,
        dimension=dimension[0],
        direction=dimension[1],
        rank=None,
        reference=other,
        query=f"{subject} {other}",
    )


# ── Frame from the model (FRAME_PARSER = "llm") ───────────────────────────────
# The model fills SLOTS, it does not write a program. ProgramFC's generated
# programs need a parser and a fallback for malformed output; a JSON schema
# enforced by Ollama's constrained decoding removes the malformed case, and the
# validation in _frame_from_llm removes the hallucinated one: every name the
# model returns must be a span of the claim.
_FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "comparison":  {"type": "string", "enum": ["none", "pairwise", "ranking"]},
        "subject":     {"type": "string"},
        "reference":   {"type": "string"},
        "superlative": {"type": "string"},
        "dimension":   {"type": "string", "enum": [*_DIMENSION_SPECS, "none"]},
        "direction":   {"type": "string", "enum": ["max", "min"]},
        "rank":        {"type": "integer", "minimum": 1, "maximum": 10},
        "category":    {"type": "string"},
        "scope":       {"type": "string"},
    },
    "required": ["comparison", "subject", "reference", "superlative",
                 "dimension", "direction", "rank", "category", "scope"],
}

_FRAME_PROMPT = """Classify how this claim compares or ranks things. Reply with JSON only.

comparison: "pairwise" if it compares the subject with ONE named entity,
  "ranking" if it places the subject at a position within a whole category
  (superlatives, "second-largest", "than any other"), otherwise "none".
subject: the entity the claim is about, copied exactly from the claim.
reference: pairwise only - the other entity, copied exactly. Otherwise "".
superlative: ranking only - the superlative word(s) WITHOUT the ordinal
  ("highest", "most populous"). For "longer than any other", write "longest".
dimension: the measured quantity, one of: {dimensions}; "none" if it is not
  one of these or is ambiguous ("larger" could be area or population).
direction: "max" if the larger value wins (taller, deepest), "min" if the
  smaller one does (lower, least populous).
rank: 1 for a plain superlative, 2 for "second-...", and so on. 1 for pairwise.
category: ranking only - what is ranked ("peak", "European settlement").
scope: ranking only - where it is ranked ("Earth", "Africa"), or "".

Examples:
"Mount Everest is taller than K2." ->
{{"comparison":"pairwise","subject":"Mount Everest","reference":"K2","superlative":"","dimension":"height","direction":"max","rank":1,"category":"","scope":""}}
"K2 is the second-highest peak on Earth." ->
{{"comparison":"ranking","subject":"K2","reference":"","superlative":"highest","dimension":"height","direction":"max","rank":2,"category":"peak","scope":"Earth"}}
"Paris hosted the 2024 Olympics." ->
{{"comparison":"none","subject":"Paris","reference":"","superlative":"","dimension":"none","direction":"max","rank":1,"category":"","scope":""}}

Claim: "{claim}" ->"""


def _frame_from_llm(claim: str,
                    rules: ComparisonFrame | None) -> ComparisonFrame | None:
    """
    The model's frame, or None when it is unusable — in which case build_frame
    keeps the rules' frame.

    VALIDATION, SLOT BY SLOT. subject, reference and category must occur in the
    claim (no names from the model's own knowledge); a pairwise frame needs a
    measurable dimension, exactly as _pairwise_frame does; and where the parse
    found an ordinal, the model's rank must agree with it — the parse is right
    about which word is there even when it is wrong about what it means.
    """
    try:
        raw = llm.generate(
            ANSWER_MODEL,
            _FRAME_PROMPT.format(claim=claim, dimensions=", ".join(_DIMENSION_SPECS)),
            think=False,
            keep_alive=KEEP_ALIVE,
            format=_FRAME_SCHEMA,
            options={"num_ctx": 4096, "num_predict": 160, "temperature": 0,
                     "top_p": 1, "top_k": 1, "seed": 0},
        ).text
        data = json.loads(raw)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"[Program] Frame model unavailable ({type(e).__name__}) — using the parse.")
        return None
    if not isinstance(data, dict):
        return None

    low = claim.lower()

    def span(key: str) -> str:
        value = str(data.get(key) or "").strip()
        return value if value and value.lower() in low else ""

    kind = data.get("comparison")
    subject = span("subject")
    if kind not in ("pairwise", "ranking") or not subject:
        return None

    dimension = data.get("dimension")
    dimension = dimension if dimension in _DIMENSION_SPECS else None
    direction = data.get("direction") if data.get("direction") in ("max", "min") else "max"

    if kind == "pairwise":
        reference = span("reference")
        if not reference or dimension is None:
            return None
        return ComparisonFrame(subject=subject, marker="", dimension=dimension,
                               direction=direction, rank=None, reference=reference,
                               query=f"{subject} {reference}", source="llm")

    rank = data.get("rank")
    category = span("category")
    superlative = str(data.get("superlative") or "").strip().lower()
    if not isinstance(rank, int) or not 1 <= rank <= 10 or not category or not superlative:
        return None
    if rules is not None and rules.rank is not None and rules.rank != rank:
        print(f"[Program] Frame model said rank {rank}, the parse says "
              f"{rules.rank} — using the parse.")
        return None

    scope = span("scope") or None
    marker = " ".join(w for w in (_ORDINAL_TEXT.get(rank), superlative) if w)
    query = ner._sanitize_query(" ".join(w for w in (marker, category, scope) if w))
    return ComparisonFrame(
        subject=subject, marker=marker, dimension=dimension, direction=direction,
        rank=rank, category=category, scope=scope, query=query,
        type_word=category.split()[-1].lower(), source="llm",
        # A one-word category carries no narrowing modifier.
        bare=len(category.split()) == 1,
    )


def _program_for(frame: ComparisonFrame) -> Program:
    """The steps a frame executes. The steps are what is shown; the frame drives."""
    if frame.rank is not None:
        steps = [
            Step("answer_1", "Question", (frame.query,)),
            Step("label", "Match", ("answer_1", frame.subject)),
        ]
        if frame.dimension is not None and frame.rank == 1:
            # Consulted only when Match refutes — see _numeric_corroborate.
            spec = _DIMENSION_SPECS[frame.dimension]
            steps.append(Step("check", "Compare",
                              ("answer_1", frame.subject, spec.name, frame.direction)))
        return Program(frame=frame, steps=steps)

    spec = _DIMENSION_SPECS[frame.dimension]
    return Program(
        frame=frame,
        steps=[
            Step("q_1", "Quantity", (frame.subject, spec.name, spec.family)),
            Step("q_2", "Quantity", (frame.reference, spec.name, spec.family)),
            Step("label", "Compare", ("q_1", "q_2", frame.direction)),
        ],
    )


# ── Handlers ──────────────────────────────────────────────────────────────────
def _retrieve(query: str, top_k: int,
              rank_against: str | None = None,
              rank: int | None = None) -> tuple[list[str], str]:
    """
    Chunks for a free-text query, ranked by retriever.rank_chunks — the same
    scorer and selection the verifier path uses.

    `query` selects the ARTICLES. `rank_against` selects the CHUNKS within them,
    defaulting to `query` so every existing caller is unaffected.

    They differ only on _handle_question's fallback path, where the subject name
    fetches the right article but ranks its chunks near-uniformly — the subject
    appears in most of them, so it discriminates nothing. The chunk that answers
    is the one matching the CATEGORY, so fetch by subject and rank by category.

    `rank`, for a ranking claim, adds the table rows at that rank ahead of the
    ranked chunks — see _ranked_rows.
    """
    articles = retriever.fetch_articles(query)
    if not articles:
        return [], ""
    chunks = retriever.chunk_articles(articles)
    # Ranked exactly as the verifier path ranks its evidence — see
    # retriever.rank_chunks for why this must never be a separate scorer again.
    ranked, url = retriever.rank_chunks(rank_against or query, chunks, top_k=top_k)

    rows = _ranked_rows(articles, rank) if rank is not None else []
    if not rows:
        return ranked, url
    print(f"[Program] TableRank({rank}) -> {len(rows)} row(s): "
          + " | ".join(r[:90] for r, _ in rows))
    # The row's article is the citation: it is the evidence that states the rank.
    return [r for r, _ in rows] + ranked, rows[0][1]


# A table's rank column, as table_sentences renders its header.
_RANK_HEADER = r'(?:rank|no\.?|#|pos\.?|position)'

# Rows taken per lookup. One table per article is the usual case; two allows a
# second list to disagree with the first, which NLI then sees.
RANKED_ROWS_MAX = 2


def _ranked_rows(articles: list[tuple[str, str]], rank: int) -> list[tuple[str, str]]:
    """
    TableRank(k): the rows whose rank cell is k, from the retrieved articles'
    tables — selected by VALUE, not by similarity.

    WHY NOT RANK THE ROWS LIKE ANY OTHER CHUNK. Measured on 'List of highest
    mountains on Earth' with all 120 rows as chunks, the query 'second highest
    peak Earth' selected the rows ranked 11, 12, 57 and 60 — the ones whose
    names contain 'Peak'. Nothing in a similarity score maps 'second' to
    'Rank: 2'; every row looks alike except for that one cell. A rank is a
    lookup on structure, which is how table fact-checking treats it (TabFact's
    argmax/argmin program functions, KoPL's comparison operators), so it is
    done as one.

    WHAT KEEPS A WRONG TABLE OUT. A row with 'Rank: 2' in a list ranked by some
    other quantity (discharge, prominence) is still a candidate here. The NLI
    kind check is what rejects it: the row must entail the claim restated about
    its name, and "List of rivers by discharge - Rank: 2; River: Congo" does
    not entail "the Congo is the second-longest river".
    """
    pattern = re.compile(rf' - {_RANK_HEADER}\s*:\s*{rank}\.?;', re.I)
    rows = []
    for content, url in articles:
        if retriever.TABLE_MARKER not in content:
            continue
        for line in content.split(retriever.TABLE_MARKER, 1)[1].split("\n"):
            if pattern.search(line):
                rows.append((line.strip(), url))
                break                # one row per article: its first ranked table
        if len(rows) >= RANKED_ROWS_MAX:
            break
    return rows


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
    # Short words are skipped as noise ('of', 'the') — except a token carrying a
    # digit, which is a name: 'K2' was rejected as "not in the evidence" while
    # the evidence was K2's own article. Short tokens must match as whole words,
    # so 'k2' is not found inside 'k2-18b'.
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower())
             if len(w) >= 4 or any(c.isdigit() for c in w)]
    if not words:
        return False
    return all(
        w in haystack if len(w) >= 4
        else re.search(rf'(?<![a-z0-9-]){re.escape(w)}(?![a-z0-9-])', haystack)
        for w in words
    )


def _handle_question(query: str,
                     fallback_queries: tuple[str, ...] = (),
                     type_word: str = "",
                     claim: str = "",
                     subject: str = "",
                     rank: int | None = None,
                     rejected: list | None = None,
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
    article is never read on the category path.

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
                                         claim=claim, subject=subject, rank=rank,
                                         rejected=rejected)
    if answer is not None:
        return answer, chunks, url, False

    for fallback in fallback_queries:
        if not fallback or fallback == query:
            continue
        print(f"[Program] Question('{query}') found nothing — retrying with "
              f"fallback query '{fallback}'.")
        fb_answer, fb_chunks, fb_url = _question_once(
            fallback, rank_against=query, type_word=type_word,
            claim=claim, subject=subject, rank=rank, rejected=rejected,
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
                   subject: str = "",
                   rank: int | None = None,
                   rejected: list | None = None) -> tuple[str | None, list[str], str]:
    """
    One Question attempt: retrieve, ask, validate. Returns the entity or None.

    `rank_against` is passed through to _retrieve; the prompt and the kind check
    both use it in preference to `query`, so the model is still asked the
    CATEGORY question even when the articles were fetched by subject name.

    `type_word` is supplied by the caller from the claim's own parse. When it is
    empty the prompt asks for a name of any kind rather than guessing a type —
    see ner.superlative_type_word for why the type cannot be read off the query.

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

    The claim already carries the answer to "what kind": the frame's type_word
    ('settlement', 'building', 'river'). Telling the model directly — answer with the
    {type_word}'s own name, not a person connected to it — removes the
    ambiguity that let the wrong pick happen, rather than catching it once it
    already has. The downstream check stays; a prompt instruction is a
    preference an 8B-class model can still miss, not a guarantee, and
    defense in depth costs nothing extra here.
    """
    asked = rank_against or query

    chunks, url = _retrieve(query, QUESTION_TOP_K, rank_against=rank_against, rank=rank)
    if not chunks:
        print(f"[Program] Question('{query}') — no evidence retrieved.")
        return None, [], ""

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
        # Grounded but not confirmed: the evidence names it, without stating
        # it holds the record. Kept for a caller that can check it another way
        # (the Wikidata route); never a verdict on its own.
        if rejected is not None:
            rejected.append(answer)
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

      - the type word itself: _type_word (since removed) returned 'states' (the scope) and then
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


def _sentences(text: str) -> list[str]:
    """Cheap sentence split. No spaCy: this runs over long evidence chunks."""
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _head_and_modifiers(name: str) -> tuple[str | None, set[str]]:
    """
    ('African bush elephant') -> ('elephant', {'african', 'bush'})

    The HEAD is what the name is a kind of; the MODIFIERS say which one. English
    compounds are right-headed — 'land animal' is a kind of animal, not a kind
    of land — so the head is the last content word.
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
         'Jamestown'. REFUTES — the rival has already passed
         _answer_is_right_kind, so the evidence states it holds the record.

    KNOWN GAP: an alias sharing no words with the subject ('Loxodonta
    africana') reaches tier 3 and is reported as a rival — a wrong REFUTES.
    Resolving that needs canonical titles from Wikipedia redirects, not a
    cleverer string rule.
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


# ── Quantities ────────────────────────────────────────────────────────────────
# UNIT CONVERSION IS PINT'S, NOT A HAND-WRITTEN TABLE. The table this replaced
# (_UNIT_TO_BASE) listed metres, feet and miles and nothing else, so every area
# and mass claim was unreadable — and the 'count' and 'year' families declared
# in the dimension lexicon had no entry in it at all, which meant every
# population comparison returned NOT ENOUGH INFO without ever reading a number.
#
# Pint is pinned below 0.25 in requirements.txt: 0.24.4 is the newest release
# pip resolves for this project's Python 3.10 venv.
_UREG = pint.UnitRegistry()

_FAMILY_DIMENSIONALITY = {
    "length": _UREG.get_dimensionality("[length]"),
    "area":   _UREG.get_dimensionality("[length] ** 2"),
    "mass":   _UREG.get_dimensionality("[mass]"),
}

# A number not glued to a word ('K2'), then a unit: 'm', 'ft', 'km2', 'km²',
# 'sq mi', 'square kilometres'. The trailing guard rejects 'km/h' and 'm/s' —
# a speed in a sentence about length is not a length.
_NUM = r'(?<![\w.])(\d[\d,]*(?:\.\d+)?)'
_SCALE_WORD = r'(?:\s*(thousand|million|billion))?'
_MEASURE_RE = re.compile(
    _NUM + _SCALE_WORD + r'\s*((?:sq(?:uare)?\.?\s+)?[A-Za-z]+(?:²|2)?)(?![\w/²])', re.I
)

# Unit-less tallies: '1.4 billion', '8,336,817'. The scale word belongs to the
# number, not to a unit — in both patterns.
_COUNT_RE = re.compile(_NUM + _SCALE_WORD + r'(?![\w%])', re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _to_base(number: str, unit: str, family: str) -> float | None:
    """
    `number unit` in the family's SI base unit (m, m², kg), or None.

    PINT PARSES ENGLISH. Measured: pint reads 'in' as inch and 't' as tonne, so
    "8,849 in 2020" would be a length of 8,849 inches. A unit is therefore
    refused when it is an English stop word or a single letter other than 'm'
    — the only one-letter unit encyclopedia prose uses after a number.
    """
    u = unit.lower().strip()
    u = re.sub(r'^sq(?:uare)?\.?\s+', 'square ', u)
    u = re.sub(r'(?<=[a-z])2$', '**2', u)
    bare = u.split()[-1]
    if bare in STOP_WORDS or (len(bare) == 1 and bare != "m"):
        return None
    try:
        q = _UREG.Quantity(float(number.replace(",", "")), u)
    except (pint.errors.PintError, ValueError, TypeError):
        return None
    if q.dimensionality != _FAMILY_DIMENSIONALITY.get(family):
        return None
    return float(q.to_base_units().magnitude)


def _measurements(sentence: str, family: str) -> list[tuple[float, str]]:
    """Every (value_in_base_units, as_written) of `family` in one sentence."""
    if family == "count":
        found = []
        for m in _COUNT_RE.finditer(sentence):
            digits, scale = m.group(1), m.group(2)
            # A bare four-digit number is a year ('In 2020 the population...'),
            # not a tally, unless written with a separator or a scale word.
            if (not scale and "," not in digits and "." not in digits
                    and 1000 <= int(digits) <= 2100):
                continue
            value = float(digits.replace(",", "")) * _SCALE.get((scale or "").lower(), 1)
            found.append((value, m.group(0).strip()))
        return found

    found = []
    for m in _MEASURE_RE.finditer(sentence):
        value = _to_base(m.group(1), m.group(3), family)
        if value is not None:
            # '9.8 million km²': the scale multiplies the number, not the unit.
            value *= _SCALE.get((m.group(2) or "").lower(), 1)
            found.append((value, m.group(0).strip()))
    return found


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

    spec = _DIMENSION_SPECS.get(dimension)
    keywords = set(spec.keywords) if spec else {dimension}
    entity_tokens = [t.lower() for t in re.findall(r"[A-Za-z]+", entity) if len(t) >= 4]

    best = None
    for chunk in chunks:
        for sentence in _sentences(chunk):
            low = sentence.lower()
            if not any(k in low for k in keywords):
                continue
            names_entity = any(t in low for t in entity_tokens) if entity_tokens else False

            for value, display in _measurements(sentence, family):
                # Lower sorts first: entity-naming sentences beat others, then
                # larger values beat smaller ones.
                rank = (0 if names_entity else 1, -value)
                if best is None or rank < best[0]:
                    best = (rank, value, display, sentence)

    if best is None:
        print(f"[Program] Quantity('{entity}', {dimension}) — no value found in any "
              f"sentence mentioning {sorted(keywords)}.")
        return None, "", chunks, url

    _, value, display, sentence = best
    print(f"[Program] Quantity('{entity}', {dimension}) -> {display} ({value:g} base)")
    print(f"[Program]   from: {sentence.strip()[:110]}")
    return value, display, chunks, url


def _handle_compare(a, b, subject: str, other: str, dimension: str,
                    disp_a: str, disp_b: str, direction: str = "max") -> tuple[str, str]:
    """
    Compare(q_1, q_2) -> (label, rationale). Arithmetic, in Python.

    `direction` is the claim's: 'max' asserts the subject's value is the larger
    ("taller than"), 'min' that it is the smaller ("less populous than").
    """
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

    if a == b:
        return ("NOT ENOUGH INFO",
                f"The evidence gives both {subject} and {other} as {disp_a}, which "
                f"does not settle the comparison.")

    holds = a > b if direction == "max" else a < b
    if holds:
        return ("SUPPORTS",
                f"The evidence gives {subject} as {disp_a} and {other} as "
                f"{disp_b}, so the claim holds.")
    return ("REFUTES",
            f"The evidence gives {subject} as {disp_a} and {other} as "
            f"{disp_b}, so the claim is false.")


def _numeric_corroborate(rival: str, subject: str, frame: ComparisonFrame):
    """
    Does the rival Question() named actually BEAT the subject on the claim's
    own dimension? -> (True | False | None, trace_line, chunks, rationale)

    The second, independent route for a rank-1 REFUTES. The kind check has
    already required a passage ENTAILING the rival's record; when the
    superlative names a measurable dimension a check from different evidence
    is available too: read both values and compare them in the claim's
    direction.

        True   the rival's value wins       -> REFUTES is corroborated
        False  the subject's value wins     -> the two routes disagree -> NEI
        None   a value could not be read    -> fall back to the mention check

    Rank 1 only. For "the second-highest", a rival named second-highest may
    stand above or below the subject, so the comparison corroborates nothing.
    """
    spec = _DIMENSION_SPECS[frame.dimension]
    a, disp_a, chunks_a, _ = _handle_quantity(rival, spec.name, spec.family)
    b, disp_b, chunks_b, _ = _handle_quantity(subject, spec.name, spec.family)

    # "Does the rival hold the record against the subject?" — a claim about
    # the rival, in the superlative's own direction.
    label, rationale = _handle_compare(a, b, rival, subject, spec.name,
                                       disp_a, disp_b, frame.direction)
    verdict = {"SUPPORTS": True, "REFUTES": False}.get(label)
    if verdict is not None:
        # _handle_compare's wording ("so the claim holds") is about the
        # rival's claim; the caller quotes this inside a verdict on the
        # subject's, so state the values without a conclusion.
        rationale = (f"the evidence gives {rival} as {disp_a} and {subject} "
                     f"as {disp_b}.")
    outcome = {True: "rival wins", False: "subject wins", None: "undecided"}[verdict]
    trace = (f"check = Compare(Quantity('{rival}'), Quantity('{subject}'), "
             f"{spec.name}, {frame.direction}) -> "
             f"{disp_a or 'UNKNOWN'} vs {disp_b or 'UNKNOWN'} -> {outcome}")
    return verdict, trace, chunks_a + chunks_b, rationale


def _wikidata_title(name: str) -> str:
    """An entity name as an English Wikipedia title: first letter capitalised."""
    name = re.sub(r'^(the|a|an)\s+', '', name.strip(), flags=re.I)
    return name[:1].upper() + name[1:]


def _format_value(value: float, dimension: str) -> str:
    family = _DIMENSION_SPECS[dimension].family
    if family == "area":
        return f"{value / 1e6:,.0f} km²"
    if family == "length":
        return f"{value:,.0f} m"
    if family == "mass":
        return f"{value:,.0f} kg"
    return f"{value:,.0f}"


def _wikidata_alone_applies(frame: ComparisonFrame) -> bool:
    """May Wikidata values alone refute this claim? See the call site."""
    return (frame.rank == 1 and frame.dimension is not None and frame.bare
            and frame.world_scope
            and wikidata.supports(frame.type_word, frame.dimension))


def _wikidata_beats(rival: str, subject: str, frame: ComparisonFrame):
    """
    Does `rival` beat `subject` on the frame's dimension, by Wikidata's values?
    -> (True | False | None, trace_line, rationale_fragment)

    None when either value is missing, the two resolve to the same entity, or
    they are equal — the caller then falls back to whatever it had.
    """
    if frame.dimension is None or not wikidata.supports(frame.type_word, frame.dimension):
        return None, f"check = Wikidata({frame.type_word}, {frame.dimension}) -> not mapped", ""

    a = wikidata.value(_wikidata_title(rival), frame.type_word, frame.dimension)
    b = wikidata.value(_wikidata_title(subject), frame.type_word, frame.dimension)
    if a is None or b is None or a.qid == b.qid or a.value == b.value:
        return (None, f"check = Wikidata('{rival}', '{subject}', {frame.dimension}) "
                      f"-> undecided", "")

    beats = a.value > b.value if frame.direction == "max" else a.value < b.value
    why = (f"Wikidata gives {rival} ({a.qid}) as "
           f"{_format_value(a.value, frame.dimension)} and {subject} ({b.qid}) as "
           f"{_format_value(b.value, frame.dimension)}.")
    line = (f"check = Wikidata('{rival}' {_format_value(a.value, frame.dimension)}, "
            f"'{subject}' {_format_value(b.value, frame.dimension)}, {frame.direction}) "
            f"-> {'rival wins' if beats else 'subject wins'}")
    return beats, line, why


# ── Execution ─────────────────────────────────────────────────────────────────
def execute(program: Program, claim: str) -> ProgramOutcome | None:
    """
    Run a program and return its verdict with provenance.

    Returns None if execution could not proceed at all, which sends the caller
    back to single-shot verification rather than leaving the claim unchecked.
    """
    frame = program.frame
    if frame.rank is not None:
        category, subject = frame.query, frame.subject
        rejected: list[str] = []

        answer, chunks, url, from_fallback = _handle_question(
            category,
            fallback_queries=(subject,),
            type_word=program.type_word,
            claim=claim,
            subject=subject,
            rank=frame.rank,
            rejected=rejected,
        )
        label, rationale = _handle_match(answer, subject)

        program.trace = [
            f"answer_1 = Question('{category}'"
            + (f", fallback='{subject}'" if from_fallback else "")
            + f") -> {answer or 'UNKNOWN'}",
            f"label = Match(answer_1, '{subject}') -> {label}",
        ]

        # A SUPPORTS FROM THE SUBJECT'S OWN ARTICLE IS NOT GIVEN. It is the
        # subject asserting its own record, and that is exactly where a false
        # superlative hides: live, Jamestown's article says "the first permanent
        # ENGLISH settlement in the Americas", NLI read that as entailing
        # "earliest EUROPEAN permanent settlement in the United States" at
        # p=0.97, and the claim — false, Saint Augustine is older — came back
        # SUPPORTS. This used to be a note appended to the SUPPORTS; it is now
        # NOT ENOUGH INFO, which is this module's failure direction.
        #
        # What the fallback is still for: finding a RIVAL. A subject's article
        # naming someone else as the record holder ("Kangchenjunga ... third
        # highest, after Everest and K2") is evidence against its own claim,
        # and that REFUTES stands.
        if from_fallback and label == "SUPPORTS":
            label = "NOT ENOUGH INFO"
            rationale = (
                f"The category search for '{category}' named no record holder "
                f"the evidence confirms. '{subject}''s own article states the "
                f"record, but a subject's own article cannot establish a "
                f"ranking against rivals it need not mention, so no verdict is "
                f"given."
            )

        # A REFUTES already rests on the rival being ENTAILED, at p >= 0.9, by a
        # passage stating the claim's record about it — _answer_is_right_kind
        # is the corroboration. When the dimension is measurable a second,
        # independent route is available and the numbers must not disagree.
        #
        # THE SUBJECT-ARTICLE MENTION CHECK THAT USED TO RUN HERE IS GONE. It
        # was added when the kind check accepted at p=0.5, and it was a
        # substring search: does the subject's article contain the rival's
        # name? Under the raised threshold it only ever downgraded correct
        # verdicts — live, "Kangchenjunga is the second-highest peak" went to
        # NOT ENOUGH INFO with 'K2' entailed at p=0.97, because four chunks of
        # the Kangchenjunga article did not happen to name K2. And it would
        # have confirmed a wrong rival that the subject's article merely names.
        #
        # Structured values first: Wikidata's are typed and unit-normalised,
        # where the prose route reads numbers out of sentences (and once read
        # Mauna Kea as 9,330 m). The prose route stays as the fallback for an
        # entity or category Wikidata cannot answer.
        if label == "REFUTES" and frame.dimension is not None and frame.rank == 1:
            numeric, line, num_rationale = _wikidata_beats(answer, subject, frame)
            num_chunks = []
            if numeric is None:
                numeric, line, num_chunks, num_rationale = _numeric_corroborate(
                    answer, subject, frame)
            program.trace.append(line)
            chunks = chunks + num_chunks
            if numeric is True:
                rationale += f" The values agree: {num_rationale}"
            elif numeric is False:
                label = "NOT ENOUGH INFO"
                rationale = (
                    f"A search for '{category}' named '{answer}' as holding this "
                    f"distinction, but the values read from the evidence put "
                    f"'{subject}' ahead: {num_rationale} The two routes "
                    f"disagree, so no verdict is given."
                )

        # NO CONFIRMED HOLDER, BUT A NAMED CANDIDATE THAT BEATS THE SUBJECT.
        # "K2 is the highest mountain in the world": the table row for Everest
        # entails "Everest is the highest mountain" at only p=0.80, so the kind
        # check rejected it and the claim had no answer. It does not need one.
        # The claim is refuted by ANY member of the class that beats K2, and
        # Wikidata gives Everest 8,850 m against K2's 8,611 m.
        #
        # Sound only under three conditions, all checked by
        # _wikidata_alone_applies: rank 1 (a rival above the subject refutes
        # "the highest", but says nothing about "the second-highest"); a
        # world-wide scope ("the highest mountain in Pakistan" is not refuted
        # by Everest); and a bare category ("the highest UNCLIMBED mountain"
        # is not refuted by a climbed one). The candidate must also be a member
        # of the category's Wikidata class — wikidata.value checks that.
        if answer is None and rejected and _wikidata_alone_applies(frame):
            for candidate in dict.fromkeys(rejected):
                if _handle_match(candidate, subject)[0] == "SUPPORTS":
                    continue             # the candidate IS the subject
                beats, line, why = _wikidata_beats(candidate, subject, frame)
                program.trace.append(line)
                if beats:
                    label = "REFUTES"
                    rationale = (
                        f"The evidence names '{candidate}', and {why} A "
                        f"{frame.type_word} that exceeds '{subject}' exists, so "
                        f"'{subject}' is not the {frame.marker} {frame.type_word}."
                    )
                    break

        program.chunks, program.source_url = chunks, url

    else:
        spec = _DIMENSION_SPECS[frame.dimension]
        subject, other = frame.subject, frame.reference

        a, disp_a, chunks_a, url_a = _handle_quantity(subject, spec.name, spec.family)
        b, disp_b, chunks_b, _ = _handle_quantity(other, spec.name, spec.family)

        label, rationale = _handle_compare(
            a, b, subject, other, spec.name, disp_a, disp_b, frame.direction
        )

        program.trace = [
            f"q_1 = Quantity('{subject}', {spec.name}) -> {disp_a or 'UNKNOWN'}",
            f"q_2 = Quantity('{other}', {spec.name}) -> {disp_b or 'UNKNOWN'}",
            f"label = Compare(q_1, q_2, {frame.direction}) -> {label}",
        ]
        program.chunks = chunks_a + chunks_b
        program.source_url = url_a

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
