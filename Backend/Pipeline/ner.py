import re
import spacy

# ── spaCy model ───────────────────────────────────────────────────────────────
# Run once to download: python -m spacy download en_core_web_sm
NLP = spacy.load("en_core_web_sm")

# ── Entity priority for query construction ────────────────────────────────────
# The weights are assigned based on semantic density and historical retrieval value.
# Higher index = higher priority for anchoring a search index query.
ENTITY_PRIORITY = {
    # 1. DATE: Lowest anchor value. Chronological markers provide no contextual meaning
    #    unless paired with a concrete event, subject, or organization.
    "DATE"   : 1,
    
    # 2. NORP: Nationalities, religious groups, or political factions. 
    #    Useful for macro-demographics but too broad on their own for single-hop retrieval.
    "NORP"   : 2,
    
    # 3. ORG: Companies, institutions, or open-source engineering groups (e.g., "OpenAI", "Apache").
    #    Acts as an excellent contextual domain wrapper for factual claims.
    "ORG"    : 3,

    # 3b. PRODUCT / WORK_OF_ART / LOC: Named objects, creative works, and
    #     non-GPE locations (mountain ranges, regions). Mid-priority anchors.
    "PRODUCT"     : 3,
    "WORK_OF_ART" : 4,
    "LOC"         : 4,

    # 4. GPE: Geopolitical entities (countries, cities, sovereign states).
    #    Provides an immediate geographical constraint to eliminate out-of-domain noise.
    "GPE"    : 4,

    # 5. EVENT: Named battles, tech conferences, or historic occurrences (e.g., "WWII", "ROSCon").
    #    Strongly binds specific timeline occurrences together.
    "EVENT"  : 5,

    # 5b. FAC: Named facilities — buildings, monuments, bridges, airports
    #     (e.g., "Eiffel Tower", "Statue of Liberty"). Previously missing,
    #     which made NER blind to the very landmarks test claims are about:
    #     spaCy tags them FAC, not GPE/ORG, so only the DATE survived and
    #     every landmark query fell through to keyword fallback.
    "FAC"    : 5,

    # 5c. LAW: Named documents made into law or policy — treaties, acts,
    #     amendments, doctrines, plans (e.g., "the Marshall Plan", "the Treaty
    #     of Versailles", "the First Amendment"). THE SAME BUG AS FAC ABOVE,
    #     one label later.
    #
    #     spaCy tagged it correctly the whole time; extract_entities dropped it,
    #     because the filter is `ent.label_ in ENTITY_PRIORITY` and LAW had no
    #     key. A live run:
    #
    #         claim    "This financial rescue program was formally designated
    #                   as the Marshall Plan."
    #         spaCy    [('the Marshall Plan', 'LAW')]
    #         filtered []
    #         queries  ['financial rescue program', 'program']
    #         article  Troubled Asset Relief Program        <- 2008, not 1948
    #         verdict  NOT ENOUGH INFO
    #
    #     The one term that identifies the topic was sitting in the claim,
    #     recognised by the tagger, and thrown away — so the query fell through
    #     to the grammatical subject, 'financial rescue program' matched a
    #     bank-bailout article from sixty years later, and the verifier
    #     correctly reported that this evidence says nothing about the claim.
    #     A retrieval failure wearing a verdict's clothes.
    #
    #     Priority 5, alongside EVENT and FAC: a named document is as specific
    #     an anchor as a named building. Deliberately above GPE — for this
    #     claim 'the Marshall Plan' is worth more than any country in it.
    "LAW"    : 5,

    # 6. PERSON: Individual historical figures, authors, or developers.
    #    Highest priority because unique proper names are the strongest search anchors in knowledge bases.
    "PERSON" : 6
}

# ── spaCy stopwords for fallback keyword extraction ───────────────────────────
STOPWORDS = NLP.Defaults.stop_words


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Named Entity Recognition
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities(fact: str) -> list[tuple[str, str]]:
    """
    Runs spaCy NER on an isolated atomic fact.
    Filters the results to retain only the specific categories mapped in 
    the AFEV search-anchoring hierarchy (ENTITY_PRIORITY).

    Entity Classifications Explained:
    - PERSON: Unique human agents (e.g., "Alan Turing")
    - EVENT: Fixed, named historical occurrences (e.g., "Industrial Revolution")
    - GPE: Geopolitical map coordinates (e.g., "California")
    - ORG: Bound organizational bodies (e.g., "MIT")
    - NORP: Broad human group associations (e.g., "Byzantines")
    - DATE: Specific temporal constraints (e.g., "2026")
    """
    doc = NLP(fact)

    entities = [
        (ent.text.strip(), ent.label_)
        for ent in doc.ents
        if ent.label_ in ENTITY_PRIORITY
    ]

    print(f"[NER] Extracted entities: {entities}")
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Entity Filtering and Prioritization
# ─────────────────────────────────────────────────────────────────────────────
def filter_entities(entities: list[tuple[str, str]]) -> list[str]:
    """
    Sorts extracted entities by structural priority and enforces grounding rules.
    
    Grounding Rule (Date Isolation):
    A standalone 'DATE' entity (e.g., "in 1995") possesses zero unique semantic 
    meaning in a dense vector database or standard token search without context. 
    Therefore, a DATE entity is dropped entirely unless accompanied by an entity 
    with a priority score >= 2 (NORP, ORG, GPE, EVENT, or PERSON).

    Example:
        Fact: "Python was conceived by Guido van Rossum in 1989"
        Parsed: [("Guido van Rossum", "PERSON"), ("Python", "ORG"), ("1989", "DATE")]
        Sorted Output: ["Guido van Rossum", "Python", "1989"]
    """
    if not entities:
        return []

    # Sort entities based on their explicit knowledge-base retrieval weights (Descending)
    sorted_entities = sorted(
        entities,
        key=lambda e: ENTITY_PRIORITY.get(e[1], 0),
        reverse=True
    )

    # A DATE is dropped whenever ANY other entity survives to anchor the query.
    #
    # The rule used to be the opposite way round — a date was kept as soon as a
    # high-priority entity existed — which let "the beginning of the 17th
    # century" occupy half of a two-term query and retrieve the *17th century
    # BC* article. A date contributes nothing to a Wikipedia lookup even beside
    # a good anchor: it either matches the year article (noise) or narrows
    # nothing. It is retained only when it is the sole entity available, so the
    # keyword fallback still has something to work with.
    has_non_date = any(label != "DATE" for _, label in sorted_entities)

    filtered = []
    for text, label in sorted_entities:
        if label == "DATE" and has_non_date:
            print(f"[NER] Dropping date entity from query: '{text}'")
            continue
        filtered.append(text)

    print(f"[NER] Filtered entities: {filtered}")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0b — Sentence Segmentation
# ─────────────────────────────────────────────────────────────────────────────
def split_sentences(text: str) -> list[str]:
    """
    Split input text into sentences using spaCy's parser-driven segmentation.

    Used by the extractor to give each sentence its own decomposition budget.
    Previously the loop ran once over the whole document, so a sentence the
    model refused to decompose faithfully consumed every remaining iteration
    and starved everything after it — an observed run spent all its retries
    arguing about sentence 1 and never looked at sentence 2 at all.

    Parser-based segmentation (not a regex on '.') because abbreviations,
    decimals and initials make period-splitting wrong often enough to matter:
    "Bosporus is 3.7 km long. It is narrow." must not become three sentences.
    """
    doc = NLP(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    print(f"[NER] Segmented input into {len(sentences)} sentence(s).")
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0c — Claim Quality Gates (Fluency + Check-worthiness)
# ─────────────────────────────────────────────────────────────────────────────
# Both gates run on a parse we already need, so they cost ~10ms and no model
# call. That matters: admitting one claim to retrieval + verification costs
# ~152s measured, so a gate priced in milliseconds pays for itself the first
# time it fires, whereas an LLM-based check-worthiness classifier at ~10s per
# claim only breaks even if it rejects more than ~6.6% of claims.

# Predicates that signal opinion rather than checkable fact.
_SUBJECTIVE_LEMMAS = {
    "beautiful", "ugly", "best", "worst", "greatest", "amazing", "terrible",
    "wonderful", "awful", "boring", "interesting", "important", "nice",
    "better", "worse", "favourite", "favorite", "stunning", "lovely",
    "seem", "feel", "believe", "think", "deserve", "should", "ought",
}


# A sentence ending in one of these is cut off, whatever the tagger decided.
# Determiners, prepositions, conjunctions and relativisers all REQUIRE a
# following complement; none can legitimately end a declarative claim.
_TRUNCATING_FINAL_WORDS = {
    # determiners
    "the", "a", "an", "this", "that", "these", "those", "its", "their", "his",
    "her", "our", "your", "my",
    # prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "onto",
    "about", "between", "among", "through", "during", "over", "under",
    # conjunctions and relativisers
    "and", "or", "but", "which", "who", "whom", "whose", "because", "while",
}


def check_fluency(claim: str) -> str | None:
    """
    Is this a well-formed declarative sentence? (Ullrich et al., 2025, §4.1)

    Returns None when fluent, otherwise a plain-language defect string that the
    extractor feeds back into its retry prompt.

    Deliberately BINARY and structural rather than a perplexity/SLOR score with
    a tuned cutoff. A continuous score needs a threshold calibrated against
    labelled data that does not exist yet, and would drift; these checks are
    facts about the parse tree, identical on every run and free to compute.
    Keep SLOR for offline evaluation, where a continuous scale is the point.
    """
    doc = NLP(claim)

    root = next((t for t in doc if t.head == t), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return "it has no main verb — a claim must be a full sentence, not a fragment"

    if not any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in doc):
        return "it has no subject — state who or what the claim is about"

    if claim.rstrip().endswith("?"):
        return "it is a question, not a statement"

    # Truncation is checked LEXICALLY as well as by part of speech, because the
    # POS branch alone cannot be trusted here. A tagger is trained on
    # well-formed sentences, so its output is least reliable on exactly the
    # malformed input this gate exists to catch: given "The Bosporus connects
    # the", en_core_web_sm does not label the stranded "the" as DET — it fits
    # the slot after a transitive verb and gets tagged as a nominal instead.
    # The check silently passed and a truncated claim was admitted.
    #
    # Function words are a CLOSED CLASS — a finite list that gains no new
    # members — so an explicit set is not a heuristic standing in for a model,
    # it is the complete answer. No tagger can beat a lookup table here.
    # Skip trailing punctuation: a model that emits "…connects the." is just as
    # truncated as one that emits "…connects the", and doc[-1] would otherwise
    # be the full stop.
    last = next((t for t in reversed(doc) if not t.is_punct), None)
    if last is not None and (
        last.pos_ in ("ADP", "CCONJ", "SCONJ", "DET")
        or last.text.lower() in _TRUNCATING_FINAL_WORDS
    ):
        return "it ends mid-phrase and appears truncated"

    if not (3 <= len(doc) <= 60):
        return "it is too short or too long to be a single atomic claim"

    return None


# Pronouns that point BACK at something said earlier. A claim containing one
# of these is not intelligible on its own, and its antecedent is recoverable
# from the source document — so the extractor can and should resolve it.
_ANAPHORIC = {
    "it", "its", "they", "them", "their", "theirs",
    "he", "him", "his", "she", "her", "hers",
    "this", "that", "these", "those", "there", "then",
}

# Pronouns with NO antecedent anywhere in the document — generic or deictic
# usage ("you can walk across it" means "one can"). These cannot be resolved
# by any amount of rereading, so demanding resolution would send the extractor
# into a rejection loop it can never escape. They are left alone; the query
# path already declines to anchor on a pronoun subject.
_GENERIC_PRONOUNS = {"you", "your", "one", "we", "our", "us", "i", "my"}


# ── Polarity ──────────────────────────────────────────────────────────────────
# Negation markers, matched by surface form and by the 'neg' dependency.
#
# BOTH tests are needed and neither is redundant. The dependency label catches
# 'not'/'n't' attached to a verb, which is the common case, but spaCy does not
# label negative quantifiers and determiners that way — "no other river",
# "neither claim", "without evidence" carry negation with no neg arc anywhere.
# The lexicon catches those. Conversely the lexicon alone would miss a 'neg'
# arc on a token spelled unusually.
#
# CLOSED CLASS. Negation in English is a finite list, so an explicit set here is
# the complete answer rather than a heuristic standing in for a model.
_NEGATION_MARKERS = {
    "not", "n't", "no", "never", "none", "neither", "nor", "nothing",
    "nobody", "nowhere", "without", "cannot", "nope",
}


def _negation_count(text: str) -> int:
    """How many negation markers `text` carries."""
    doc = NLP(text)
    seen: set[int] = set()
    for token in doc:
        if token.dep_ == "neg" or token.text.lower() in _NEGATION_MARKERS:
            seen.add(token.i)
    return len(seen)


def check_negation(source: str, claim: str) -> str | None:
    """
    Does the claim carry the same polarity as the text it came from?

    Returns None when polarity matches, otherwise a plain-language defect string
    the extractor feeds back into its retry prompt.

    THE HOLE THIS CLOSES. The faithfulness gate is containment over content
    words, and it filters tokens shorter than MIN_CONTENT_WORD_LENGTH (4) before
    comparing. 'not' is three characters. So the single most meaning-changing
    word in English is invisible to the check that exists to catch meaning
    changes, and a claim can be reversed while passing every gate.

    Not hypothetical. An observed run turned

        source:  "It is universally acknowledged as longer than the Nile by all
                  international cartographers."          ('It' = the Amazon)
        claim:   "The Nile is not universally acknowledged as the world's
                  longest river."

    — wrong subject, inserted negation, meaning reversed — and every gate passed
    it, because each surviving content word does appear in the document and the
    inserted 'not' was filtered out before comparison. The verifier then refuted
    the fabrication and the run reported a confident REFUTES on a claim the
    input never made. That is worse than any missed verdict: the system did not
    fail to catch an error, it INVENTED one.

    WHY NOT JUST LOWER MIN_CONTENT_WORD_LENGTH. Because it would admit 'the',
    'and', 'of', 'in' into the faithfulness vocabulary and flood the gate with
    function words, and because negation is not a vocabulary question anyway —
    "no other river is longer" and "every other river is shorter" use different
    words to say the same thing, while "is" and "is not" differ by one token and
    say opposite things. Polarity needs its own check, which is what this is.

    SCOPED TO THE SOURCE CLAUSE, NOT THE DOCUMENT. A document containing a
    negation anywhere would otherwise license one anywhere, which is precisely
    the licence the failure above took.

    COUNTS, NOT PRESENCE. "did not fail" and "failed" differ by two markers and
    mean the same thing; comparing booleans would call them identical. Comparing
    counts also flags a dropped negation, which reverses meaning just as
    thoroughly as an added one.

    FALSE REJECTIONS ARE ACCEPTABLE HERE. A faithful rewrite can legitimately
    change polarity — "greater than any other" into "no other is greater" — and
    this will reject it. That costs one extraction call (~10s) and a retry. A
    false acceptance costs a fabricated verdict, which is the failure this whole
    project exists to prevent. The asymmetry decides the design, as it did for
    the containment gate.
    """
    source_negations = _negation_count(source)
    claim_negations = _negation_count(claim)

    if claim_negations == source_negations:
        return None

    if claim_negations > source_negations:
        return (
            "it negates something the source text does not — the source carries "
            f"{source_negations} negation(s) and this claim carries "
            f"{claim_negations}. State what the text says, not its opposite"
        )

    return (
        "it drops a negation the source text carries — the source has "
        f"{source_negations} negation(s) and this claim has {claim_negations}. "
        "Removing a 'not' reverses the meaning"
    )


def check_decontextualized(claim: str) -> str | None:
    """
    Is the claim intelligible standing alone? (AIDA's 'Independent', via
    Wright et al. 2022 / Ullrich et al. 2025.)

    Returns None when self-contained, otherwise the unresolved reference.

    This is the metric the pipeline was missing, and its absence was visible:
    "You can walk between them with Yavuz Sultan Selim Bridge" reached
    retrieval with 'them' unresolved on the first pass. The fix is not to
    forbid pronouns — generic 'you' has no referent to recover — but to
    require that ANAPHORIC ones be replaced with wording from the document.

    Relativisers are exempt: "the strait that connects the Black Sea" is a
    perfectly self-contained claim, and its 'that' is grammatical machinery
    rather than a dangling reference. spaCy tags those WDT/WP, which is how
    they are told apart from the demonstrative 'that'.
    """
    doc = NLP(claim)

    # A pronoun is only a decontextualization failure when its antecedent lies
    # OUTSIDE the claim. Two cases that look alike and are not:
    #
    #   "The Republic of Sale traces ITS origins to the 17th century."
    #        -> 'its' is bound by the subject in the same clause. The claim is
    #           perfectly intelligible alone. Nothing to resolve.
    #
    #   "THEY were expelled by the order of the Spanish king."
    #        -> nothing in the claim says who. Genuinely broken standalone.
    #
    # The earlier version rejected both, so an entirely correct extraction was
    # refused three times in a row and the sentence was abandoned with zero
    # facts. Tracking whether a candidate antecedent has already been seen
    # separates them: a pronoun preceded by a noun or proper noun in the same
    # claim can bind to it; one that appears before any noun cannot.
    #
    # This is a heuristic, not coreference resolution. It mishandles cataphora
    # ("Because IT was costly, the bridge was delayed") and cannot arbitrate
    # between two candidate antecedents. See the note in the module docstring
    # about promoting this to a real coref model.
    seen_candidate_antecedent = False

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            seen_candidate_antecedent = True
            continue

        if token.pos_ != "PRON":
            continue
        if token.tag_ in ("WDT", "WP", "WP$", "WRB"):   # relativiser
            continue

        lower = token.text.lower()
        if lower in _GENERIC_PRONOUNS:
            continue
        if lower in _ANAPHORIC and not seen_candidate_antecedent:
            return (
                f"'{token.text}' refers to something outside the claim — "
                f"replace it with the name it stands for, using wording from the document"
            )

    return None


def check_worthy(claim: str) -> str | None:
    """
    Is this claim worth sending to retrieval + verification?

    Returns None when check-worthy, otherwise the reason it is not.

    Rejects the two cases that cost ~152s to discover the expensive way and
    always come back NOT ENOUGH INFO — indistinguishable, in the final report,
    from a genuine retrieval failure:

      - Pure opinion: "The Bosporus is beautiful." No evidence can settle it.
      - No referent:  "It is very long." Nothing nameable to retrieve on.

    Conservative by design. A false reject silently drops a checkable claim,
    which is worse than paying 152s, so anything carrying a proper noun is
    admitted even if it also carries a subjective word — "Istanbul is the most
    populous city in Turkey" is both evaluative-sounding and perfectly
    verifiable.
    """
    doc = NLP(claim)

    has_proper_noun = any(t.pos_ == "PROPN" for t in doc)
    has_number      = any(t.like_num for t in doc)

    if has_proper_noun or has_number:
        return None

    subjective = {
        t.lemma_.lower() for t in doc
        if t.lemma_.lower() in _SUBJECTIVE_LEMMAS
    }
    if subjective:
        return (
            f"it expresses an opinion ({', '.join(sorted(subjective))}) about no "
            f"named entity — there is no evidence that could confirm or deny it"
        )

    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return "it names nothing concrete that could be looked up"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — Grammatical Subject Recovery
# ─────────────────────────────────────────────────────────────────────────────
# Determiners and quantifiers stripped from the front of a recovered subject.
#
# Was just (the|a|an), which let this through:
#
#     "Both Python and C++ are widely used..."  ->  subject 'Both Python'
#                                               ->  query   'Both Python C++'
#
# 'Both' is a predeterminer attached to the subject, so it lands inside the
# subject's subtree and survives into the search string. Wikipedia has no
# article about Both Python. These words quantify a noun phrase without naming
# anything, so none of them belongs in a query — and unlike the input character
# allowlist, this IS a closed class worth enumerating.
_LEADING_ARTICLE = re.compile(
    r'^(the|a|an|both|all|each|every|either|neither|some|any|this|that|these|those)\s+',
    re.IGNORECASE,
)

# Dependency relations that hang extra CLAUSES or alternatives off a noun.
# They belong to the sentence, not to the entity's name, and dragging them into
# a search query destroys it: "the bridge that connects Asia and Europe" would
# otherwise be queried in full.
_SPAN_STOP_DEPS = {"relcl", "acl", "advcl", "appos", "conj", "cc", "punct"}

# Upper bound on subject length in tokens. A name longer than this is a parse
# artefact rather than an entity worth searching for.
_MAX_SUBJECT_TOKENS = 7


def _subject_span(head) -> str:
    """
    Text of the subject noun phrase, INCLUDING prepositional attachments.

    Written to replace a doc.noun_chunks lookup that silently truncated names.
    spaCy's noun chunks are *base* noun phrases and exclude PP attachments, so
    "The Republic of Sale" chunked as "The Republic" and the query was built
    from the bare word "Republic". That retrieved Spain in the 17th century and
    17th century BC, and the claim came back NOT ENOUGH INFO despite being
    stated verbatim in the Republic of Salé article.

    Walking token.subtree keeps "of Sale" attached, but the subtree also drags
    in relative clauses and coordinated alternatives — so anything reached
    through a relation in _SPAN_STOP_DEPS is pruned, and the result is the
    contiguous run of surviving tokens containing the head.
    """
    doc = head.doc

    # Compare by INDEX, never by identity. spaCy builds a new Token object on
    # every attribute access, so `cur is not head` is true even when both refer
    # to the same word — the ascent never terminates and the request hangs.
    # The ROOT guard is the second half of the same lesson: at the root,
    # token.head is the token itself, so without it the walk would spin there.
    keep = set()
    for tok in head.subtree:
        cur, pruned = tok, False
        while cur.i != head.i:
            if cur.dep_ in _SPAN_STOP_DEPS:
                pruned = True
                break
            if cur.head.i == cur.i:      # reached ROOT without passing head
                break
            cur = cur.head
        if not pruned:
            keep.add(tok.i)

    # Contiguity matters: a gap means the intervening words were pruned, and
    # splicing across it would invent a phrase the sentence never contained.
    start = end = head.i
    while start - 1 in keep:
        start -= 1
    while end + 1 in keep:
        end += 1

    # OVER-LONG SPANS ARE NOT TRUNCATED — they are abandoned.
    #
    # This used to clip to the first _MAX_SUBJECT_TOKENS tokens, which turns a
    # too-long span into a MID-PHRASE FRAGMENT. Observed on
    #
    #   "The title of the world's longest river belongs to the Amazon River."
    #
    # the subject span ran past the cap and was clipped to
    #
    #   'title of the world's longest'
    #
    # — cut before the noun it modifies. That fragment was then promoted to the
    # head of the search query, producing 'title of the world longest the Amazon
    # River': three queries, 66 chunks across 5 articles, to answer a claim
    # whose only real anchor is 'Amazon River'. It happened to score 0.86 and
    # find the right passage, which is luck, not design.
    #
    # A fragment is worse than nothing here, because the query builder cannot
    # tell the difference and will faithfully search for it. So when the span
    # overruns, fall back to the head token alone — and only if the head is a
    # PROPER NOUN, i.e. a name worth searching for.
    #
    # The PROPN condition is the part that took two attempts. Falling back to
    # the bare head noun unconditionally gave 'title' for the sentence above:
    # still not the entity, still promoted to the front of the query, and now
    # generic enough to match anything. A common-noun head means the parse found
    # a description rather than a name, and the honest answer is that no subject
    # was recovered — the entity path still has NER to fall back on.
    if end - start + 1 > _MAX_SUBJECT_TOKENS:
        if head.pos_ == "PROPN":
            # THE WHOLE NAME, NOT THE LAST WORD OF IT.
            #
            # This returned head.text alone, which on
            #
            #     "The First Transcontinental Railroad in the United States..."
            #
            # gave the subject 'Railroad' — a word that names nothing in
            # particular and retrieves an article about rail transport in
            # general. The head is the last token of a multi-word name, and
            # taking it by itself discards the part that identifies the entity.
            #
            # Walking outward over contiguous PROPN tokens recovers
            # 'First Transcontinental Railroad'. Contiguity matters: it stops at
            # the first non-PROPN, so 'in the United States' stays out and the
            # span cannot creep back over the cap it was invoked to escape.
            span_start = span_end = head.i
            while span_start - 1 >= 0 and doc[span_start - 1].pos_ == "PROPN":
                span_start -= 1
            while span_end + 1 < len(doc) and doc[span_end + 1].pos_ == "PROPN":
                span_end += 1

            name = doc[span_start: span_end + 1].text
            print(f"[NER] Subject span exceeds {_MAX_SUBJECT_TOKENS} tokens — "
                  f"falling back to the proper noun '{name}'.")
            return name

        print(f"[NER] Subject phrase exceeds {_MAX_SUBJECT_TOKENS} tokens and heads "
              f"on the common noun '{head.text}' — no subject promoted.")
        return ""

    return doc[start:end + 1].text


def extract_subject(fact: str) -> str | None:
    """
    Recover the grammatical subject of a claim via the dependency parse.

    WHY THIS EXISTS: NER and the query builder can both silently discard the
    entity the claim is actually about.

      Claim : "Bosporus is located between Africa and Europe."
      NER   : [('Africa', 'LOC'), ('Europe', 'LOC')]        <- no Bosporus
      Query : "Africa Europe"

    The Bosporus article was therefore never retrieved, the verifier saw only
    chunks about African settlement patterns, and it returned NOT ENOUGH INFO
    for a claim that Wikipedia flatly refutes. Two independent causes:
    en_core_web_sm often misses a bare proper noun with no determiner ("the
    Bosporus" is recognised far more reliably than "Bosporus"), and even when
    NER does fire, build_query() caps at the top-2 priority entities — where
    the subject can lose to its own objects, since LOC and GPE tie.

    The dependency parser is the right tool here because it is far more robust
    than the NER model for this particular question: identifying the nsubj of
    a clause needs only syntax, not knowledge of what the token refers to. It
    finds "Bosporus" whether or not any gazetteer has heard of it.

    Returns the subject noun phrase with any leading article stripped, or None
    for pronoun subjects ("It connects the Black Sea"), which make useless
    search queries and should fall through to the entity-based path.
    """
    doc = NLP(fact)

    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        if token.pos_ == "PRON":
            print(f"[NER] Subject '{token.text}' is a pronoun — not usable as a query anchor.")
            return None

        subject = _LEADING_ARTICLE.sub("", _subject_span(token)).strip()
        if subject:
            print(f"[NER] Grammatical subject: '{subject}'")
            return subject

    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Query Construction
# ─────────────────────────────────────────────────────────────────────────────
# ── Intent-aware query planning ───────────────────────────────────────────────
# Superlative and ordinal markers. A claim carrying one of these asserts
# UNIQUENESS within a category, and that is the class this whole mechanism
# exists for.
#
# Detected by TAG rather than by a word list wherever possible: JJS is the
# superlative adjective tag ('tallest', 'earliest', 'largest'), RBS the
# superlative adverb ('most', 'least'). The lexicon covers what the tagger
# misses — 'first' and 'only' are usually tagged ORDINAL or ADJ, and 'sole' and
# 'unique' are plain adjectives, yet all four assert the same uniqueness.
_SUPERLATIVE_TAGS = {"JJS", "RBS"}
_SUPERLATIVE_LEMMAS = {
    "first", "earliest", "oldest", "last", "latest", "only", "sole",
    "unique", "foremost", "premier", "original",
}

# Words that describe the claim's rhetorical framing rather than its content.
# 'Jamestown is CELEBRATED AS the earliest...' — the celebrating is not what
# needs looking up, the chronology is. Dropping them shortens the query to the
# assertion itself.
_FRAMING_LEMMAS = {
    "celebrate", "know", "regard", "consider", "recognise", "recognize",
    "acknowledge", "describe", "call", "name", "say", "believe", "report",
    "widely", "universally", "commonly", "generally", "often", "traditionally",
}

# Upper bound on the predicate query. Wikipedia's search degrades on long
# strings, and past this length the query stops naming a category and starts
# paraphrasing the claim — which retrieves the claim's own subject again, the
# exact failure this is meant to avoid.
_MAX_PREDICATE_TOKENS = 8


def is_superlative(token) -> bool:
    """
    Does this token RANK something, or is it just part of a name?

    A CAPITALISED WORD INSIDE A NAME IS NOT DOING THE GRAMMATICAL WORK ITS LEMMA
    SUGGESTS. 'first' is in _SUPERLATIVE_LEMMAS because "the first settlement"
    ranks. But observed:

        "The First Transcontinental Railroad in the United States was officially
         completed on May 10, 1869, at Promontory Summit, Utah."

    'First' there is a component of a proper name — the railroad is CALLED that
    — and the sentence ranks nothing at all. Treating it as a superlative sent
    the whole claim down the program path, built the predicate query
    '10 1869 Promontory Summit Utah', and asked a model which entity held that
    distinction. It answered 'Promontory Summit', and only a corroboration check
    stopped a confident REFUTES on a true claim.

    spaCy already distinguishes the two: inside a name the token is tagged
    PROPN, while a genuine superlative adjective is JJS (or ADJ for the lexicon
    entries). So the test is one condition, and it costs nothing — the parse
    exists either way.

    NNP is excluded alongside PROPN because the fine-grained tag and the coarse
    POS disagree often enough on title-case tokens that checking one and not the
    other leaves a gap.
    """
    if token.pos_ == "PROPN" or token.tag_ == "NNP":
        return False
    return token.tag_ in _SUPERLATIVE_TAGS or token.lemma_.lower() in _SUPERLATIVE_LEMMAS


def superlative_type_word(fact: str) -> str | None:
    """
    The noun a superlative claim ranks WITHIN — 'settlement', 'animal', 'zone'.

    Returns None when the claim carries no superlative, or when no noun governs
    the one it has.

    READS THE CLAIM, NOT THE QUERY. program._type_word answers the same question
    from build_predicate_query's output, and its docstring records three separate
    failures caused by doing so: that output is "a keyword fragment, not a
    sentence", so en_core_web_sm re-parses it by the rightward-compounding rule
    and roots the whole span on its last word. Each fix narrowed which last word
    got taken — chunk root, then first NOUN, then last NOUN — and every version
    was a better guess at a structure the fragment no longer contains.

    The fragment lost that structure because the prepositions were stripped.
    "the largest living terrestrial animal ON EARTH TODAY" becomes 'largest
    living terrestrial animal Earth today', where nothing distinguishes the
    category from the scope; in the original sentence the scope sits under a
    preposition and the category does not.

    THE LAST-NOUN RULE HOLDS ONLY WHILE THE SCOPE IS A PROPER NOUN. _type_word's
    own comment says so: "THE SCOPE ENTITY IS ALWAYS PROPN, BY CONSTRUCTION ...
    the trailing scope ('United States', 'East Asia') is always the proper noun
    at the end". True for those examples, false in general — 'on the planet',
    'in history', 'on Earth today' all end in a COMMON noun, so the last NOUN is
    the scope and the rule returns it. Measured:

        'expansive subtropical arid zone planet'        -> 'planet'   (want 'zone')
        'gigantic creature history'                     -> 'history'  (want 'creature')
        'largest living terrestrial animal Earth today' -> 'today'    (want 'animal')
        'largest animal'                                -> 'animal'   correct
        'earliest European permanent settlement ...'     -> 'settlement' correct

    Fed downstream that produces the instruction "The answer is the today
    itself — its own name" for a question about an animal, and
    _answer_is_right_kind then type-checks the answer against 'today'.

    SO ASK THE SENTENCE INSTEAD. A superlative modifies the thing it ranks, so
    walking up the dependency tree from the superlative token to the nearest NOUN
    ancestor names the category directly. That walk crosses the adjective in
    'most expansive zone' (most -advmod-> expansive -amod-> zone) and stops at
    the first noun, and it never enters a prepositional phrase because the scope
    is a sibling of the category, not its head.

    Measured on the real parse, 5/5 including the case that motivated
    _type_word:

        "...the most expansive subtropical arid zone on the planet."   -> 'zone'
        "...the most gigantic creature known to have existed..."       -> 'creature'
        "...the largest living terrestrial animal on Earth today."     -> 'animal'
        "...the earliest European permanent settlement in the US."     -> 'settlement'
        "...the largest animal known to have lived."                   -> 'animal'

    Only the FIRST superlative is consulted. A claim with two ("the first and
    largest...") ranks one category, and both markers modify the same noun.
    """
    doc = NLP(fact)

    superlatives = [t for t in doc if is_superlative(t)]
    if not superlatives:
        return None

    node = superlatives[0]
    # Bounded by the sentence: doc roots are their own head in spaCy, so the
    # `node.head is node` test is the loop's terminator, not a guard against a
    # cycle that cannot occur.
    while True:
        if node.pos_ == "NOUN":
            return node.text.lower()
        if node.head is node:
            return None
        node = node.head


def build_predicate_query(fact: str) -> str | None:
    """
    A query for the CATEGORY a superlative claim ranks within, not for its subject.

    Returns None when the claim carries no superlative or ordinal — most claims
    do not, and for those the existing entity-anchored queries are correct.

    WHY THIS EXISTS. Queries are built from the claim's named entities, and a
    claim's entities describe its SUBJECT. That is fine when the claim names
    both sides of a comparison:

        "The Amazon is longer than the NILE"  ->  query 'Amazon River Nile'
                                              ->  List of river systems by length
                                              ->  REFUTES, correctly

    A superlative names only one side. The competitor is implicit, so there is
    nothing in the claim to retrieve it with:

        "Jamestown is the earliest EUROPEAN settlement in the United States"
                                              ->  query 'English settlement of Jamestown'
                                              ->  articles about Jamestown
                                              ->  SUPPORTS, wrongly

    Evidence that would REFUTE that claim is a passage about Saint Augustine —
    and a passage about Saint Augustine does not contain the word 'Jamestown'.
    The most discriminating term in the query is precisely the term the
    counter-evidence is structurally unable to hold, so no amount of ranking,
    reranking or top-k tuning can surface it. The query has to change.

    So: strip the subject and the framing, keep the superlative and what it
    ranges over.

        "earliest European permanent settlement United States"

    That asks "who actually was first?" instead of "tell me about Jamestown",
    and it is the query that finds Saint Augustine.

    DETERMINISTIC ON PURPOSE — no model call. It runs on a parse the pipeline
    already computes, costs no inference, and is testable offline. An LLM could
    write a more fluent query, but it would cost ~10s per claim and could not be
    unit-tested, and this is the cheapest thing that addresses the class.

    The result is an ADDITIONAL query, never a replacement: the subject query
    still has to run, because a claim about Jamestown does need the Jamestown
    article to confirm the parts that are true.
    """
    doc = NLP(fact)

    superlatives = [t for t in doc if is_superlative(t)]
    if not superlatives:
        return None

    # Everything the subject's subtree covers is dropped: the claim's subject is
    # what the existing queries already retrieve, and including it here would
    # reproduce them.
    subject_indices: set[int] = set()
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            subject_indices |= {t.i for t in token.subtree}

    start = min(t.i for t in superlatives)

    # THE SUPERLATIVE MARKER WAS BEING DELETED FROM THE QUERY THAT DEPENDS ON IT.
    #
    # English marks a superlative two ways. An INFLECTED superlative carries it
    # in the adjective ('largest', 'earliest', 'tallest'); an ANALYTIC one uses a
    # separate adverb ('most gigantic', 'most expansive', 'least developed').
    # Roughly, adjectives of one or two syllables inflect and longer ones take
    # 'most' — so the analytic form covers the longer, more technical half of the
    # vocabulary, which is the half encyclopedic prose is written in.
    #
    # Measured, en_core_web_sm on "the most expansive ... zone":
    #
    #     most        tag=RBS  pos=ADV  is_stop=True
    #     expansive   tag=JJ   pos=ADJ  is_stop=False
    #
    # 'largest' is JJS/ADJ/not-stop and passes the POS whitelist below. 'most' is
    # stopped by `is_stop` one line earlier, and the whitelist would have
    # excluded it anyway for being ADV — two independent reasons, either one
    # sufficient. Observed output:
    #
    #     "the most gigantic creature ... in history"
    #         -> 'gigantic creature history'
    #     "the most expansive subtropical arid zone on the planet"
    #         -> 'expansive subtropical arid zone planet'
    #
    # NEITHER QUERY ASKS FOR A RECORD. is_superlative accepts RBS (it is in
    # _SUPERLATIVE_TAGS, and the comment there names 'most' explicitly), so the
    # claim IS routed down the superlative path — and then handed a query with
    # the ranking removed. program._handle_question builds "What is the
    # expansive subtropical arid zone planet?", a question no superlative
    # appears in, and _handle_match compares the subject against whatever that
    # returns. Same class of defect as the '10 1869 Promontory Summit Utah'
    # query: a malformed question answered confidently.
    #
    # NOTHING BELOW CATCHES IT, because both guards test the query's SHAPE and
    # the damage is to its MEANING. `len(keep) < 2` sees five tokens.
    # `has_category_noun` sees 'arid', 'zone' and 'planet'. The query is
    # well-formed by every test this function applies; it simply no longer
    # ranks anything.
    #
    # NOT A GAP IN _SUPERLATIVE_LEMMAS. That set is read in exactly one place
    # (is_superlative, below) and is a fallback for what the TAGGER misses.
    # Measured: is_superlative('most') was already True via _SUPERLATIVE_TAGS,
    # so adding 'most' to the lemma set changes nothing — detection was never
    # the broken half. The defect is that this loop discards a token the
    # detector had already accepted.
    superlative_indices = {t.i for t in superlatives}

    keep = []
    has_category_noun = False
    for token in doc[start:]:
        if token.i in subject_indices or token.is_punct or token.is_space:
            continue

        # Kept unconditionally: the marker is the one word the query cannot do
        # without. This cannot change behaviour for inflected superlatives —
        # those already reach `keep` through the ADJ branch below, and the
        # `continue` prevents a second append.
        if token.i in superlative_indices:
            keep.append(token.text)
            # Preserved from the general branch. A superlative lemma tagged NOUN
            # ('the only', 'the original') used to satisfy the category check,
            # and silently withdrawing that would start rejecting claims this
            # function previously accepted.
            if token.pos_ == "NOUN":
                has_category_noun = True
            if len(keep) >= _MAX_PREDICATE_TOKENS:
                break
            continue

        if token.is_stop or token.lemma_.lower() in _FRAMING_LEMMAS:
            continue
        # Keep content words only. Adjectives matter here as much as nouns —
        # 'earliest', 'European' and 'permanent' are the qualifiers that define
        # the category, and dropping them would query 'settlement United States'.
        if token.pos_ in ("NOUN", "PROPN", "ADJ", "NUM"):
            keep.append(token.text)
            # A COMMON noun, specifically. The category a superlative ranks
            # within is always a common noun — settlement, river, building — and
            # never a name: nothing is "the earliest Utah".
            if token.pos_ == "NOUN":
                has_category_noun = True
        if len(keep) >= _MAX_PREDICATE_TOKENS:
            break

    if len(keep) < 2:
        # A single surviving word is the superlative alone ('the tallest'), which
        # names no category and would retrieve a disambiguation page.
        return None

    # NO CATEGORY NOUN, NO QUERY.
    #
    # A superlative ranks members of a CLASS, so the query has to name that
    # class or it is not asking the question. Observed when 'First' inside a
    # proper name was mistaken for a superlative:
    #
    #     predicate query: '10 1869 Promontory Summit Utah'
    #
    # A date and a place, ranking nothing. The Question handler was asked which
    # entity holds that distinction and answered 'Promontory Summit', which is
    # the only sane reply to an insane question.
    #
    # is_superlative now prevents that particular trigger, but the check belongs
    # here too: any future path that produces a category-less query should be
    # stopped before it reaches retrieval, not after. It also closes a gap in
    # program._answer_is_right_kind, whose type check FAILS OPEN when the query
    # has no type word — correct for a well-formed query, useless for this one.
    if not has_category_noun:
        print(f"[NER] Superlative-looking claim, but the predicate names no "
              f"category ({keep}) — no predicate query built.")
        return None

    query = _sanitize_query(" ".join(keep))
    if not query:
        return None

    print(f"[NER] Superlative claim — predicate query: '{query}'")
    return query


def _sanitize_query(text: str) -> str:
    """
    Shared query sanitizer used by build_query() and extract_queries().

    - Strips English possessive suffixes BEFORE deleting apostrophes —
      otherwise "Gustave Eiffel's" collapses into the garbage token
      "Gustave Eiffels" and the Wikipedia search misses the article.
      Covers both the straight apostrophe and the typographic U+2019.
    - Removes syntax characters that invalidate search engine parsers.
    - Collapses whitespace.
    """
    text = re.sub(r"[’']s\b", "", text)
    text = re.sub(r'[\"\'(){}\[\]]+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_query(filtered_entities: list[str], fact: str) -> str | None:
    """
    Synthesizes prioritized entities into a clean, targeted query string.

    Strategy for Information Density Optimization:
    - 1 entity  -> Extracted directly as the primary anchor.
    - 2 entities -> Combined sequentially to form a precise multi-hop intersection.
    - 3+ entities -> Caps extraction at the top 2 highest priority entities. This 
                     prevents over-specification, which causes document retrieval 
                     misses in standard indexing engines.
    """
    if not filtered_entities:
        return None

    # Cap at top 2 entities to maintain search breadth while ensuring query precision
    top_entities = filtered_entities[:2]

    query = _sanitize_query(" ".join(top_entities))

    print(f"[NER] Built query: '{query}'")
    return query if query else None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Fallback Keyword Extraction
# ─────────────────────────────────────────────────────────────────────────────
def fallback_keyword_extraction(fact: str) -> str:
    """
    Fallback mechanism triggered when spaCy fails to identify explicit named entities.
    Strips grammatical punctuation and language stopwords, extracting raw, 
    low-level descriptive content tokens.

    Example:
        Fact: "Large language models mitigate structural hallucinations."
        Filtered Keywords: ["language", "models", "mitigate", "hallucinations"]
    """
    # Isolate atomic alphabetic terms, casting to lowercase to align with index tokens
    tokens = re.findall(r'\b[a-zA-Z]+\b', fact.lower())

    # Filter language noise and drop shorthand words
    keywords = [
        t for t in tokens
        if t not in STOPWORDS and len(t) > 2
    ]

    # Constrain the search phrase to the first 4 high-density concepts to maintain focus
    query = " ".join(keywords[:4])

    print(f"[NER] Fallback query generated: '{query}'")
    return query


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Multi-Query Extraction (used by retriever.fetch)
# ─────────────────────────────────────────────────────────────────────────────
def extract_queries(fact: str, max_queries: int = 3) -> list[str]:
    """
    The Named Entity Query Extraction pipeline: the queries retrieval runs.

    Execution Flow:
    [Atomic Fact] -> extract_entities() -> filter_entities() -> build_query()
                                                                     |
                                                 (If Query is Empty) v
                                                      fallback_keyword_extraction()

    Single-query retrieval has a brittle failure mode: the query anchors on
    the top-2 priority entities, so ONE hallucinated or mis-typed entity
    poisons the entire evidence pool for the fact. Observed examples:
    "Gustave Eiffel Bedford Basin" retrieved a World Heritage Sites list,
    and spaCy tagging "Statue of Liberty" as ORG dropped it from the top-2
    cut entirely. Issuing an additional standalone query per top entity
    means at least one query still lands on the right article.

    Returns (deduplicated, priority order):
        1. The combined top-2 entity query
        2. One standalone query per top-2 entity — date entities excluded,
           since a bare "1889" retrieves the year article (pure noise)
        3. Keyword fallback if nothing else was produced
    """
    print(f"\n[NER] Processing atomic unit (multi-query): '{fact}'")

    entities = extract_entities(fact)
    filtered = filter_entities(entities)

    # Captured BEFORE subject promotion, which unconditionally puts the subject
    # into `filtered` and would make "did NER find anything?" unanswerable
    # afterwards. The bare-head fallback at the end of this function needs the
    # answer.
    filtered_entities_found = bool(filtered)

    # SUBJECT PROMOTION: whatever the claim is *about* must reach the index.
    #
    # The subject is MOVED to the front, not merely inserted when absent. The
    # earlier version only prepended a subject that NER had missed entirely,
    # which left the case that actually matters unfixed: NER finds the entity,
    # but ENTITY_PRIORITY ranks it below the claim's objects, so build_query()'s
    # top-2 cap discards it anyway. Observed exactly that on
    #
    #   "Yavuz Sultan Selim Bridge was located between Africa and Europe."
    #   filtered -> ['Africa', 'Europe', 'Yavuz Sultan Selim Bridge']   (LOC 4 > ORG 3)
    #   query    -> 'Africa Europe'                                     (bridge cut)
    #
    # Rebuilding the list rather than inserting also de-duplicates: an entity
    # matching the subject by substring either way ('Bosporus' vs 'the
    # Bosporus') is dropped from the tail so it cannot occupy a second slot.
    subject = extract_subject(fact)
    if subject:
        # UPGRADE BEFORE PROMOTING. If NER recognised a fuller form of the same
        # entity, use NER's string. Without this, a truncated subject would
        # evict the better name during de-duplication: the subject 'Republic'
        # matched the entity 'The Republic of Sale' by substring, so the entity
        # was dropped as a duplicate and the query was built from the fragment.
        # Promotion made retrieval worse than leaving it alone.
        for entity in filtered:
            if subject.lower() in entity.lower() and len(entity) > len(subject):
                print(f"[NER] Subject '{subject}' upgraded to NER's fuller form '{entity}'.")
                subject = entity
                break

        rest = [
            e for e in filtered
            if subject.lower() not in e.lower() and e.lower() not in subject.lower()
        ]
        if filtered[:1] != [subject]:
            print(f"[NER] Promoting subject '{subject}' to the head of the query.")
        filtered = [subject] + rest

    queries: list[str] = []

    combined = build_query(filtered, fact)
    if combined:
        queries.append(combined)

    for ent_text in filtered[:2]:
        if len(queries) >= max_queries:
            break
        q = _sanitize_query(ent_text)
        # Skip empty strings, standalone date-like entities, and duplicates
        if not q or re.fullmatch(r'[\d\s\-–to]+', q) or q in queries:
            continue
        queries.append(q)

    # INTENT-AWARE QUERY, appended last.
    #
    # Last, not first, and additional rather than replacing anything: the
    # entity-anchored queries are what confirm the parts of a claim that are
    # true, and they stay correct. This one exists only to reach the evidence
    # they structurally cannot — see build_predicate_query.
    #
    # It is allowed to exceed max_queries. That cap bounds how many entity
    # lookups a claim may spend, and a superlative claim asking one question the
    # other queries cannot ask is not the thing the cap is protecting against.
    # Cost is one Wikipedia fetch (~5-15s) against verifications that run
    # 100-600s.
    # BARE-HEAD FALLBACK for a multi-word subject when NER found nothing.
    #
    #     "Standard C++ relies primarily on manual memory management..."
    #     entities: []            (NER recognised nothing at all)
    #     query:    'Standard C++'
    #     result:   the C STANDARD LIBRARY article — wrong language
    #
    # diagnose_nei caught it correctly ("'c++' appears in 0 of 2 chunks"), but
    # nothing recovered. With no entities there was exactly one query, and one
    # bad query is a lost claim.
    #
    # The head token alone — 'C++' — retrieves the right article. Adding it
    # costs one fetch and only when NER produced nothing, so the common case is
    # untouched. This is a fallback for a recogniser that failed, not a general
    # query strategy: 'Standard' is doing no work here, and when NER is
    # working the entity list already carries the name.
    if subject and not filtered_entities_found and len(subject.split()) > 1:
        head_token = _sanitize_query(subject.split()[-1])
        if head_token and head_token not in queries and len(head_token) > 1:
            print(f"[NER] NER found no entities — adding bare head '{head_token}' "
                  f"as a fallback query.")
            queries.append(head_token)

    predicate_query = build_predicate_query(fact)
    if predicate_query and predicate_query not in queries:
        queries.append(predicate_query)

    if not queries:
        print("[NER] Structural resolution failed. Triggering semantic keyword fallback extraction.")
        queries.append(fallback_keyword_extraction(fact))

    print(f"[NER] Final query set: {queries}")
    return queries