"""
verification.py — one entry point for verifying a single atomic fact.

    claim_extractor ──► verify_claim ──┬─► program.try_verify        specialised: may decline
                                       └─► retriever.fetch → verifier.verify    default

WHY THIS IS NOT IN claim_extractor. Choosing a verification strategy, running
retrieval, classifying an NOT ENOUGH INFO and timing the stages are not
extraction. They lived in claim_extractor.grounded_verify because that is where
the AFEV loop first needed a verdict, not because they belonged there. Here,
program and verifier read as two strategies behind one decision point, and
claim_extractor depends on exactly one verification function.

THE PROGRAM PATH GOES FIRST, for the two claim types single-shot verification
cannot decide — superlatives and comparatives. See Pipeline/program.py for why
those two are different in kind rather than merely harder. try_verify returns
None for everything else, which is most claims, and also when its own
preconditions fail (no subject, no category query, execution error), so a claim
can never be lost to that branch.

BOTH PATHS BUILD THEIR RECORD THROUGH results.make_fact_result, so every key is
present on every result, and timing is measured here for both rather than by
each strategy for itself.
"""

import time

import Pipeline.ner as ner
import Pipeline.program as program
import Pipeline.retriever as retriever
import Pipeline.verifier as verifier
from Pipeline.results import FactResult, make_fact_result


def diagnose_nei(fact: str, chunks: list[str]) -> str:
    """
    Distinguish the two very different situations that both surface as NEI.

    'NOT ENOUGH INFO' currently collapses two states that call for opposite
    responses:

      RETRIEVAL_FAILURE — the evidence never mentions what the claim is about,
        so no verdict was ever possible. Recoverable: re-query and try again.
        This is what happened to "Bosporus is located between Africa and
        Europe": the query was built from Africa and Europe alone, the
        Bosporus article was never fetched, and the verifier was asked to
        adjudicate a strait using chunks about African rainfall. Its NEI was
        the correct response to bad evidence, not a statement about the claim.

      GENUINE — the right articles were retrieved and are authentically
        silent or ambiguous on this particular point. Terminal: no better
        query will help.

    The test is deliberately mechanical rather than another LLM call: does the
    claim's subject appear anywhere in the retrieved text? Zero mentions
    across every chunk means the evidence is not about the claim at all.

    Returns "RETRIEVAL_FAILURE", "GENUINE", or "UNKNOWN" when the subject
    could not be recovered (pronoun subject, parse failure) and the question
    cannot be decided this way.
    """
    subject = ner.extract_subject(fact)
    if not subject:
        return "UNKNOWN"

    # Match on the subject's PROPER NOUN, falling back to its longest token.
    #
    # Requiring the whole phrase would produce false negatives that look like
    # retrieval failures — retrieved prose says "the Bosporus", "Bosporus
    # Strait", "Bosphorus" — so some single token has to stand in for the
    # subject. Choosing WHICH token is the entire difficulty, and two previous
    # answers were wrong:
    #
    #   split()[-1]            The head word. Too permissive when that head is a
    #                          common noun: "Yavuz Sultan Selim Bridge" matched
    #                          on 'bridge' against a passage comparing tower
    #                          heights, and "water molecules" matched on
    #                          'molecules' in an article about the properties of
    #                          water. Both reported GENUINE; both were retrieval
    #                          failures.
    #
    #   max(split(), key=len)  The longest token, as a proxy for the rarest. It
    #                          fixes those two and then reproduces the same bug
    #                          whenever a common noun is simply longer:
    #
    #                            'English settlement of Jamestown'
    #                             English=7  settlement=10  Jamestown=9
    #                             -> picks 'settlement'
    #
    #                          which matches almost any colonial-era passage, so
    #                          the diagnosis is GENUINE by construction. Observed
    #                          on a real run, where it happened to agree with the
    #                          truth and therefore told the reader nothing.
    #
    # Length was never the property being reached for. The property is "does
    # this string name ONE thing?", and part of speech answers it directly: a
    # proper noun is a name, a common noun is a category. spaCy has already
    # tagged this text, so the answer costs nothing. Length survives only as the
    # fallback for subjects with no proper noun at all ("water molecules"),
    # where the old proxy remains the best available.
    #
    # WHY THIS MATTERS BEYOND TIDINESS: GENUINE means "the right evidence was
    # retrieved and is authentically silent", which is terminal — no better
    # query will help. RETRIEVAL_FAILURE means "try again". Getting them the
    # wrong way round tells the operator to stop looking at exactly the moment
    # retrieval is what needs fixing.
    subject_doc = ner.NLP(subject)
    proper_nouns = [t.text for t in subject_doc if t.pos_ == "PROPN"]

    if proper_nouns:
        head = max(proper_nouns, key=len).lower()
        basis = "proper noun"
    else:
        head = max(subject.split(), key=len).lower()
        basis = "longest token (no proper noun in subject)"

    haystack = " ".join(chunks).lower()

    # BOTH outcomes are logged. Only RETRIEVAL_FAILURE used to print, so a
    # GENUINE diagnosis reached the reader through run()'s summary suffix — a
    # different function than the one that made the decision. That made the
    # diagnostic look disconnected from the pipeline when it was merely silent,
    # and it hid which token the decision rested on, which is precisely what was
    # wrong above.
    if head in haystack:
        print(
            f"[Diagnosis] Subject '{subject}' matched on '{head}' ({basis}) in "
            f"{len(chunks)} evidence chunk(s) — the evidence IS about this claim, "
            f"so NEI is a considered verdict."
        )
        return "GENUINE"

    print(
        f"[Diagnosis] Subject '{subject}' — '{head}' ({basis}) appears in 0 of "
        f"{len(chunks)} evidence chunk(s). The evidence is not about this claim. "
        f"NEI is a retrieval failure, not a verdict."
    )
    return "RETRIEVAL_FAILURE"


def verify_claim(fact: str) -> FactResult:
    """
    Evidence-grounded verification of one atomic fact.

    Called from inside the extraction feedback loop. It replaced
    lightweight_verify(), which judged facts with phi3's general knowledge.
    That was the pipeline's main hallucination channel: world-knowledge
    rationales (invented company names, people, dates) were fed back into the
    extraction history, and rule 4 of the prompt explicitly told the extractor
    to recycle entities from those rationales into new "facts" that never
    appeared in the input text.

    This uses the Wikipedia-grounded retriever + verifier (AFEV's intended
    design), so rationales can only contain entities present in retrieved
    evidence. Each fact is verified exactly once — main.py does not re-run
    verification.
    """
    t_prog = time.perf_counter()
    outcome = program.try_verify(fact)
    if outcome is not None:
        program_s = time.perf_counter() - t_prog
        print(f"[Timing] program: {program_s:.2f}s — '{fact[:60]}'")
        return make_fact_result(
            fact, outcome.verdict,
            ner_query=outcome.query,
            source_url=outcome.source_url,
            program_kind=outcome.kind,
            # A program interleaves retrieval and inference, so its time is
            # reported as one figure under verification_s.
            timings={"retrieval_s": 0.0, "verification_s": round(program_s, 2)},
        )

    t0 = time.perf_counter()
    evidence_pack = retriever.fetch(fact)
    retrieval_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    verdict = verifier.verify(fact, evidence_pack["chunks"])
    verification_s = time.perf_counter() - t1

    print(
        f"[Timing] retrieval: {retrieval_s:.2f}s | "
        f"verification: {verification_s:.2f}s — '{fact[:60]}'"
    )

    # Only meaningful for NEI; None for SUPPORTS/REFUTES, where the verifier
    # reached a verdict and the quality of retrieval is not in question.
    nei_kind = (
        diagnose_nei(fact, evidence_pack["chunks"])
        if verdict.label.upper().startswith("NOT ENOUGH")
        else None
    )

    return make_fact_result(
        fact, verdict,
        ner_query=evidence_pack["query"],
        source_url=evidence_pack["source_url"],
        nei_kind=nei_kind,
        # extraction_s is attached by the extraction loop, which knows how
        # much extraction-LLM time this fact consumed.
        timings={
            "retrieval_s": round(retrieval_s, 2),
            "verification_s": round(verification_s, 2),
        },
    )
