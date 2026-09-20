# AutoCitation

Fact-checks a paragraph by decomposing it into atomic claims, retrieving
Wikipedia evidence for each one, and classifying it as **SUPPORTS**,
**REFUTES**, or **NOT ENOUGH INFO** — with the source passage and a URL
attached to every verdict.

```
input paragraph
   → sentence segmentation            (spaCy)
   → iterative atomic claim extraction (local LLM + faithfulness gates)
   → entity-anchored query building    (spaCy dependency parse + NER)
   → Wikipedia retrieval and chunk ranking
   → evidence-grounded verification    (local reasoning LLM)
   → per-claim verdict + citation
```

The design goal that shapes everything: **the extractor must preserve a false
claim rather than silently correcting it.** A checker that "fixes" *"the
Bosporus separates Africa and Europe"* into *"…Asia and Europe"* before
verifying can only ever return SUPPORTS, and can never catch the error it
exists to catch.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- ~10 GB free disk for the models
- A GPU is strongly recommended — see [Performance](#performance)

## Setup

```bash
cd Backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Start Ollama, then start the API:

```bash
ollama serve            # in one terminal
uvicorn main:app        # in another, from Backend/
```

**Models download themselves on first run.** The startup hook checks which
models are present and pulls anything missing, so no manual `ollama pull` and
no 2 GB file to fetch by hand. The first start therefore takes several minutes;
every start afterwards is instant.

To disable that (CI, or a metered connection):

```bash
AUTOCITATION_NO_AUTOPULL=1 uvicorn main:app
```

## Models

| Role | Model | Used by |
|---|---|---|
| Extraction | `phi3:mini` *(or the fine-tuned extractor)* | `Pipeline/claim_extractor.py` |
| Verification | `qwen3:8b` | `Pipeline/verifier.py` |

### The fine-tuned extractor

`autocitation-extractor` is a Phi-3-mini QLoRA fine-tune, distilled from
`qwen3:8b` as teacher, trained to decompose text into atomic claims **without
correcting them**. Its weights are not in this repository — GitHub rejects
files over 100 MB and the GGUF is 2.2 GB.

Two ways to get it:

**Pull the published model** (what the auto-pull does):

```bash
ollama pull mehmetba/autocitation-extractor
```

**Or rebuild it from source**, which is what makes the fine-tune verifiable
rather than merely trustworthy — everything needed is tracked here:

```
Backend/finetune/
├── autocitation_finetune_colab.ipynb   training notebook (Unsloth + LoRA)
├── build_dataset.py                    distillation dataset builder
├── data/train.jsonl                    957 examples
├── data/val.jsonl                      103 examples
└── Modelfile                           chat template + stop tokens
```

Run the notebook, download the resulting GGUF into `Backend/finetune/`, then:

```bash
cd Backend/finetune
ollama create autocitation-extractor -f Modelfile
```

> The `Modelfile` is not optional. Without its explicit `TEMPLATE`, Ollama
> guesses the chat format from the GGUF metadata — it picks *zephyr* — and
> wraps every prompt in a format the fine-tune never saw during training,
> producing generic off-task replies regardless of how good the model is.

## Usage

```bash
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{"text": "The Bosporus is located between Africa and Europe."}'
```

```json
{
  "claims": [{
    "claim": "Bosporus is located between Africa and Europe.",
    "label": "REFUTES",
    "rationale": "The evidence states the Bosporus forms a continental boundary between Asia and Europe.",
    "source_url": "https://en.wikipedia.org/wiki/Bosporus"
  }],
  "overall_label": "REFUTES"
}
```

A frontend is served from `Frontend/index.html`.

## Tests

```bash
cd Backend
pip install -r requirements-dev.txt
pytest
```

99 unit tests, no network and no model required — they run in seconds. Every
case documents the bug it prevents.

`regression.py` is the end-to-end suite. It needs Ollama and takes minutes:

```bash
python regression.py                    # run and record a baseline
python regression.py --sweep-thinking   # compare verifier reasoning budgets
```

## Evaluation

`pytest` says the code is correct. `evaluation.py` says how *good* it is —
scored numbers, on [FEVER](https://fever.ai/dataset/fever.html), against gold
labels.

**Running and scoring are separate commands.** A run costs ~150 seconds of CPU
inference per claim, so a metric you can only change by re-running an hour of
inference is a metric nobody improves. `run` caches raw output to
`evaluation_results.json`; `score` reads that file and computes everything in
milliseconds.

### Quick check — the built-in cases

```bash
cd Backend
python -m evaluation list          # 6 cases, 13 gold propositions
python -m evaluation run           # ~45-90 min
python -m evaluation score
```

These six paragraphs were used to *develop* the pipeline, so their numbers are
overfit by construction. Treat them as a regression check, never as a result.

### Real numbers — FEVER

```bash
# 1. Download the labelled dev split (no registration, ~40 MB)
curl -O https://fever.ai/download/fever/shared_task_dev.jsonl

# 2. Sample it. Start small: 5 per label runs in ~35 minutes.
python -m benchmarks.fever_sample --input shared_task_dev.jsonl --per-label 5

# 3. Run, then score
python -m evaluation run   --gold fever_gold.json
python -m evaluation score --gold fever_gold.json --markdown ../docs/evaluation.md
```

Raise `--per-label` to 30 for a reportable run — 90 claims, roughly 3.75 hours.
FEVER's labels are the exact strings this pipeline emits, so nothing is mapped
or lost.

`--gold` must match on both commands. Scoring a FEVER run against the built-in
cases silently reports zero of everything.

### Reading the report

**Extraction first**, as TP / FN / FP. Recall bounds every number below it —
a claim that is never extracted can never be labelled, so it cannot appear in
any label metric. This is the number that catches silently dropped sentences.

**A 3×3 confusion matrix**, and everything else is a reading of it: per-class
precision / recall / F1, macro-F1 (comparable to FEVER and AFEV reporting), and
accuracy.

Read the **REFUTES** row first. Its *precision* is the safety number — a false
REFUTES tells a user their true sentence is false and hands them a citation for
it. Its *recall* is the usefulness number. A low SUPPORTS recall is the benign
failure: unhelpful rather than harmful.

**Then the selective-prediction pair**, because this system may decline to
answer and NOT ENOUGH INFO is a refusal, not a wrong answer:

| | |
|---|---|
| **Answered accuracy** | of the decidable claims it chose to answer, how many were right |
| **Abstention rate** | of the decidable claims, how many it declined |

Neither means anything alone. A system answering one claim in twenty at 100%
is useless; one answering all of them at 60% is worse than useless, because a
wrong verdict arrives with a citation attached.

**Finally the abstention breakdown** — `RETRIEVAL_FAILURE` (recoverable, the
evidence was not about the claim), `GENUINE` (terminal), `EXTRACTION_FAILURE`
(no claim was produced at all). This turns an abstention rate into something
actionable: mostly-recoverable abstentions mean the ceiling is far above the
current score.

> **What FEVER does and does not measure here.** Its claims are already single
> atomic sentences, so the extractor mostly passes them through — a FEVER
> number is a *verifier and retriever* score, not an end-to-end one. And its
> evidence was annotated against a June 2017 Wikipedia dump while this pipeline
> queries live Wikipedia, so some disagreement with gold is the corpus moving
> rather than the system failing.

## Performance

Verification dominates: roughly 65–75% of wall-clock time. On CPU an 8B model
produces ~2 tokens/sec, which puts a single verification in the 2–4 minute
range. Check what you are actually running on:

```bash
ollama ps       # the PROCESSOR column reads 100% GPU, 100% CPU, or a split
```

Tuning knobs, in order of impact:

| Setting | Where | Effect |
|---|---|---|
| GPU offload | `ollama ps` | 10–30× — dwarfs everything else |
| `VERIFIER_THINKING` | `Pipeline/verifier.py` | `False` / `"low"` / `True` |
| `TOP_K_CHUNKS` | `Pipeline/retriever.py` | fewer chunks = shorter prompt |
| `MAX_ITERATIONS_PER_SENTENCE` | `Pipeline/claim_extractor.py` | caps extraction calls |

`VERIFIER_THINKING` is deliberately a knob to measure rather than a settled
choice: qwen3's reasoning tokens are discarded after generation, so disabling
them is nearly free speed — unless the reasoning is what produces the correct
verdict. `regression.py --sweep-thinking` answers that empirically.

## Layout

```
Backend/
├── main.py                 FastAPI app, startup provisioning
├── config.py               URLs, model names, input limits
├── regression.py           end-to-end behavioural suite
├── Pipeline/
│   ├── claim_extractor.py  per-sentence decomposition + gates
│   ├── ner.py              spaCy: segmentation, subject recovery, quality gates
│   ├── retriever.py        Wikipedia fetch, chunking, ranking
│   ├── verifier.py         evidence-grounded classification
│   └── postprocessor.py    response assembly, citation URLs
├── Tests/                  unit tests (pytest)
├── models/ollama_client.py Ollama transport + model provisioning
└── finetune/               training data, notebook, Modelfile
Frontend/index.html
Documents/                  design plan and reference papers
```

## Known limitations

- **Unsupported ≠ refuted.** The system refutes claims the evidence
  *contradicts*. A fabricated private conversation produces NOT ENOUGH INFO,
  because Wikipedia contains no sentence denying it.
- **Retrieval is the main failure mode.** When a claim carries no named
  entities, query construction has nothing to anchor on and NEI results from a
  failed lookup rather than a considered judgement. `nei_kind` distinguishes
  the two in the output.
- **English only.** `en_core_web_sm` and the English Wikipedia are assumed
  throughout.
