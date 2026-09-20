# Extractor Fine-tune — Workflow

Goal: replace phi3:mini's embellishment-prone extraction with a LoRA-tuned
version trained on faithful atomic decompositions, in the exact prompt format
the pipeline uses.

**Important:** the AutoCitation project itself never goes to Colab. Only the
dataset (2 JSONL files) goes up; only one GGUF model file comes back.

## 1. Build the dataset (locally, overnight)

```
cd Backend
python finetune/build_dataset.py --num-paragraphs 400
```

- Uses your local Ollama (qwen3:8b) as the teacher — make sure `ollama serve` is running.
- Output: `finetune/data/train.jsonl` and `val.jsonl`.
- Re-running appends more examples. Aim for 200+ kept paragraphs
  (≈ 800–1500 training examples) before a first training run.
- When you switch to non-Wikipedia sources: put paragraph `.txt` files in a
  folder and run with `--source-dir path/ --num-paragraphs 0` so the model
  trains on your real domain.

## 2. Upload dataset to Google Drive

Create a folder named exactly `AutoCitation_finetune` in My Drive and upload
`train.jsonl` + `val.jsonl` into it.

## 3. Train in Colab

1. Go to https://colab.research.google.com → Upload → select
   `finetune/autocitation_finetune_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. Approve the Drive access prompt.
4. Total time ≈ 1–2 h. The final cell writes
   `autocitation-extractor.q4_k_m.gguf` back to the Drive folder.

## 4. Download the GGUF

From Drive, download `autocitation-extractor.q4_k_m.gguf` into this folder
(`Backend/finetune/`).

## 5. Load into Ollama

Create a file named `Modelfile` in this folder:

```
FROM ./autocitation-extractor.q4_k_m.gguf
PARAMETER temperature 0.2
```

Then:

```
ollama create autocitation-extractor -f Modelfile
ollama run autocitation-extractor "test"
```

## 6. Point the pipeline at it

In `Backend/config.py`:

```python
FAST_MODEL: str = "autocitation-extractor"
```

Restart uvicorn. No other code changes needed.

## 7. Evaluate (do not skip)

Run the same test sentences before and after the swap (Eiffel sentence,
"Istanbul is a city", the Marie Curie mixed-truth sentence) and compare:
number of claims, embellished details, malformed iterations, terminate
behavior. For thesis numbers, run a FEVER dev sample through the full
pipeline with both extractor models.
