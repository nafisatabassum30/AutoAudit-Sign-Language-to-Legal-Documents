# AutoAudit — Sign Language to Legal Documents

An AI system that converts **Bangladeshi Sign Language (BdSL)** into **FIR-ready
legal complaints**. The full system uses a two-stage pipeline:

1. **Vision stage** — MediaPipe pose estimation + a Spatial-Temporal Graph
   Neural Network (ST-GNN) recognises continuous BdSL gestures and outputs raw
   Bangla text.
2. **Language stage (this repository)** — a fine-tuned Large Language Model
   converts the raw Bangla statement into a structured, formal, FIR-ready legal
   complaint.

```
BdSL video ──▶ MediaPipe ──▶ ST-GNN ──▶ raw Bangla text ──▶ [ LLM ] ──▶ FIR complaint
                                                              ▲
                                                     this repository
```

This repository implements the **LLM (language) stage**.

---

## What's included

| Component | File | Purpose |
|-----------|------|---------|
| FIR schema | `src/autoaudit_llm/schema.py` | Pydantic models: `FIRExtraction`, `FIRComplaint` |
| Prompts | `src/autoaudit_llm/prompts.py` | Bangla system/instruction prompts (JSON output contract) |
| Synthetic data | `src/autoaudit_llm/data_generation.py` | Generates realistic Bangla `(statement → FIR)` pairs |
| Dataset I/O | `src/autoaudit_llm/dataset.py` | JSONL read/write, train/eval split, chat formatting |
| QLoRA training | `src/autoaudit_llm/train.py` | 4-bit LoRA fine-tuning (Llama-3 / any causal LM) |
| Post-processing | `src/autoaudit_llm/postprocess.py` | Parse/repair/validate model JSON → `FIRComplaint` |
| Rule-based baseline | `src/autoaudit_llm/rule_based.py` | Deterministic extractor + FIR builder (no GPU) |
| Inference | `src/autoaudit_llm/inference.py` | `FIRGenerator` (LLM with rule-based fallback) |
| REST API | `src/autoaudit_llm/api.py` | Flask endpoints to connect pipeline stages |
| CLI | `src/autoaudit_llm/cli.py` | `generate-data`, `train`, `infer`, `serve` |

> **Runs anywhere:** if a GPU / fine-tuned adapter / ML libraries are not
> available, the inference engine automatically falls back to a deterministic
> rule-based FIR builder, so the whole pipeline stays runnable and demonstrable.
> With the adapter present it uses the fine-tuned LLM.

---

## About the dataset

The user-provided dataset is **BdSL sign-language videos** — those are the input
to the *vision* stage. The **LLM stage trains on text pairs**
`(raw Bangla statement → FIR complaint)`, which are produced downstream of sign
recognition.

Until the proprietary corpus of 1,000+ collected FIR templates (BLAST / police)
is available, this repo ships a **synthetic Bangla FIR dataset generator** that
produces realistic training pairs covering theft (চুরি), mugging (ছিনতাই),
assault (মারধর/হামলা), threats (হুমকি) and fraud (প্রতারণা). Swap in the real
corpus by replacing `data/processed/train.jsonl` with the same JSONL format
(see `data/sample_fir_dataset.jsonl`).

Each record:

```json
{
  "raw_statement": "আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকেল ৫টা",
  "extraction": { "offense_type": "চুরি", "location": "উত্তরা", "...": "..." },
  "complaint":  { "offense_type": "চুরি", "complaint_body": "আমি, ...", "...": "..." }
}
```

---

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate

# Core (API + rule-based pipeline, no GPU needed)
pip install -r requirements.txt

# For QLoRA fine-tuning + LLM inference (GPU recommended)
pip install -r requirements-train.txt
# choose the torch build matching your CUDA, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### 1. Generate the training dataset

```bash
python -m autoaudit_llm.cli generate-data --num 400
# writes data/processed/train.jsonl and data/processed/eval.jsonl
```

### 2. Fine-tune with QLoRA (GPU)

Set your base model in `configs/default.yaml` (default:
`meta-llama/Meta-Llama-3-8B-Instruct`) and run:

```bash
python -m autoaudit_llm.cli train --config configs/default.yaml
# saves the LoRA adapter to outputs/adapter
```

### 3. Run inference on a single statement

```bash
python -m autoaudit_llm.cli infer --text "আমার মোবাইল ফোন ছিনতাই মিরপুর ১০ রাত ৯টা"
python -m autoaudit_llm.cli infer --json --text "..."   # structured JSON
python -m autoaudit_llm.cli infer --no-model --text "..." # force rule-based baseline
```

### 4. Serve the REST API

```bash
python -m autoaudit_llm.cli serve --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"raw_statement":"আমার স্বর্ণের চেইন ছিনতাই হয়েছে গুলশান ১ সন্ধ্যা ৭টা"}'
```

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| GET | `/health` | — | service + model status |
| POST | `/generate` | `{"raw_statement": "..."}` | FIR complaint (JSON + rendered doc) |
| POST | `/extract` | `{"raw_statement": "..."}` | extracted entities only |

---

## Configuration

All behaviour is driven by `configs/default.yaml` — base model, LoRA
hyperparameters, training args, data paths and inference settings. See inline
comments there for details.

## Testing

```bash
pip install pytest
PYTHONPATH=src python -m pytest -q
```

The suite covers the schema, prompts, config, synthetic data generation,
dataset I/O, JSON post-processing, the rule-based extractor, the inference
engine and the Flask API (30 tests, no GPU required).

## Roadmap

- [ ] Replace synthetic data with the real 1,000+ FIR template corpus.
- [ ] Fine-tune and publish the LoRA adapter for a Bangla-strong base model.
- [ ] Add automated FIR field-level evaluation metrics.
- [ ] On-device deployment (TensorFlow Lite / GGUF) for the mobile app.
