# LLM Stage — BdSL Raw Text → FIR-Ready Legal Complaint

This module implements the **LLM stage** of the AutoAudit pipeline: it takes
the short, informal Bangla text produced by the upstream ST-GNN sign-language
recognition stage (e.g. `আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা`) and turns it
into a complete, structured, FIR-ready legal complaint in Bangla.

```
ST-GNN output (raw Bangla text)
        │
        ▼
 [ this module ]  fine-tuned LLM (QLoRA)  →  structured FIR JSON  →  formatted FIR document
        │
        ▼
Flask API (api/app.py)  →  downstream apps (mobile app, police review UI, ...)
```

## Why structured JSON as the model's output?

Rather than asking the LLM to directly write free-form legal prose (which is
hard to validate and easy to hallucinate in), the model is trained to emit a
**single JSON object** with a fixed schema (thana, date/time, complainant,
victim, accused, offense type, penal-code sections, items involved, and a
narrative paragraph). A deterministic, non-LLM renderer
(`src/fir_parser.py::render_document`) then turns that validated JSON into the
final formatted Bangla FIR document. This keeps the legally-sensitive
formatting deterministic and makes the LLM's job (structured information
extraction + narrative drafting) easier to fine-tune, evaluate, and repair.

## Directory layout

```
llm/
├── config/
│   └── training_config.yaml   # base model, LoRA, quantization, training hyperparameters
├── data/
│   ├── value_banks.py          # names/locations/time & date phrase pools (Bangla)
│   ├── offense_catalog.py      # per-offense raw-text + formal-narrative templates
│   ├── generate_synthetic_data.py  # builds the starter train/val/test JSONL dataset
│   ├── prepare_dataset.py      # converts real collected FIR CSVs into the same schema
│   └── processed/              # generated starter dataset (train/val/test .jsonl)
├── src/
│   ├── prompts.py               # system/user prompt construction, required FIR schema
│   ├── dataset.py               # tokenization + prompt/response loss masking
│   ├── train.py                 # QLoRA/LoRA fine-tuning entry point
│   ├── infer.py                 # load base model + adapter, generate FIR JSON + document
│   ├── evaluate.py               # field-level accuracy / JSON-validity metrics
│   └── fir_parser.py            # robust JSON extraction, validation, document rendering
├── api/
│   └── app.py                   # Flask API bridging this stage to the rest of the system
├── scripts/
│   ├── run_training.sh
│   └── download_base_model.sh
└── tests/                        # unit tests (no GPU / model download required)
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

`bitsandbytes` (used for 4-bit QLoRA) only installs/works on Linux+CUDA. On
CPU-only machines, pass `--no-4bit` to `src/train.py` (full-precision LoRA;
fine for smoke tests, slow for real training).

## 2. Dataset

### Starter synthetic dataset (already generated, committed under `data/processed/`)

Since we don't yet have the ~1,000+ real FIR templates mentioned in the
project roadmap (to be collected from BLAST / police sources), this module
ships with an **offline, template-based synthetic dataset generator**
(`data/generate_synthetic_data.py`) covering 13 common offense categories
(চুরি, ছিনতাই, ডাকাতি, গৃহে চুরি, মারামারি, হুমকি, প্রতারণা, যৌন হয়রানি,
ইভটিজিং, নিখোঁজ ব্যক্তি, সম্পত্তি ভাংচুর, মাদক সংক্রান্ত, সাইবার হয়রানি) with
randomized names, Dhaka thanas, relative dates/times, and known/unknown
suspects. This lets you develop, test, and smoke-test the whole fine-tuning
pipeline immediately, without waiting on real data collection.

Regenerate/resize it with:

```bash
python data/generate_synthetic_data.py --n 800 --seed 42 --out-dir data/processed
```

**This synthetic dataset is a bootstrap, not a replacement for real data.**
It is useful for validating the pipeline and for a small amount of
data-augmentation, but a model fine-tuned only on it will not generalize well
to real, messy sign-recognition output. Prioritize collecting and adding real
`(raw_signed_text, FIR)` pairs.

### Real data

Once real FIR templates (and either real ST-GNN outputs or manually-written
approximations of them) are collected, convert them into the same schema with:

```bash
python data/prepare_dataset.py --csv real_fir_data.csv --out-dir data/processed_real
```

then point `config/training_config.yaml`'s `data.train_path` / `data.val_path`
at the combined/real files (or pass `--train-path` / `--val-path` to
`src/train.py`).

## 3. Fine-tune (QLoRA)

```bash
./scripts/run_training.sh
# or, with overrides:
python -m src.train --config config/training_config.yaml \
    --base-model meta-llama/Meta-Llama-3.1-8B-Instruct
```

The default `base_model` in `config/training_config.yaml` is
`Qwen/Qwen2.5-0.5B-Instruct` — small, ungated, and multilingual, so the
pipeline runs out of the box on modest hardware for development/testing. For
production quality, swap in a larger, stronger multilingual/Bangla-capable
base model, e.g.:

- `meta-llama/Meta-Llama-3.1-8B-Instruct` (matches the project's original
  roadmap; gated on Hugging Face — accept the license and set `HF_TOKEN`)
- `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-14B-Instruct` (open license,
  strong multilingual/Bangla performance)

Pre-fetch weights ahead of time with `scripts/download_base_model.sh <model_id>`.

On CPU or memory-constrained machines, use:

```bash
python -m src.train --config config/training_config.yaml --no-4bit \
    --per-device-batch-size 1 --gradient-accumulation-steps 1 --max-steps 5
```

## 4. Run inference

```bash
python -m src.infer \
    --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter checkpoints/fir-lora/final \
    --text "আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা"
```

Add `--json-only` to print just the structured FIR JSON instead of the
rendered document.

## 5. Evaluate

```bash
python -m src.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter checkpoints/fir-lora/final --data data/processed/test.jsonl
```

Reports JSON-validity rate, per-field exact-match accuracy, and set-F1 for the
list fields (`penal_code_sections`, `items_involved`).

## 6. Serve via API (bridge to the rest of the pipeline)

```bash
FIR_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
FIR_ADAPTER_PATH=checkpoints/fir-lora/final \
python api/app.py
```

```bash
curl -X POST http://localhost:8000/api/v1/generate-fir \
    -H "Content-Type: application/json" \
    -d '{"text": "আমার মানিব্যাগ চুরি উত্তরা বিকাল ৫টা"}'
```

Response:

```json
{
  "fir": { "thana": "উত্তরা", "offense_type": "চুরি", "...": "..." },
  "document": "বাংলাদেশ পুলিশ - প্রাথমিক অভিযোগ (FIR) ফরম\n...",
  "llm_raw_output": "{...raw model output...}"
}
```

The upstream ST-GNN stage should POST its recognized raw Bangla text to this
endpoint; the caller may optionally include `complainant_name`,
`complainant_address`, `complainant_phone` (e.g. from the logged-in user's
profile in the mobile app) to override fields the LLM cannot reliably infer
from a short signed utterance alone.

## Testing

```bash
python -m pytest tests/ -v
```

All 17 tests run fully offline (no model download / GPU required) and cover
JSON extraction/repair, default-filling, document rendering, prompt
construction, and the synthetic dataset generator. They were verified to pass
in this environment.

An end-to-end mechanical smoke test (dataset → tokenize+mask → LoRA train →
save adapter → load adapter → generate → parse → render) was also run
manually against `Qwen/Qwen2.5-0.5B-Instruct` on CPU using `--no-4bit
--per-device-batch-size 1` to confirm the full pipeline is wired correctly;
see the "Notes on this environment" section below.

## Notes on this environment / what "production-ready" still requires

- **No GPU was available in this sandbox.** All code paths (dataset
  generation, tokenization/masking, LoRA fine-tuning, adapter save/load,
  generation, JSON repair, document rendering, and the Flask API) were
  exercised successfully on CPU with a small model, but a real fine-tuning
  run (enough steps/epochs, a large enough base model, and 4-bit
  quantization on a CUDA GPU) is required to get FIR outputs of usable
  quality. `config/training_config.yaml` and `scripts/run_training.sh` are
  ready to run as-is on a GPU box.
- **The committed dataset is synthetic.** It's useful for pipeline
  development and as light data augmentation, but real collected FIR
  templates (per the project roadmap) are needed for production quality and
  legal accuracy.
- **Legal review required.** The offense categories, penal-code section
  references, and document template in this module are illustrative
  approximations to give the model realistic structure. Any FIR text
  generated by this system must be reviewed by a qualified legal
  professional (and/or the complainant) before submission to law
  enforcement — this system is a drafting aid, not a legal authority.
