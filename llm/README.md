# LLM Stage: Bangla Text -> FIR-Ready Legal Complaint

This is the **LLM stage** of the AutoAudit pipeline:

```
BdSL video --(MediaPipe + ST-GNN)--> raw Bangla text --(THIS MODULE)--> FIR-ready legal complaint
```

Input: informal, possibly grammatically-loose Bangla text produced by the
sign-recognition stage, e.g. `"আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়"`
("my wallet stolen Uttara 5pm").

Output: a schema-validated JSON record (`src/schema.py::FIRRecord`) *and* a
formatted, submission-ready Bangla FIR document (`src/fir_template.py`).

## Why a schema in the middle?

Rather than asking the LLM to freely write a legal document (hard to
validate, easy to hallucinate/omit fields), the model is fine-tuned to
extract structured facts into a **fixed JSON schema**. Document formatting,
default penal-code-section suggestions, and Bangla legal boilerplate are
then rendered deterministically from that JSON -- so output quality/format
never depends on the LLM's writing style, only on correct field extraction.

```
Bangla text -> [LLM] -> JSON (FIRRecord) -> [deterministic renderer] -> FIR document
```

## Directory layout

```
llm/
├── src/
│   ├── schema.py           FIRRecord / Person / PropertyItem (Pydantic) + offense types
│   ├── fir_template.py     Renders a FIRRecord into the final Bangla document
│   ├── prompts.py          System prompt + chat-template builders (shared by train & inference)
│   ├── data_generation.py  Synthetic bootstrap dataset generator
│   └── dataset.py          JSONL -> HF `datasets.Dataset` loader for training
├── configs/
│   ├── training_config.yaml   Base model, QLoRA hyperparameters, paths
│   └── lora_config.yaml       LoRA adapter config (rank, target modules, ...)
├── data/
│   ├── templates/          Drop real, human-collected FIR examples here (see its README)
│   └── processed/          train/val/test.jsonl (synthetic bootstrap data, regenerable)
├── train.py                 QLoRA fine-tuning entrypoint
├── inference.py              Load base model (+ LoRA adapter) -> generate FIRRecord
├── evaluate.py                Evaluate any generate_fn against the test split
├── api/app.py                Flask API exposing POST /api/v1/generate-fir
├── scripts/                   Thin CLI wrappers (generate_dataset.sh, run_training.sh, run_api.sh)
├── requirements.txt           Full deps (training + serving, needs GPU)
└── requirements-lite.txt      Deps for data gen / schema / API-with-stub / tests (CPU only)
```

## 1. Data

Real FIR templates haven't been collected yet (see the project's overall
workflow table: "Collect 1,000+ FIR templates"). To unblock building and
testing the *entire* LLM pipeline today, `src/data_generation.py` generates
a large, diverse synthetic bootstrap dataset: informal Bangla phrasing
(mimicking short, telegraphic sign-language-derived text) paired with a
gold `FIRRecord` (offense type, entities, formal Bangla narrative, ...)
across 6 offense categories (theft, robbery, assault, fraud, missing
person, harassment).

A ready-to-use sample is committed at `data/processed/{train,val,test}.jsonl`
(960 / 120 / 120 examples). Regenerate or resize it with:

```bash
pip install -r requirements-lite.txt
python -m src.data_generation --n 1200 --out-dir data/processed
```

**As real FIR templates are collected**, drop them into
`data/templates/real_examples.jsonl` (same JSON shape, validated against
`FIRRecord`) and mix them into the splits with:

```bash
python -m src.data_generation --n 1200 --out-dir data/processed \
    --extra-jsonl data/templates/real_examples.jsonl
```

Replacing/augmenting the synthetic data with real examples (and eventually
training exclusively on real data) requires no code changes.

## 2. Fine-tuning (QLoRA)

`train.py` fine-tunes a Bangla-capable instruction model with QLoRA (4-bit
base weights + LoRA adapters via `peft`/`bitsandbytes`/`trl`). Requires a
CUDA GPU.

```bash
pip install -r requirements.txt   # torch/transformers/peft/trl/bitsandbytes
python train.py --config configs/training_config.yaml
```

Base model is configurable in `configs/training_config.yaml`
(`base_model:`). Suggested options, by available GPU memory:

| Model | Notes | Approx. GPU RAM (QLoRA) |
|---|---|---|
| `hishab/titulm-llama-3.2-3b-v2.0` | Small, continually pretrained on Bangla; default | ~10-12 GB |
| `Qwen/Qwen2.5-7B-Instruct` | Strong multilingual/Bangla instruction following | ~20-24 GB |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Alternative strong multilingual base | ~20-24 GB |

The adapter + tokenizer are saved to `output_dir` (default
`outputs/fir-llm-lora/`).

## 3. Inference

```bash
python inference.py \
    --base-model hishab/titulm-llama-3.2-3b-v2.0 \
    --adapter outputs/fir-llm-lora \
    --text "আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়"
```

Prints the parsed `FIRRecord` JSON and the rendered FIR document. Output
parsing (`src/postprocess.py`) tolerates markdown code fences, trailing
commas, and validates strictly against the schema -- returning a clear
`FIRParseError` (surfaced as HTTP 422 by the API) rather than silently
emitting malformed data.

## 4. Serving (Flask API)

```bash
FIR_LLM_BASE_MODEL=hishab/titulm-llama-3.2-3b-v2.0 \
FIR_LLM_ADAPTER_PATH=outputs/fir-llm-lora \
python api/app.py
```

```
GET  /health
POST /api/v1/generate-fir   {"text": "আমার মানিব্যাগ চুরি হয়েছে উত্তরায় বিকাল ৫টায়"}
```

Response:

```json
{
  "ok": true,
  "fir_record": { "offense_type": "চুরি", "...": "..." },
  "document_text": "প্রথম তথ্য প্রতিবেদন (FIR) - অভিযোগপত্র\n...",
  "repaired_json": false
}
```

`create_app(generate_fn=...)` uses dependency injection for the generation
backend, so the API layer is fully unit-testable (see `tests/test_api.py`)
without loading a real model, and can be pointed at alternative serving
backends (vLLM, TGI, etc.) by swapping `generate_fn`.

## 5. Evaluation

```bash
python evaluate.py --adapter outputs/fir-llm-lora --split test
```

Reports: JSON-parse success rate, per-field accuracy (offense type,
date/time/location, police station, district, accused_unknown), and an
average token-overlap F1 for the generated Bangla narrative vs. gold.

## Testing

The full non-GPU pipeline (schema, synthetic data generation, JSON
repair/parsing, document rendering, prompts, the Flask API via an injected
stub, and the evaluation harness) is covered by `pytest` and requires only
`requirements-lite.txt`:

```bash
pip install -r requirements-lite.txt
pytest -q
```
