# AutoAudit: Sign Language to Legal Documents

AutoAudit converts Bangladeshi Sign Language (BdSL) into structured Bangla legal
complaints. The full system has two major stages:

1. A vision/sign-recognition stage that converts videos into Bangla gloss or text.
2. A Bangla legal LLM stage that rewrites those recognized signs into an
   FIR-ready complaint draft.

This repository currently implements the **LLM part**: dataset preparation,
Bangla complaint prompt templates, QLoRA fine-tuning, inference, and an optional
FastAPI endpoint.

## What the LLM module expects

The LLM does not train directly on raw video. It expects the output of your
BdSL recognizer, for example:

```json
{
  "recognized_text": "আমার মোবাইল ফোন উত্তরায় পাঁচটার সময় চুরি হয়েছে",
  "complainant_name": "রহিমা খাতুন",
  "incident_date": "২০২৬-০৭-০১",
  "incident_time": "বিকাল ৫টা",
  "incident_location": "উত্তরা ৭ নম্বর সেক্টর",
  "offense_type": "মোবাইল ফোন চুরি"
}
```

If you already have human-written FIR/legal complaint templates, include them as
`complaint_text`. If not, the preparation script creates a safe weak-label
complaint draft that can be improved later by legal reviewers.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Large-model training requires a CUDA GPU and access to the selected base model.

## Prepare instruction data

Input can be CSV, JSON, or JSONL. Supported fields include:

- `recognized_text` or `sign_text`
- `complainant_name`
- `complainant_address`
- `complainant_phone`
- `incident_date`
- `incident_time`
- `incident_location`
- `accused_name`
- `accused_details`
- `offense_type`
- `requested_action`
- `additional_context`
- `complaint_text` or `response` for supervised targets

Example:

```bash
python scripts/prepare_llm_dataset.py \
  --input examples/sign_incidents.jsonl \
  --output data/bangla_legal_train.jsonl
```

Each output line has:

```json
{"instruction": "...", "response": "...", "metadata": {"video_id": "..."}}
```

## Fine-tune with QLoRA

Use a Bangla-capable instruction model. Good starting points are multilingual
instruct models such as Llama, Qwen, Gemma, or Mistral variants that support
Bangla text.

```bash
python -m bdsllm.train_qlora \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --train-file data/bangla_legal_train.jsonl \
  --output-dir outputs/bangla-legal-lora \
  --num-train-epochs 3
```

If you have a validation set:

```bash
python -m bdsllm.train_qlora \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --train-file data/train.jsonl \
  --eval-file data/valid.jsonl \
  --output-dir outputs/bangla-legal-lora
```

## Generate a complaint

Without a model, the command uses the deterministic FIR-style Bangla template:

```bash
python scripts/generate_complaint.py --input examples/incident_request.json
```

With a fine-tuned adapter:

```bash
python scripts/generate_complaint.py \
  --input examples/incident_request.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --adapter outputs/bangla-legal-lora
```

## Run as an API

```bash
export BANGLA_LEGAL_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export BANGLA_LEGAL_LLM_ADAPTER=outputs/bangla-legal-lora
uvicorn bdsllm.api:app --host 0.0.0.0 --port 8000
```

POST `/generate-complaint` with the incident JSON fields above.

If no model environment variables are set, the endpoint still returns a
template-based complaint. This is useful while the sign-recognition model and
LLM adapter are being developed independently.

## Test

```bash
pytest
```
