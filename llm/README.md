# AutoAudit LLM — Bangla Sign-Language Text → FIR-Ready Legal Complaint

This is the **LLM stage** of the AutoAudit pipeline (see the top-level
[README](../README.md)). It takes the short, informal Bangla text produced by
the upstream BdSL sign-recognition ST-GNN stage (e.g. *"আমার ওয়ালেট চুরি
উত্তরা বিকেল ৫"*) and turns it into a structured, validated JSON object plus a
formatted, submission-ready Bangla FIR (First Information Report) document.

```
raw Bangla text  --->  fine-tuned LLM  --->  structured JSON (FIRComplaint)  --->  formatted FIR document
(from ST-GNN)          (LoRA/QLoRA)          (validated w/ pydantic)              (Bangla legal template)
```

## Why structured JSON instead of free-text generation?

The model is trained to emit **one JSON object** with fixed, English field
names (see `autoaudit_llm/schema.py`) whose *values* are Bangla. This keeps
the model's job focused on the hard part (extracting/composing the right
Bangla content) while making the output deterministically parseable and
validated (`autoaudit_llm/postprocess.py`), instead of hoping the model
free-writes a perfectly formatted legal document every time. The final
human-readable FIR document is rendered separately from the validated JSON
using a fixed Bangla template, so formatting is always consistent even if the
model's phrasing varies.

## Directory layout

```
llm/
├── autoaudit_llm/          # Core library
│   ├── schema.py           # FIRComplaint pydantic schema + OffenseType enum
│   ├── prompts.py          # System prompt + chat message builders
│   ├── dataset.py          # Instruction-tuning Dataset + label masking + padding collator
│   └── postprocess.py      # Parse/validate model output, render final FIR document
├── data/
│   ├── vocab.py                 # Slot vocabularies (names, locations, items, times...)
│   ├── seed_examples.jsonl      # Hand-authored, high-quality examples
│   ├── generate_dataset.py      # Synthetic data generator/augmenter (bootstrap dataset)
│   ├── prepare_dataset.py       # Merge + validate + format + train/val/test split
│   ├── raw_fir_templates/       # Drop real collected FIR templates here (see its README)
│   └── processed/                # Generated train.jsonl / val.jsonl / test.jsonl (committed starter set)
├── configs/
│   ├── lora_qlora.yaml     # Real training config (Qwen2.5-7B-Instruct by default)
│   └── smoke_test.yaml     # Tiny CPU config to verify the whole pipeline runs
├── train.py                # (Q)LoRA fine-tuning entrypoint
├── infer.py                 # Load adapter + generate a FIR from raw Bangla text
├── evaluate.py              # JSON validity / field accuracy / ROUGE-L on a held-out split
├── tests/                   # pytest unit tests (schema, data gen, masking, parsing) — no GPU/network needed
└── requirements.txt
```

## 1. The FIR schema

`autoaudit_llm/schema.py` defines `FIRComplaint`, a pydantic model with the
fields a Bangladeshi FIR needs: complainant info, incident date/time/location,
`offense_type` (a closed set of categories: চুরি, ছিনতাই, ডাকাতি, প্রতারণা,
মারধর/শারীরিক নির্যাতন, যৌন নির্যাতন, অপহরণ, নিখোঁজ/হারানো ব্যক্তি, যানবাহন চুরি,
সাইবার অপরাধ, হুমকি, যৌতুকের জন্য নির্যাতন, অন্যান্য), accused info, property
description, witnesses, and a `narrative` field containing the full formal
Bangla legal paragraph. This is the single contract every other piece of code
here depends on.

## 2. Training data

Real collected FIR/GD templates (the "Collect 1,000+ FIR templates" step in
the project workflow) aren't wired in yet, so this module ships a **bootstrap
dataset** so the full pipeline can be built, run, and iterated on today:

- `data/seed_examples.jsonl` — a handful of hand-written, natural examples.
- `data/generate_dataset.py` — a template + slot-filling synthetic generator
  covering all 13 offense categories (theft, robbery, dacoity, fraud,
  assault, sexual harassment, kidnapping, missing person, vehicle theft,
  cyber crime, threats, dowry harassment, other), producing realistic,
  *telegraphic* Bangla input text (mimicking concatenated sign-gloss output)
  paired with a fully-specified `FIRComplaint` target. It explicitly avoids
  ever putting a fact (date/time) in the target that isn't also stated in the
  raw input — otherwise the model would learn to hallucinate.
- `data/prepare_dataset.py` — merges all sources, validates every target
  against the pydantic schema (invalid rows are dropped, not silently kept),
  formats each example as a 3-turn chat (`system`/`user`/`assistant`, with the
  assistant turn being the target JSON), deduplicates, and splits into
  `train.jsonl` / `val.jsonl` / `test.jsonl` under `data/processed/`.

Regenerate/scale the dataset with:

```bash
python data/generate_dataset.py --n-per-template 40 --seed 42
python data/prepare_dataset.py
```

**Once you have real FIR templates**, drop them in `data/raw_fir_templates/`
(see its README) and write a small converter into the same
`{"raw_text": ..., "target": {...}}` shape — no other code needs to change.

## 3. Fine-tuning

`train.py` fine-tunes a causal LM with LoRA/QLoRA (via `peft` +
`bitsandbytes`) so that, given the chat prompt, it generates the target JSON.
Only the assistant's completion contributes to the loss (`autoaudit_llm/dataset.py`
masks out the system+user tokens).

```bash
pip install -r requirements.txt

# Real run (needs a GPU with bitsandbytes support):
python train.py --config configs/lora_qlora.yaml
```

`configs/lora_qlora.yaml` defaults to **`Qwen/Qwen2.5-7B-Instruct`** (ungated,
strong multilingual/Bangla support, no HF token needed) with standard QLoRA
settings (4-bit NF4, r=16, alpha=32, targeting all attention+MLP projection
matrices). To use **Llama-3** instead (as in the original architecture
diagram), just point `base_model` at e.g.
`meta-llama/Meta-Llama-3.1-8B-Instruct` (requires accepting its license and
`export HF_TOKEN=...`) — everything else (LoRA target module names, data
format, training/inference code) stays the same.

Any config field can be overridden from the CLI, e.g.:

```bash
python train.py --config configs/lora_qlora.yaml \
    --base_model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num_train_epochs 2
```

### Smoke-testing without a GPU

`configs/smoke_test.yaml` trains a tiny real model
(`HuggingFaceTB/SmolLM2-135M-Instruct`, 135M params) on a small subset of the
data on CPU, purely to verify the *code path* (tokenization, masking,
training loop, saving/loading the adapter, generation, JSON
parsing/validation, rendering). It is **not** expected to produce fluent
Bangla — SmolLM2's tokenizer is English-centric and wasn't pretrained on much
Bangla text, so the outputs will typically be malformed JSON. This is purely
a plumbing check; run it in a couple of minutes with:

```bash
python train.py --config configs/smoke_test.yaml
python infer.py --adapter outputs/smoke_test/final --text "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫"
```

Verified on this environment: training loss drops from ~1.03 → ~0.43 and
eval loss from ~0.83 → ~0.54 over 3 epochs on 8 examples, confirming the
training loop is actually learning (not just running without crashing).

For a real, usable model, run `configs/lora_qlora.yaml` (or a Llama-3
variant) on a GPU with the full dataset.

## 4. Inference

```bash
python infer.py --adapter outputs/fir-qlora-adapter/final \
    --text "আমার ওয়ালেট চুরি উত্তরা বিকেল ৫"
```

This loads the base model + LoRA adapter, generates the JSON completion,
parses/validates it into a `FIRComplaint` (`autoaudit_llm/postprocess.py`),
and prints the final rendered FIR document. Add `--load-in-4bit` to serve
with 4-bit quantization on a GPU, or `--merge-and-save <dir>` to bake the
LoRA weights into the base model once for simpler/faster serving (e.g. behind
the Flask API that will connect this stage to the rest of AutoAudit).

## 5. Evaluation

```bash
python evaluate.py --adapter outputs/fir-qlora-adapter/final \
    --test-file data/processed/test.jsonl
```

Reports: JSON validity rate, per-field exact-match accuracy on the
structured fields, mean field accuracy, and ROUGE-L on the `narrative` field.

## 6. Tests

```bash
pip install -r requirements.txt   # or: pip install pytest pydantic pyyaml
python -m pytest tests/ -v
```

All 22 tests are self-contained (no GPU, no network, no downloaded models —
a `DummyTokenizer` is used to test the label-masking logic) and run in under
a second.

## Integration notes for the rest of AutoAudit

- **Upstream (ST-GNN)**: this module expects a single Bangla string per
  incident report as input (`raw_text`). Whatever text the sign-recognition
  stage produces (a sequence of recognized glosses joined into a sentence)
  can be passed directly to `infer.generate_fir_json`.
- **Downstream (Flask API / mobile app)**: `infer.load_pipeline` +
  `infer.generate_fir_json` + `postprocess.try_parse_fir_json` +
  `postprocess.render_fir_document` is the complete call sequence to expose
  as an API endpoint (raw text in, FIR document + structured JSON out).
