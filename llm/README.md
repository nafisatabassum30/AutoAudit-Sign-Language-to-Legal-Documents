# Bangla Legal Complaint LLM Module

This module builds the **LLM stage** of your pipeline:

1. Input: Bangla text generated from Bangladeshi Sign Language recognition.
2. Output: Structured FIR-ready legal complaint fields + natural Bangla complaint narrative.

## What is implemented

- Dataset preparation pipeline for supervised fine-tuning data.
- Prompt template designed for legal complaint generation in Bangla.
- QLoRA training script for efficient fine-tuning.
- Inference script that returns:
  - Structured JSON fields (`incident_date`, `location`, `offense_type`, etc.)
  - A final Bangla complaint text for submission.
- Basic post-processing + schema validation.

## Recommended training data format

Prepare a JSONL file where each line contains:

```json
{
  "sign_text_bn": "গতকাল রাতে বাজার থেকে ফেরার সময় দুইজন লোক আমাকে মারধর করে টাকা নিয়ে যায়",
  "metadata": {
    "source": "bdsl_video_00123.mp4"
  },
  "target": {
    "incident_date": "2026-06-30",
    "incident_time": "21:30",
    "location": "মিরপুর ১০, ঢাকা",
    "offense_type": "ছিনতাই ও শারীরিক আঘাত",
    "complainant_name": "অজানা",
    "accused_name": "অজ্ঞাত",
    "summary_bn": "দুইজন অজ্ঞাত ব্যক্তি মারধর করে টাকা ছিনিয়ে নেয়।",
    "full_complaint_bn": "আমি গত ৩০ জুন ২০২৬ রাত ৯:৩০ মিনিটে মিরপুর ১০ এলাকায় ...",
    "requested_action_bn": "ঘটনার তদন্ত করে আইনানুগ ব্যবস্থা গ্রহণের অনুরোধ করছি।"
  }
}
```

`sign_text_bn` should be generated from your sign recognition model.

## Quick start

### 1) Install dependencies

```bash
pip install -r llm/requirements.txt
```

### 2) Convert raw data into SFT-ready JSONL

```bash
python llm/scripts/prepare_dataset.py \
  --input_jsonl path/to/raw_legal_pairs.jsonl \
  --output_jsonl data/train_sft.jsonl
```

### 3) Fine-tune with QLoRA

```bash
python llm/src/train_qlora.py \
  --train_jsonl data/train_sft.jsonl \
  --base_model meta-llama/Meta-Llama-3-8B-Instruct \
  --output_dir outputs/bangla-legal-qlora
```

### 4) Run inference

```bash
python llm/src/infer.py \
  --model_path outputs/bangla-legal-qlora \
  --sign_text_bn "আজ দুপুরে আমার দোকান থেকে নগদ টাকা চুরি হয়েছে।"
```

## Notes for your BdSL project

- Keep legal templates in Bangla and include district/police-station style from real FIRs.
- Add examples of:
  - harassment
  - physical assault
  - theft/robbery
  - extortion
  - domestic violence
- Include noisy sign-to-text phrasing in training data to improve robustness.
