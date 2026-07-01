# Bangla LLM Training for Legal Complaint Generation

This folder contains the main code for fine-tuning a Bangla-language LLM to generate legal complaints from Bangla input text.

## Contents

- `train_llm.py`: fine-tunes a base model with QLoRA.
- `infer_llm.py`: runs generation using the fine-tuned model.
- `sign_to_complaint.py`: converts a recognized Bangla sign transcript into a legal complaint.
- `prepare_bangla_dataset.py`: converts raw CSV/JSON dataset metadata into JSONL training examples.
- `convert_sentence_corpus.py`: converts a Bangla sentence corpus from Excel into JSONL training data.
- `inspect_dataset.py`: inspects raw dataset files and optionally converts them to JSONL.

## Quick start

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Inspect the raw dataset and convert it to JSONL:
   ```bash
   python llm/inspect_dataset.py \
     --input_file data/raw/FinalSheet2.xlsx \
     --output_file data/train/sentence_corpus.jsonl \
     --text_field Names \
     --output_format prompt_completion
   ```

3. Convert a sentence corpus or prepare your legal dataset:
   - Sentence corpus from Excel:
     ```bash
     python llm/convert_sentence_corpus.py \
       --input_file data/raw/FinalSheet2.xlsx \
       --output_file data/train/sentence_corpus.jsonl \
       --text_field Names \
       --instruction "Bangla বাক্য পূরণ করুন" \
       --example_type prompt_completion
     ```
   - Legal complaint dataset (if you have labeled complaint pairs):
     ```bash
     python llm/prepare_bangla_dataset.py \
       --input_file data/raw/sign_transcripts.csv \
       --output_file data/train/bangla_legal_train.jsonl \
       --transcript_field transcript \
       --complaint_field complaint
     ```

4. Fine-tune the Bangla model:
   ```bash
   python llm/train_llm.py \
     --train_file data/train/sentence_corpus.jsonl \
     --output_dir output/llm \
     --model_name_or_path sagorsarker/bangla-llama-3b \
     --epochs 4 \
     --batch_size 4 \
     --dataset_format prompt_completion
   ```

4. Generate a complaint from a prompt:
   ```bash
   python llm/infer_llm.py \
     --model_dir output/llm \
     --prompt "সারার ব্যাগ চুরি হয়েছে, ছিনতাই হয়েছে" \
     --max_length 512
   ```

5. Generate a complaint from a recognized sign transcript:
   ```bash
   python llm/sign_to_complaint.py \
     --model_dir output/llm \
     --transcript "সারার ব্যাগ ছিনিয়ে নেওয়া হয়েছে, কোনো মামলা হয়েছে কিনা জানতে চাই" \
     --additional_context "স্থান: ঢাকা, সময়: গতকাল বিকাল ৫ টা"
   ```

## Notes

- Use a Bangla-compatible base model if available. The default is `sagorsarker/bangla-llama-3b`.
- The training data must be a JSONL file with `instruction`, `input`, and `output` fields.
- If the dataset is small, use `--max_train_samples` for faster experiments.

## Recommended dataset format

Each JSONL example should look like:

```json
{"instruction": "ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন", "input": "আমার ব্যাগ ছিনিয়ে নেওয়া হয়েছে।", "output": "দায়েরকারী অভিযোগ করেন যে তা ..."}
```
