#!/usr/bin/env bash
# End-to-end smoke test for the Bangla-FIR LLM pipeline.
#
# Runs a handful of LoRA fine-tuning steps on a small CPU-friendly base
# model against the synthetic seed dataset, then runs inference and
# evaluation against the resulting adapter. This does NOT produce a
# usable model -- its only purpose is to verify that data preparation,
# training, saving/loading, inference, and evaluation all work
# end-to-end without errors before committing to a full GPU training
# run on a real base model (Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct
# / etc.) and a larger, real dataset.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${SMOKE_TEST_MODEL:-distilgpt2}"
OUT_DIR="${SMOKE_TEST_OUT:-outputs/smoke-test}"

echo "=== [1/4] Generating synthetic seed dataset ==="
python3 scripts/generate_seed_dataset.py --out data/seed_dataset.jsonl --count 300

echo "=== [2/4] Training (LoRA, CPU, tiny step count) on ${MODEL} ==="
python3 -m src.train \
  --model-name-or-path "${MODEL}" \
  --no-4bit \
  --max-steps 5 \
  --batch-size 2 \
  --grad-accum 1 \
  --max-length 512 \
  --output-dir "${OUT_DIR}"

echo "=== [3/4] Inference smoke test ==="
python3 -m src.infer \
  --base-model "${MODEL}" \
  --adapter "${OUT_DIR}" \
  --max-new-tokens 64 \
  --text "আমার মানিব্যাগ চুরি হয়েছে উত্তরা বিকাল ৫টায়" || echo "(expected: tiny model / few steps may not produce valid JSON yet -- this only checks the code path runs)"

echo "=== [4/4] Evaluation smoke test (3 examples) ==="
python3 -m src.evaluate \
  --base-model "${MODEL}" \
  --adapter "${OUT_DIR}" \
  --val data/processed/val.jsonl \
  --limit 3

echo "=== Smoke test complete ==="
