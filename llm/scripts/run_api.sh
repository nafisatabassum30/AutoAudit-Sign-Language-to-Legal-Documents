#!/usr/bin/env bash
# Start the Flask API serving the fine-tuned FIR-generation model.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 api/app.py \
  --base-model "${FIR_LLM_BASE_MODEL:-hishab/titulm-llama-3.2-3b-v2.0}" \
  --adapter "${FIR_LLM_ADAPTER_PATH:-outputs/fir-llm-lora}" \
  "$@"
