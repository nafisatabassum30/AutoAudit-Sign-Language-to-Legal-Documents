#!/usr/bin/env bash
# Pre-fetch base model weights from the Hugging Face Hub so training/inference
# don't need to download them on first run. For gated models (e.g. Llama 3.x)
# you must first accept the license on huggingface.co and run
# `huggingface-cli login` (or set the HF_TOKEN env var) before this will work.
#
# Usage:
#   ./scripts/download_base_model.sh Qwen/Qwen2.5-7B-Instruct
#   HF_TOKEN=hf_xxx ./scripts/download_base_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-7B-Instruct}"

python - "$MODEL_ID" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_id = sys.argv[1]
print(f"Downloading {model_id} ...")
path = snapshot_download(repo_id=model_id)
print(f"Downloaded to: {path}")
PY
