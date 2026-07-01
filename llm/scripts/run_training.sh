#!/usr/bin/env bash
# Convenience wrapper for a full QLoRA fine-tuning run.
#
# Usage:
#   ./scripts/run_training.sh
#   ./scripts/run_training.sh --base-model meta-llama/Meta-Llama-3.1-8B-Instruct
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f "data/processed/train.jsonl" ]; then
  echo "[run_training] No dataset found at data/processed/. Generating the synthetic starter dataset..."
  python data/generate_synthetic_data.py --n 800 --seed 42 --out-dir data/processed
fi

python -m src.train --config config/training_config.yaml "$@"
