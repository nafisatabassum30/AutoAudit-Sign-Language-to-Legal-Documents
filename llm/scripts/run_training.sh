#!/usr/bin/env bash
# Launch QLoRA fine-tuning. Requires a CUDA GPU (see llm/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 train.py --config configs/training_config.yaml "$@"
