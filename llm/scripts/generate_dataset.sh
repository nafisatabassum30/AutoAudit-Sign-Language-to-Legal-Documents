#!/usr/bin/env bash
# Regenerate the synthetic bootstrap dataset used to train/validate/test the
# FIR-generation LLM. Real, human-collected FIR examples (JSONL, same shape
# as data/processed/*.jsonl) can be mixed in via --extra-jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m src.data_generation --n "${1:-1200}" --out-dir data/processed "${@:2}"
