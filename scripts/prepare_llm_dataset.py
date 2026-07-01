#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdsllm.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bangla legal LLM instruction JSONL.")
    parser.add_argument("--input", required=True, help="CSV, JSON, or JSONL sign/FIR records")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    count = prepare_dataset(args.input, args.output)
    print(f"Wrote {count} instruction examples to {args.output}")


if __name__ == "__main__":
    main()
