#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdsllm.inference import BanglaLegalComplaintGenerator
from bdsllm.schema import IncidentFacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Bangla legal complaint from incident JSON.")
    parser.add_argument("--input", required=True, help="Incident JSON path")
    parser.add_argument("--model", help="Base or merged model path")
    parser.add_argument("--adapter", help="Optional LoRA adapter path")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    generator = BanglaLegalComplaintGenerator(model_name_or_path=args.model, adapter_path=args.adapter)
    print(generator.generate(IncidentFacts.from_mapping(payload)))


if __name__ == "__main__":
    main()
