#!/usr/bin/env python3
"""Evaluate the fine-tuned FIR-generation LLM on a held-out JSONL split.

Reports:
  - json_validity_rate: fraction of generations that parsed as JSON directly
    (before default-filling repairs).
  - field-level exact-match accuracy for each scalar FIR field.
  - list-field (penal_code_sections / items_involved) set-overlap F1.

Example:
    python -m src.evaluate --base-model Qwen/Qwen2.5-0.5B-Instruct \\
        --adapter checkpoints/fir-lora/final --data data/processed/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fir_parser import FIRParseError, parse_and_validate, parse_llm_output  # noqa: E402
from src.infer import FIRGenerator  # noqa: E402
from src.prompts import REQUIRED_FIR_KEYS  # noqa: E402

SCALAR_FIELDS = [k for k in REQUIRED_FIR_KEYS if k not in ("penal_code_sections", "items_involved")]
LIST_FIELDS = ["penal_code_sections", "items_involved"]


def set_f1(pred: list[str], gold: list[str]) -> float:
    pred_set, gold_set = set(pred), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(generator: FIRGenerator, examples: list[dict]) -> dict:
    n = len(examples)
    valid_json_count = 0
    field_correct = {f: 0 for f in SCALAR_FIELDS}
    list_f1_sums = {f: 0.0 for f in LIST_FIELDS}

    for ex in examples:
        gold = ex["fir"]
        raw_output = generator.generate_raw(ex["raw_signed_text"])
        try:
            parse_llm_output(raw_output)
            valid_json_count += 1
        except (FIRParseError, json.JSONDecodeError):
            pass

        pred = parse_and_validate(raw_output)
        for f in SCALAR_FIELDS:
            if str(pred.get(f, "")).strip() == str(gold.get(f, "")).strip():
                field_correct[f] += 1
        for f in LIST_FIELDS:
            list_f1_sums[f] += set_f1(pred.get(f, []), gold.get(f, []))

    return {
        "n_examples": n,
        "json_validity_rate": valid_json_count / n if n else 0.0,
        "field_accuracy": {f: field_correct[f] / n if n else 0.0 for f in SCALAR_FIELDS},
        "list_field_f1": {f: list_f1_sums[f] / n if n else 0.0 for f in LIST_FIELDS},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data", required=True, help="Path to a JSONL file with raw_signed_text + fir fields")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate on the first N examples")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.data, encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        examples = examples[: args.limit]

    generator = FIRGenerator(args.base_model, args.adapter, max_new_tokens=args.max_new_tokens)
    metrics = evaluate(generator, examples)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
