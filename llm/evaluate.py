#!/usr/bin/env python3
"""Evaluate a FIR-generation backend against the held-out test split.

Works with any ``generate_fn: str -> str`` (real fine-tuned model, base
model baseline, or a stub), so it doubles as:

* a real evaluation harness once a model is trained, and
* a regression/smoke test for the schema + parsing pipeline without any
  ML dependencies (see ``tests/test_evaluate.py``).

Usage
-----
    python evaluate.py --adapter outputs/fir-llm-lora --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from src.dataset import load_jsonl
from src.postprocess import FIRParseError, parse_fir_output
from src.schema import FIRRecord

SCALAR_FIELDS = [
    "offense_type",
    "incident_date",
    "incident_time",
    "incident_location",
    "police_station",
    "district",
    "accused_unknown",
]


def _token_f1(pred: str, gold: str) -> float:
    """Cheap, dependency-free token-overlap F1 as a narrative-quality proxy."""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    gold_counts: Dict[str, int] = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            common += 1
            gold_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(generate_fn: Callable[[str], str], test_rows: List[Dict]) -> Dict:
    n = len(test_rows)
    parse_success = 0
    field_matches = {f: 0 for f in SCALAR_FIELDS}
    field_totals = {f: 0 for f in SCALAR_FIELDS}
    narrative_f1_sum = 0.0
    per_example = []

    for row in test_rows:
        input_text = row["input_text"]
        gold = FIRRecord.model_validate(row["output_json"])

        raw = generate_fn(input_text)
        example_report = {"input_text": input_text, "raw_output": raw}

        try:
            parsed = parse_fir_output(raw, original_input_text=input_text)
            parse_success += 1
            pred = parsed.record
        except FIRParseError as e:
            example_report["error"] = str(e)
            per_example.append(example_report)
            continue

        for f in SCALAR_FIELDS:
            gold_val = getattr(gold, f)
            field_totals[f] += 1
            if getattr(pred, f) == gold_val:
                field_matches[f] += 1

        f1 = _token_f1(pred.narrative_bn, gold.narrative_bn)
        narrative_f1_sum += f1
        example_report["narrative_token_f1"] = f1
        per_example.append(example_report)

    metrics = {
        "n_examples": n,
        "json_parse_success_rate": parse_success / n if n else 0.0,
        "field_accuracy": {
            f: (field_matches[f] / field_totals[f] if field_totals[f] else None)
            for f in SCALAR_FIELDS
        },
        "avg_narrative_token_f1": narrative_f1_sum / n if n else 0.0,
    }
    return {"metrics": metrics, "per_example": per_example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--base-model", default="hishab/titulm-llama-3.2-3b-v2.0")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_jsonl(args.data_dir / f"{args.split}.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    from inference import FIRGenerator

    generator = FIRGenerator(
        base_model=args.base_model, adapter_path=args.adapter, load_in_4bit=not args.no_4bit
    )
    report = evaluate(generator.generate_raw, rows)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
