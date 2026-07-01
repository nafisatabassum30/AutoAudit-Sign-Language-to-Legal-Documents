# -*- coding: utf-8 -*-
"""Evaluate a fine-tuned FIR-generation adapter on a held-out JSONL split.

Reports:
  - JSON validity rate (did the model even produce a schema-valid object?)
  - Field-level exact-match accuracy for the structured (non-narrative) fields
  - ROUGE-L between the generated and reference ``narrative`` text

Usage:
    python evaluate.py --adapter outputs/fir-qlora-adapter/final \\
        --test-file data/processed/test.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

from autoaudit_llm.postprocess import try_parse_fir_json
from autoaudit_llm.schema import FIRComplaint
from infer import generate_fir_json, load_pipeline

FIELD_KEYS = [k for k in FIRComplaint.model_fields.keys() if k != "narrative"]


def rouge_l(reference: str, hypothesis: str) -> float:
    """Minimal, dependency-free ROUGE-L (LCS-based F1) over whitespace tokens."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def evaluate(model, tokenizer, rows: List[Dict], max_new_tokens: int = 512) -> Dict:
    n_valid_json = 0
    field_correct = defaultdict(int)
    field_total = defaultdict(int)
    rouge_scores = []

    for row in rows:
        reference = FIRComplaint.model_validate_json(row["target_json"])
        raw_output = generate_fir_json(model, tokenizer, row["raw_text"], max_new_tokens)
        prediction, error = try_parse_fir_json(raw_output)

        if prediction is None:
            for key in FIELD_KEYS:
                field_total[key] += 1
            rouge_scores.append(0.0)
            continue

        n_valid_json += 1
        for key in FIELD_KEYS:
            field_total[key] += 1
            if getattr(prediction, key) == getattr(reference, key):
                field_correct[key] += 1
        rouge_scores.append(rouge_l(reference.narrative, prediction.narrative))

    n = len(rows)
    per_field_accuracy = {
        key: field_correct[key] / field_total[key] if field_total[key] else 0.0 for key in FIELD_KEYS
    }
    return {
        "n_examples": n,
        "json_validity_rate": n_valid_json / n if n else 0.0,
        "per_field_accuracy": per_field_accuracy,
        "mean_field_accuracy": sum(per_field_accuracy.values()) / len(per_field_accuracy)
        if per_field_accuracy
        else 0.0,
        "mean_narrative_rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples.")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    model, tokenizer = load_pipeline(args.adapter, args.base_model, args.load_in_4bit)
    metrics = evaluate(model, tokenizer, rows, args.max_new_tokens)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
