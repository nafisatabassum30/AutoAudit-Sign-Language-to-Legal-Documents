"""Evaluate the fine-tuned Bangla-FIR LLM against a held-out validation
split.

Reports:
  - Per-field exact-match accuracy for the structured fields
    (date, time, location, offense_type, complainant_*, accused).
  - ROUGE-L on the free-text ``description`` field.
  - JSON parse success rate (how often the model emits valid,
    schema-conforming JSON at all).

Usage:
    python -m src.evaluate --adapter outputs/bangla-fir-lora \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --val data/processed/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    from .fir_schema import FIR_FIELDS, FIRParseError, FIRRecord
    from .infer import FIRGenerator
except ImportError:  # pragma: no cover
    from fir_schema import FIR_FIELDS, FIRParseError, FIRRecord
    from infer import FIRGenerator

DESCRIPTION_FIELD = "description"
STRUCTURED_FIELDS = [f for f in FIR_FIELDS if f != DESCRIPTION_FIELD]


def _load_val_set(path: Path) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _rouge_l(pred: str, ref: str) -> float:
    """Lightweight ROUGE-L (LCS-based F1) so the script only needs
    ``rouge_score`` if available, otherwise falls back to a pure-python
    implementation (character-level, appropriate for Bangla which
    doesn't whitespace-tokenize as cleanly as English)."""

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return scorer.score(ref, pred)["rougeL"].fmeasure
    except ImportError:
        return _lcs_f1(pred, ref)


def _lcs_f1(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def evaluate(generator: FIRGenerator, val_records: List[Dict]) -> Dict:
    field_hits = {f: 0 for f in STRUCTURED_FIELDS}
    rouge_scores: List[float] = []
    parse_failures = 0
    per_example = []

    for rec in val_records:
        input_text = rec["input_text"]
        gold = FIRRecord.from_dict(rec["fir"])

        try:
            pred, _ = generator.generate(input_text)
        except FIRParseError as exc:
            parse_failures += 1
            per_example.append({"input_text": input_text, "error": str(exc)})
            continue

        example_result = {"input_text": input_text, "fields": {}}
        for f in STRUCTURED_FIELDS:
            match = getattr(pred, f).strip() == getattr(gold, f).strip()
            field_hits[f] += int(match)
            example_result["fields"][f] = match

        r = _rouge_l(pred.description, gold.description)
        rouge_scores.append(r)
        example_result["description_rouge_l"] = r
        per_example.append(example_result)

    n = len(val_records)
    n_parsed = n - parse_failures
    field_accuracy = {f: (hits / n_parsed if n_parsed else 0.0) for f, hits in field_hits.items()}
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

    return {
        "n_examples": n,
        "n_parse_failures": parse_failures,
        "parse_success_rate": n_parsed / n if n else 0.0,
        "field_accuracy": field_accuracy,
        "mean_field_accuracy": sum(field_accuracy.values()) / len(field_accuracy) if field_accuracy else 0.0,
        "description_rouge_l": avg_rouge,
        "per_example": per_example,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to dump full JSON results.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples.")
    args = parser.parse_args()

    val_records = _load_val_set(args.val)
    if args.limit:
        val_records = val_records[: args.limit]

    generator = FIRGenerator(base_model=args.base_model, adapter_path=args.adapter)
    results = evaluate(generator, val_records)

    summary = {k: v for k, v in results.items() if k != "per_example"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Full results written to {args.out}")


if __name__ == "__main__":
    _cli()
