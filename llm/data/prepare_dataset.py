# -*- coding: utf-8 -*-
"""Merge seed + synthetic examples, validate against the FIR schema, format
as chat-style SFT examples, and split into train/val/test JSONL files.

Run:
    python prepare_dataset.py \\
        --inputs llm/data/seed_examples.jsonl llm/data/processed/synthetic_raw.jsonl \\
        --out-dir llm/data/processed
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autoaudit_llm.prompts import build_chat_messages  # noqa: E402
from autoaudit_llm.schema import FIRComplaint  # noqa: E402

from pydantic import ValidationError  # noqa: E402


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_and_format(rows: List[Dict]) -> List[Dict]:
    formatted = []
    n_invalid = 0
    for row in rows:
        try:
            complaint = FIRComplaint.model_validate(row["target"])
        except (ValidationError, KeyError) as exc:
            n_invalid += 1
            print(f"[skip] invalid example: {exc}", file=sys.stderr)
            continue
        target_json = json.dumps(
            complaint.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        messages = build_chat_messages(row["raw_text"], assistant_json=target_json)
        formatted.append({"raw_text": row["raw_text"], "target_json": target_json, "messages": messages})
    if n_invalid:
        print(f"Skipped {n_invalid} invalid example(s).", file=sys.stderr)
    return formatted


def split(rows: List[Dict], seed: int, val_frac: float, test_frac: float):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val :]
    return train, val, test


def write_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            os.path.join(here, "seed_examples.jsonl"),
            os.path.join(here, "processed", "synthetic_raw.jsonl"),
        ],
    )
    parser.add_argument("--out-dir", default=os.path.join(here, "processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()

    all_rows: List[Dict] = []
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[warn] input not found, skipping: {path}", file=sys.stderr)
            continue
        all_rows.extend(load_jsonl(path))

    if not all_rows:
        raise SystemExit(
            "No input rows found. Run generate_dataset.py first or check --inputs."
        )

    formatted = validate_and_format(all_rows)
    # Deduplicate identical raw_text -> target_json pairs (synthetic generation can repeat).
    seen = set()
    deduped = []
    for row in formatted:
        key = (row["raw_text"], row["target_json"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    train, val, test = split(deduped, args.seed, args.val_frac, args.test_frac)

    write_jsonl(train, os.path.join(args.out_dir, "train.jsonl"))
    write_jsonl(val, os.path.join(args.out_dir, "val.jsonl"))
    write_jsonl(test, os.path.join(args.out_dir, "test.jsonl"))

    print(
        f"Prepared {len(deduped)} valid examples "
        f"(train={len(train)}, val={len(val)}, test={len(test)}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
