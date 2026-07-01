#!/usr/bin/env python3
"""Convert real, collected (raw sign-text, FIR) pairs into the training JSONL
schema used by ``src/dataset.py``.

Use this once the team has collected the 1,000+ real FIR templates (per the
project roadmap, sourced from BLAST / police records) and paired them with
either (a) real ST-GNN sign-recognition outputs, or (b) manually written
raw/informal Bangla text approximating what the sign-recognition stage would
produce for that incident.

Input CSV format (UTF-8, header row required):
    raw_signed_text,thana,district,date_of_occurrence,time_of_occurrence,
    date_of_report,place_of_occurrence,complainant_name,complainant_address,
    complainant_phone,victim_name,accused_name,offense_type,
    penal_code_sections,items_involved,narrative_bn

``penal_code_sections`` and ``items_involved`` should be semicolon-separated
within their CSV cell (e.g. "দণ্ডবিধি ১৮৬০ - ধারা ৩৭৯;দণ্ডবিধি ১৮৬০ - ধারা ৩৮০").

Usage:
    python prepare_dataset.py --csv real_fir_data.csv --out-dir processed_real
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_synthetic_data import write_jsonl  # noqa: E402

FIR_FIELDS = [
    "thana",
    "district",
    "date_of_occurrence",
    "time_of_occurrence",
    "date_of_report",
    "place_of_occurrence",
    "complainant_name",
    "complainant_address",
    "complainant_phone",
    "victim_name",
    "accused_name",
    "offense_type",
    "narrative_bn",
]
LIST_FIELDS = ["penal_code_sections", "items_involved"]


def row_to_record(idx: int, row: dict) -> dict:
    fir = {field: (row.get(field) or "").strip() for field in FIR_FIELDS}
    for field in LIST_FIELDS:
        raw = (row.get(field) or "").strip()
        fir[field] = [v.strip() for v in raw.split(";") if v.strip()]

    return {
        "id": f"REAL-{idx:06d}",
        "raw_signed_text": (row.get("raw_signed_text") or "").strip(),
        "offense_type_key": "real",
        "fir": fir,
        "target_json": json.dumps(fir, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "processed_real")
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [row_to_record(i, row) for i, row in enumerate(reader, start=1)]

    if not records:
        raise SystemExit(f"No rows found in {args.csv}")

    n_train = int(len(records) * args.train_frac)
    n_val = int(len(records) * args.val_frac)

    write_jsonl(records[:n_train], args.out_dir / "train.jsonl")
    write_jsonl(records[n_train : n_train + n_val], args.out_dir / "val.jsonl")
    write_jsonl(records[n_train + n_val :], args.out_dir / "test.jsonl")

    print(f"Wrote {len(records)} records ({args.csv}) to {args.out_dir}")


if __name__ == "__main__":
    main()
