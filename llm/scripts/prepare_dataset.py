"""Normalize raw Bangla legal pairs into training JSONL format."""

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Bangla legal complaint training data.")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Raw JSONL path.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Clean JSONL path.")
    return parser.parse_args()


def normalize_record(obj: dict[str, Any]) -> dict[str, Any]:
    sign_text = obj.get("sign_text_bn") or obj.get("sign_text") or obj.get("input")
    target = obj.get("target") or obj.get("output")
    metadata = obj.get("metadata", {})

    if not sign_text or not target:
        raise ValueError("Each row needs sign_text_bn/input and target/output.")

    required_target_keys = {
        "incident_date",
        "incident_time",
        "location",
        "offense_type",
        "complainant_name",
        "accused_name",
        "summary_bn",
        "full_complaint_bn",
        "requested_action_bn",
    }
    missing = required_target_keys - set(target.keys())
    if missing:
        raise ValueError(f"Missing target keys: {sorted(missing)}")

    return {
        "sign_text_bn": str(sign_text).strip(),
        "metadata": metadata,
        "target": {k: str(v).strip() for k, v in target.items()},
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            normalized = normalize_record(obj)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
