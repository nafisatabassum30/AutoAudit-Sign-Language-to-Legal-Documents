import argparse
from pathlib import Path
import pandas as pd
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw Bangla sign-language dataset metadata into a JSONL training file")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to raw dataset file (CSV or JSON)")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to output JSONL training file")
    parser.add_argument("--transcript_field", type=str, default="transcript", help="Field name containing Bangla transcript or description")
    parser.add_argument("--complaint_field", type=str, default="complaint", help="Field name containing the desired legal complaint text")
    parser.add_argument("--instruction", type=str, default="ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন", help="Instruction template for the LLM")
    return parser.parse_args()


def load_raw_data(input_file: Path):
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_file)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(input_file, lines=suffix == ".jsonl")
    raise ValueError("Unsupported input file format. Use CSV or JSON/JSONL.")


def build_example(row, transcript_field, complaint_field, instruction):
    transcript = str(row.get(transcript_field, "")).strip()
    complaint = str(row.get(complaint_field, "")).strip()
    if not transcript or not complaint:
        return None
    return {
        "instruction": instruction,
        "input": transcript,
        "output": complaint,
    }


def main():
    args = parse_args()
    df = load_raw_data(args.input_file)
    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as writer:
        valid = 0
        for _, row in df.iterrows():
            example = build_example(row, args.transcript_field, args.complaint_field, args.instruction)
            if example is None:
                continue
            writer.write(json.dumps(example, ensure_ascii=False) + "\n")
            valid += 1

    print(f"Wrote {valid} examples to {output_file}")


if __name__ == "__main__":
    main()
