import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm.dataset_utils import build_synthetic_complaint, load_dataset_frame, resolve_complaint_column, resolve_text_column


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw Bangla sign-language dataset metadata into a JSONL training file")
    parser.add_argument("--input_file", type=Path, default=None, help="Path to raw dataset file (CSV, Excel, JSON, or JSONL)")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to output JSONL training file")
    parser.add_argument("--sheet_name", type=str, default=None, help="Excel sheet to read if the input is XLSX")
    parser.add_argument("--transcript_field", type=str, default=None, help="Field name containing Bangla transcript or description")
    parser.add_argument("--complaint_field", type=str, default=None, help="Field name containing the desired legal complaint text")
    parser.add_argument("--instruction", type=str, default="ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন", help="Instruction template for the LLM")
    return parser.parse_args()


def build_example(row, transcript_field, complaint_field, instruction):
    transcript = str(row.get(transcript_field, "")).strip()
    if not transcript:
        return None

    complaint = None
    if complaint_field:
        complaint = str(row.get(complaint_field, "")).strip()
        if not complaint:
            complaint = None

    if complaint is None:
        category = str(row.get("Category", "") or row.get("category", "")).strip()
        complaint = build_synthetic_complaint(transcript, category or None)

    if not complaint:
        return None

    return {
        "instruction": instruction,
        "input": transcript,
        "output": complaint,
    }


def main():
    args = parse_args()
    df = load_dataset_frame(args.input_file, args.sheet_name)
    transcript_field = args.transcript_field or resolve_text_column(df)
    complaint_field = args.complaint_field or resolve_complaint_column(df)
    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as writer:
        valid = 0
        for _, row in df.iterrows():
            example = build_example(row, transcript_field, complaint_field, args.instruction)
            if example is None:
                continue
            writer.write(json.dumps(example, ensure_ascii=False) + "\n")
            valid += 1

    print(f"Wrote {valid} examples to {output_file}")


if __name__ == "__main__":
    main()
