import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm.dataset_utils import build_synthetic_complaint, load_dataset_frame, resolve_complaint_column, resolve_text_column


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a Bangla dataset file and optionally export it as JSONL for LLM training.")
    parser.add_argument("--input_file", type=Path, default=None, help="Path to the raw dataset file (Excel, CSV, JSON, JSONL)")
    parser.add_argument("--sheet_name", type=str, default=None, help="Excel sheet to read if the input is XLSX")
    parser.add_argument("--text_field", type=str, default=None, help="Field containing Bangla text")
    parser.add_argument("--complaint_field", type=str, default=None, help="Field containing complaint text if available")
    parser.add_argument("--output_file", type=Path, default=None, help="Optional JSONL file to write converted examples")
    parser.add_argument("--output_format", type=str, choices=["text_only", "prompt_completion"], default="text_only", help="Output JSONL format")
    parser.add_argument("--synthesize_complaints", action="store_true", help="Generate synthetic legal complaint outputs from raw text")
    parser.add_argument("--max_rows", type=int, default=20, help="Number of sample rows to show")
    return parser.parse_args()


def build_example(row, text_field, complaint_field, output_format, synthesize):
    text = str(row.get(text_field, "")).strip()
    if not text:
        return None

    if output_format == "text_only":
        return {"text": text}

    if complaint_field and not synthesize:
        complaint = str(row.get(complaint_field, "")).strip()
        if not complaint:
            return None
        return {
            "instruction": "ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন",
            "input": text,
            "output": complaint,
        }

    return {
        "instruction": "ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন",
        "input": text,
        "output": build_synthetic_complaint(text),
    }


def summarize_dataframe(df, max_rows=20):
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print("dtypes:")
    print(df.dtypes)
    print("null counts:")
    print(df.isna().sum().to_dict())
    print("unique counts:")
    print(df.nunique(dropna=False).to_dict())
    print("sample rows:")
    print(df.head(max_rows).to_string(index=False))


def main():
    args = parse_args()
    df = load_dataset_frame(args.input_file, args.sheet_name)

    text_field = args.text_field or resolve_text_column(df)
    complaint_field = args.complaint_field or resolve_complaint_column(df)

    print("=== Dataset summary ===")
    summarize_dataframe(df, max_rows=args.max_rows)
    print(f"Resolved text column: {text_field}")
    if complaint_field:
        print(f"Resolved complaint column: {complaint_field}")
    else:
        print("No complaint column detected; synthetic complaints will be generated.")

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with args.output_file.open("w", encoding="utf-8") as writer:
            for _, row in df.iterrows():
                example = build_example(
                    row,
                    text_field,
                    complaint_field,
                    args.output_format,
                    args.synthesize_complaints,
                )
                if example is None:
                    continue
                writer.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
        print(f"Wrote {written} examples to {args.output_file}")
    else:
        print("No output file specified. Dataset summary only.")


if __name__ == "__main__":
    main()
