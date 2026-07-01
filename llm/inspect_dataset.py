import argparse
import json
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a Bangla dataset file and optionally export it as JSONL for LLM training.")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the raw dataset file (Excel, CSV, JSON, JSONL)")
    parser.add_argument("--sheet_name", type=str, default=None, help="Excel sheet to read if the input is XLSX")
    parser.add_argument("--text_field", type=str, default="Names", help="Field containing Bangla text")
    parser.add_argument("--complaint_field", type=str, default=None, help="Field containing complaint text if available")
    parser.add_argument("--output_file", type=Path, default=None, help="Optional JSONL file to write converted examples")
    parser.add_argument("--output_format", type=str, choices=["text_only", "prompt_completion"], default="text_only", help="Output JSONL format")
    parser.add_argument("--synthesize_complaints", action="store_true", help="Generate synthetic legal complaint outputs from raw text")
    parser.add_argument("--max_rows", type=int, default=20, help="Number of sample rows to show")
    return parser.parse_args()


def load_raw_data(input_file: Path, sheet_name: str = None):
    suffix = input_file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_file)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(input_file, lines=(suffix == ".jsonl"))
    if suffix in {".xls", ".xlsx"}:
        if sheet_name:
            return pd.read_excel(input_file, sheet_name=sheet_name)
        excel = pd.ExcelFile(input_file)
        if len(excel.sheet_names) == 1:
            return pd.read_excel(input_file, sheet_name=0)
        raise ValueError(f"Excel file contains multiple sheets: {excel.sheet_names}. Specify --sheet_name.")
    raise ValueError("Unsupported input file format. Use CSV, JSON, JSONL, or Excel.")


def synthesize_complaint_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).strip().rstrip('.')
    return f"দায়েরকারী অভিযোগ করেন যে {text} এবং আইনি ব্যবস্থা গ্রহণের অনুরোধ জানানো হয়েছে।"


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
        "output": synthesize_complaint_text(text),
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
    df = load_raw_data(args.input_file, args.sheet_name)

    print("=== Dataset summary ===")
    summarize_dataframe(df, max_rows=args.max_rows)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with args.output_file.open("w", encoding="utf-8") as writer:
            for _, row in df.iterrows():
                example = build_example(
                    row,
                    args.text_field,
                    args.complaint_field,
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
