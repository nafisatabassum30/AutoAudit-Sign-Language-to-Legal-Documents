import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm.dataset_utils import load_dataset_frame, resolve_dataset_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an Excel dataset file to CSV.")
    parser.add_argument("--input_file", type=Path, default=None, help="Path to the source Excel file")
    parser.add_argument("--output_file", type=Path, default=None, help="Path to the output CSV file")
    parser.add_argument("--sheet_name", type=str, default=None, help="Excel sheet to read if the input is XLSX")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = resolve_dataset_path(args.input_file)
    output_path = args.output_file or input_path.with_suffix('.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset_frame(input_path, args.sheet_name)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✅ Converted {input_path} to {output_path}")
    print(f"📋 Rows: {len(df)}, Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()