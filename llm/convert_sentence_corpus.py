import argparse
import json
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Bangla sentence corpus into JSONL training data for LLM fine-tuning.")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to the raw Excel corpus file")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to the output JSONL file")
    parser.add_argument("--text_field", type=str, default="Names", help="Column name containing Bangla text")
    parser.add_argument("--instruction", type=str, default="Bangla বাক্য পূরণ করুন", help="Instruction template for each example")
    parser.add_argument("--example_type", type=str, choices=["prompt_completion", "text_only"], default="prompt_completion", help="Training format type")
    return parser.parse_args()


def convert_example(text: str, instruction: str, example_type: str):
    text = text.strip()
    if example_type == "prompt_completion":
        prompt = f"নির্দেশ: {instruction}\nইনপুট: {text}\nউত্তর:"
        completion = text
        return {"instruction": instruction, "input": text, "output": completion}
    return {"text": text}


def main():
    args = parse_args()
    df = pd.read_excel(args.input_file)
    output_file = args.output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as writer:
        count = 0
        for _, row in df.iterrows():
            text = str(row.get(args.text_field, "")).strip()
            if not text:
                continue
            example = convert_example(text, args.instruction, args.example_type)
            writer.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} examples to {output_file}")


if __name__ == "__main__":
    main()
