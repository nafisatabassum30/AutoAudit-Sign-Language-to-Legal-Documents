# simple_prepare.py
import argparse
import json
from pathlib import Path

import pandas as pd
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic FIR-style training data from a Bangla sentence corpus.")
    parser.add_argument("--input_file", type=Path, default=None, help="Path to the raw dataset file (Excel or CSV).")
    parser.add_argument("--output_file", type=Path, default=None, help="Path to the output JSONL file.")
    parser.add_argument("--sheet_name", type=str, default=None, help="Excel sheet name to read if the input file has multiple sheets.")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit number of rows for quick experiments.")
    parser.add_argument("--inject_offense", action="store_true", help="Inject an offense type into the generated legal summary.")
    parser.add_argument("--offense_types", type=str, default=None, help="Comma-separated list of offense types to use (e.g. 'ধর্ষণ,চুরি,মাদক ব্যবসা').")
    parser.add_argument("--use_synthetic_crimes", action="store_true", help="Ignore input file and use built-in synthetic crime sentences.")
    parser.add_argument("--synthetic_count", type=int, default=100, help="Number of synthetic crime sentences to generate when using synthetic crimes.")
    return parser.parse_args()


def load_corpus(input_path: Path, sheet_name: str = None):
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xls", ".xlsx"}:
        if sheet_name is None:
            return pd.read_excel(input_path, sheet_name=0)
        return pd.read_excel(input_path, sheet_name=sheet_name)
    raise ValueError("Unsupported input file format. Use .csv, .xls, or .xlsx.")


LEGAL_SUMMARY_TEMPLATES = [
    "বাদী অভিযোগ করেন যে, {sentence} সম্পর্কিত ঘটনা ঘটেছে এবং এতে তিনি গুরুতর ক্ষতির সম্মুখীন হয়েছেন।",
    "বাদী জানান যে, {sentence} সম্পর্কিত বিরোধের কারণে অপর পক্ষের বিরুদ্ধে আইনি ব্যবস্থা গ্রহণের প্রয়োজন হয়েছে।",
    "বাদী বলেন যে, {sentence} নিয়ে তার ও অপর ব্যক্তির মধ্যে তিক্ত সমস্যা দেখা দিয়েছে এবং তিনি দ্রুত বিচার চান।",
    "বাদী লিখেছেন যে, {sentence} বিষয়ক ঘটনায় তার আত্মীয় বা সম্পত্তি ক্ষতিগ্রস্ত হয়েছে এবং তিনি আইনি সহায়তা চেয়েছেন।",
]


def build_legal_summary(sentence: str, offense: str | None = None) -> str:
    sentence = sentence.strip()
    if sentence.endswith('.'):
        sentence = sentence[:-1]
    template = LEGAL_SUMMARY_TEMPLATES[abs(hash(sentence)) % len(LEGAL_SUMMARY_TEMPLATES)]
    base = template.format(sentence=sentence)
    if offense:
        # Append an explicit offense mention to the summary
        return f"{base} অভিযুক্তের বিরুদ্ধে অভিযোগ: {offense}."
    return base


def build_fir_text(summary: str) -> str:
    return f"""মামলা নং: _______________
ঘটনার তারিখ: _______________
ঘটনার সময়: _______________
ঘটনার স্থান: _______________

ঘটনার বিবরণ:
{summary}

অভিযোগিত পক্ষ: _______________
প্রমাণ: _______________

বাদী অভিযোগ দায়ের করেছেন এবং দ্রুত প্রয়োজনীয় আইনানুগ ব্যবস্থা গ্রহণের অনুরোধ জানিয়েছেন।

বাদীর নাম: _______________
বাদীর ঠিকানা: _______________

স্বাক্ষর: _______________
তারিখ: _______________"""


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    input_path = args.input_file or root / "data" / "raw" / "FinalSheet2.xlsx"
    output_path = args.output_file or root / "data" / "train" / "bangla_legal_train.jsonl"

    assert input_path.exists(), f"Dataset file not found: {input_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.use_synthetic_crimes:
        # Built-in small synthetic crime sentence list (non-graphic)
        CRIME_SENTENCES = [
            "এক ব্যক্তি আমার দোকান থেকে জিনিস চুরি করেছে",
            "অপর ব্যক্তি আমার জমি দখল করার চেষ্টা করেছে",
            "মেডিক্যাল স্টোরে মাদক ব্যবসার তথ্য পাওয়া গেছে",
            "অপর রহস্যজনকভাবে আমার ব্যাঙ্ক অ্যাকাউন্ট থেকে অর্থ সরিয়ে নিয়েছে",
            "রাস্তায় গাড়ি ধাক্কা দিয়ে আহত করেছে",
            "অপর ব্যক্তি আমার বাড়িতে অনুপ্রবেশ করে সম্পত্তি ক্ষতিসাধন করেছে",
            "ধর্ষণের চেষ্টা অথবা যৌন হেনস্থার অভিযোগ আছে",
        ]
        # Expand or truncate to requested synthetic_count
        repeats = max(1, (args.synthetic_count + len(CRIME_SENTENCES) - 1) // len(CRIME_SENTENCES))
        flat = (CRIME_SENTENCES * repeats)[: args.synthetic_count]
        df = pd.DataFrame({"Names": flat})
    else:
        df = load_corpus(input_path, args.sheet_name)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    training_data = []
    # Prepare offense types list
    offense_list = None
    if args.offense_types:
        offense_list = [o.strip() for o in args.offense_types.split(",") if o.strip()]
    else:
        offense_list = ["ধর্ষণ", "চুরি", "মাদক ব্যবসা", "হত্যা", "চাঁদাবাজি"]

    for _, row in df.iterrows():
        sentence = str(row.get("Names", "")).strip()
        if not sentence:
            continue
        selected_offense = None
        if args.inject_offense:
            selected_offense = random.choice(offense_list) if offense_list else None
        summary = build_legal_summary(sentence, selected_offense)
        example = {
            "instruction": "Convert the following incident summary into a formal First Information Report (FIR) in Bengali.",
            "input": summary,
            "output": build_fir_text(summary),
        }
        training_data.append(example)

    with output_path.open("w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(training_data)} examples to {output_path}")


if __name__ == "__main__":
    main()
