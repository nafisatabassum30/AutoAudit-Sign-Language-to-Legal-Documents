import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("data/story_tests/filter_test_stories.csv")
OUTPUT_DIR = Path("outputs/story_fir")
ADAPTER_DIR = "output/llm_lora_banglat5_final"
INSTRUCTION = "ভিডিওতে দেওয়া বর্ণনার উপর ভিত্তি করে একটি আইনগত অভিযোগ তৈরি করুন"


def load_stories(input_file):
    stories = defaultdict(list)

    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="|")

        for row in reader:
            stories[row["story_id"].strip()].append({
                "segment_no": int(row["segment_no"]),
                "sentence": row["sentence"].strip(),
                "type": row["type"].strip(),
                "filter_decision": row["filter_decision"].strip(),
            })

    return stories


def build_story(rows, keep_only=False):
    sentences = []

    for row in sorted(rows, key=lambda x: x["segment_no"]):
        if keep_only:
            if row["filter_decision"] == "keep":
                sentences.append(row["sentence"])
        else:
            sentences.append(row["sentence"])

    return " ".join(sentences)


def run_llm(filtered_story):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    command = [
        sys.executable,
        "llm/infer_lora_t5.py",
        "--adapter_dir",
        ADAPTER_DIR,
        "--instruction",
        INSTRUCTION,
        "--transcript",
        filtered_story,
        "--max_new_tokens",
        "500",
        "--num_beams",
        "4",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("LLM generation failed")

    marker = "=== Generated Legal Complaint ==="

    if marker in result.stdout:
        return result.stdout.split(marker, 1)[1].strip()

    return result.stdout.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stories = load_stories(INPUT_FILE)

    for story_id, rows in stories.items():
        original_story = build_story(rows, keep_only=False)
        filtered_story = build_story(rows, keep_only=True)

        removed = [
            row["sentence"]
            for row in sorted(rows, key=lambda x: x["segment_no"])
            if row["filter_decision"] == "remove"
        ]

        print("\n" + "=" * 70)
        print(f"Story ID: {story_id}")
        print("=" * 70)

        print("\nOriginal Story:")
        print(original_story)

        print("\nRemoved Sentences:")
        for sentence in removed:
            print("-", sentence)

        print("\nFiltered Story:")
        print(filtered_story)

        print("\nGenerating FIR...")
        fir_text = run_llm(filtered_story)

        print("\nGenerated FIR:")
        print(fir_text)

        output_file = OUTPUT_DIR / f"{story_id}_fir.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Story ID:\n")
            f.write(story_id + "\n\n")

            f.write("Original Story:\n")
            f.write(original_story + "\n\n")

            f.write("Removed Sentences:\n")
            for sentence in removed:
                f.write("- " + sentence + "\n")

            f.write("\nFiltered Story:\n")
            f.write(filtered_story + "\n\n")

            f.write("Generated FIR:\n")
            f.write(fir_text + "\n")

        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
