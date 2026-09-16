import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("data/story_tests/filter_test_stories.csv")
OUTPUT_FILE = Path("outputs/story_fir/story_filter_preview.txt")


def load_stories(input_file):
    stories = defaultdict(list)

    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="|")

        for row in reader:
            story_id = row["story_id"].strip()
            segment_no = int(row["segment_no"])
            sentence = row["sentence"].strip()
            sentence_type = row["type"].strip()
            filter_decision = row["filter_decision"].strip()

            stories[story_id].append({
                "segment_no": segment_no,
                "sentence": sentence,
                "type": sentence_type,
                "filter_decision": filter_decision,
            })

    return stories


def build_story_text(rows, only_keep=False):
    selected_sentences = []

    for row in sorted(rows, key=lambda x: x["segment_no"]):
        if only_keep:
            if row["filter_decision"] == "keep":
                selected_sentences.append(row["sentence"])
        else:
            selected_sentences.append(row["sentence"])

    return " ".join(selected_sentences)


def main():
    stories = load_stories(INPUT_FILE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    for story_id, rows in stories.items():
        rows = sorted(rows, key=lambda x: x["segment_no"])

        original_story = build_story_text(rows, only_keep=False)
        filtered_story = build_story_text(rows, only_keep=True)

        removed_sentences = [
            row["sentence"]
            for row in rows
            if row["filter_decision"] == "remove"
        ]

        lines.append("=" * 70)
        lines.append(f"Story ID: {story_id}")
        lines.append("=" * 70)

        lines.append("\nOriginal Story:")
        lines.append(original_story)

        lines.append("\nRemoved Sentences:")
        if removed_sentences:
            for sentence in removed_sentences:
                lines.append(f"- {sentence}")
        else:
            lines.append("- None")

        lines.append("\nFiltered Story:")
        lines.append(filtered_story)

        lines.append("\n")

    final_text = "\n".join(lines)

    print(final_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\nSaved output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
