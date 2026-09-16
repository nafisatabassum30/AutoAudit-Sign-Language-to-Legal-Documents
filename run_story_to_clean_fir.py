import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("data/story_tests/filter_test_stories.csv")
OUTPUT_DIR = Path("outputs/story_fir_clean")


def load_stories():
    stories = defaultdict(list)

    with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
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


def detect_case_type(filtered_story):
    text = filtered_story

    if "মোবাইল" in text and ("নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "চুরি" in text):
        return "মোবাইল চুরি / ছিনতাই"

    if "ব্যাগ নিয়ে গেছে" in text or "ব্যাগ নিয়ে গেছে" in text or "ব্যাগ চুরি" in text or "ব্যাগ কেটে" in text:
        return "ব্যাগ চুরি / ছিনতাই"

    if "হুমকি" in text or "ভয়" in text or "ভয়" in text:
        return "হুমকি প্রদান"

    if "আঘাত" in text or "ব্যথা" in text:
        return "শারীরিক আঘাত"

    if "টাকা" in text and ("নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "আত্মসাৎ" in text):
        return "টাকা নেওয়া / আর্থিক ক্ষতি"

    return "আইনগত অভিযোগ"


def build_clean_fir(filtered_story):
    case_type = detect_case_type(filtered_story)

    fir = f"""ফৌজদারি অভিযোগ / FIR

মামলা নং: ____________________

ঘটনার তারিখ: ____________________
ঘটনার সময়: ____________________
ঘটনার স্থান: ____________________

বাদীর নাম: ____________________
বাদীর ঠিকানা: ____________________

ঘটনার ধরন:
{case_type}

ঘটনার বিবরণ:
দায়েরকারী অভিযোগ করেন যে {filtered_story}

অভিযুক্ত ব্যক্তি বা প্রতিষ্ঠান:
অজ্ঞাত / উল্লিখিত ব্যক্তি

প্রমাণের বর্ণনা:
প্রয়োজনীয় সাক্ষ্য, ভিডিও ফুটেজ, প্রত্যক্ষদর্শীর বক্তব্য বা অন্যান্য প্রমাণ তদন্তসাপেক্ষে সংযোজনীয়।

আবেদন:
উক্ত ঘটনার বিষয়ে দ্রুত আইনানুগ ব্যবস্থা গ্রহণের জন্য বিনীত অনুরোধ করা হলো।

স্বাক্ষর: ____________________
তারিখ: ____________________
"""

    return fir


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stories = load_stories()

    for story_id, rows in stories.items():
        rows = sorted(rows, key=lambda x: x["segment_no"])

        original_story = build_story(rows, keep_only=False)
        filtered_story = build_story(rows, keep_only=True)

        removed_sentences = [
            row["sentence"]
            for row in rows
            if row["filter_decision"] == "remove"
        ]

        clean_fir = build_clean_fir(filtered_story)

        print("\n" + "=" * 70)
        print(f"Story ID: {story_id}")
        print("=" * 70)

        print("\nOriginal Story:")
        print(original_story)

        print("\nRemoved Sentences:")
        for sentence in removed_sentences:
            print("- " + sentence)

        print("\nFiltered Story:")
        print(filtered_story)

        print("\nClean FIR:")
        print(clean_fir)

        output_file = OUTPUT_DIR / f"{story_id}_clean_fir.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Story ID:\n")
            f.write(story_id + "\n\n")

            f.write("Original Story:\n")
            f.write(original_story + "\n\n")

            f.write("Removed Sentences:\n")
            for sentence in removed_sentences:
                f.write("- " + sentence + "\n")

            f.write("\nFiltered Story:\n")
            f.write(filtered_story + "\n\n")

            f.write("Clean FIR:\n")
            f.write(clean_fir)

        print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
