import pandas as pd
from pathlib import Path

INPUT_FILE = "data/new_videos/long_video_segments.csv"
OUTPUT_DIR = Path("outputs/long_video_fir_clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def join_sentences(sentence_list):
    clean_sentences = []

    for sentence in sentence_list:
        sentence = str(sentence).strip()
        sentence = sentence.rstrip("।")

        if sentence:
            clean_sentences.append(sentence)

    if not clean_sentences:
        return ""

    return "। ".join(clean_sentences) + "।"


def detect_case_type(text):
    if "মোবাইল" in text and (
        "নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "চুরি" in text
    ):
        return "মোবাইল চুরি / ছিনতাই"

    if (
        "ব্যাগ নিয়ে গেছে" in text
        or "ব্যাগ নিয়ে গেছে" in text
        or "ব্যাগ চুরি" in text
        or "ব্যাগ কেটে" in text
    ):
        return "ব্যাগ চুরি / ছিনতাই"

    if "হুমকি" in text or "ভয়" in text or "ভয়" in text:
        return "হুমকি প্রদান"

    if "আঘাত" in text or "ব্যথা" in text or "মারে" in text or "মারধর" in text:
        return "শারীরিক আঘাত / মারধর"

    if "টাকা" in text and (
        "নিয়ে গেছে" in text or "নিয়ে গেছে" in text or "আত্মসাৎ" in text
    ):
        return "টাকা নেওয়া / আর্থিক ক্ষতি"

    if "চোর" in text or "চুরি" in text:
        return "চুরি সংক্রান্ত অভিযোগ"

    return "আইনগত অভিযোগ"


def build_clean_fir(filtered_story):
    case_type = detect_case_type(filtered_story)

    return f"""ফৌজদারি অভিযোগ / FIR

মামলা নং: ____________________

ঘটনার তারিখ: ____________________
ঘটনার সময়: ____________________
ঘটনার স্থান: ____________________

বাদীর নাম: ____________________
বাদীর ঠিকানা: ____________________

ঘটনার বিবরণ:
দায়েরকারী অভিযোগ করেন যে {filtered_story}

ঘটনার ধরন:
{case_type}

অভিযুক্ত ব্যক্তি বা প্রতিষ্ঠান:
অজ্ঞাত / উল্লিখিত ব্যক্তি

প্রমাণের বর্ণনা:
প্রয়োজনীয় সাক্ষ্য, ভিডিও ফুটেজ, প্রত্যক্ষদর্শীর বক্তব্য বা অন্যান্য প্রমাণ তদন্তসাপেক্ষে সংযোজনীয়।

আবেদন:
বাদী এই ঘটনার বিষয়ে দ্রুত আইনানুগ ব্যবস্থা গ্রহণের জন্য বিনীত অনুরোধ করেন।

স্বাক্ষর: ____________________
তারিখ: ____________________
"""


df = pd.read_csv(INPUT_FILE, encoding="utf-8")

for story_id, group in df.groupby("story_id"):
    group = group.sort_values("segment_no")

    original_story = join_sentences(group["sentence"].tolist())

    keep_df = group[group["filter_decision"].str.lower() == "keep"]
    remove_df = group[group["filter_decision"].str.lower() == "remove"]

    filtered_story = join_sentences(keep_df["sentence"].tolist())
    clean_fir = build_clean_fir(filtered_story)

    removed_lines = []
    for _, row in remove_df.iterrows():
        removed_lines.append(
            f'- {row["start_time"]}-{row["end_time"]}: {row["sentence"]}'
        )

    output_text = f"""Story ID:
{story_id}

Video File:
{group.iloc[0]["video_file"]}

Original Story:
{original_story}

Removed Sentences:
{chr(10).join(removed_lines) if removed_lines else "None"}

Filtered Story:
{filtered_story}

Clean FIR:
{clean_fir}
"""

    output_file = OUTPUT_DIR / f"{story_id}_long_video_clean_fir.txt"
    output_file.write_text(output_text, encoding="utf-8")

    print("=" * 70)
    print("Story ID:", story_id)
    print()
    print("Original Story:")
    print(original_story)
    print()
    print("Removed Sentences:")
    print(chr(10).join(removed_lines) if removed_lines else "None")
    print()
    print("Filtered Story:")
    print(filtered_story)
    print()
    print("Clean FIR:")
    print(clean_fir)
    print("Saved:", output_file)
