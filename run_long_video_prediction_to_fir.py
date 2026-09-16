import os
import re
import sys
import cv2
import subprocess
import pandas as pd
from pathlib import Path

# =========================
# Auto path detection
# =========================

CANDIDATE_CSV_FILES = [
    Path("data/videos/video/new_videos/long_videos/long_video_segments.csv"),
    Path("videos/video/new_videos/long_videos/long_video_segments.csv"),
    Path("data/new_videos/long_video_segments.csv"),
]

INPUT_FILE = None
for p in CANDIDATE_CSV_FILES:
    if p.exists():
        INPUT_FILE = p
        break

if INPUT_FILE is None:
    raise FileNotFoundError(
        "long_video_segments.csv পাওয়া যায়নি। "
        "CSV file রাখুন: data/videos/video/new_videos/long_videos/long_video_segments.csv"
    )

VIDEO_DIRS = [
    INPUT_FILE.parent,
    Path("data/videos/video/new_videos/long_videos"),
    Path("videos/video/new_videos/long_videos"),
    Path("data/new_videos/video"),
]

PREDICT_SCRIPT = Path("vision/predict_from_video.py")
MODEL_DIR = Path("output/sign_sentence_full_gpu")

OUTPUT_DIR = Path("outputs/long_video_prediction")
CLIP_DIR = OUTPUT_DIR / "clips"
FIR_DIR = OUTPUT_DIR / "fir"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)
FIR_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helper functions
# =========================

def parse_time_to_seconds(value):
    """
    Supports:
    00:06 -> 6 sec
    00:01:05 -> 65 sec
    0.06 -> 6 sec, for your earlier sheet style
    """
    text = str(value).strip()

    if ":" in text:
        parts = text.split(":")
        parts = [float(p) for p in parts]

        if len(parts) == 2:
            minute, second = parts
            return minute * 60 + second

        if len(parts) == 3:
            hour, minute, second = parts
            return hour * 3600 + minute * 60 + second

    # handle 0.06 as 6 second, 0.31 as 31 second
    if "." in text:
        left, right = text.split(".", 1)
        if left == "0" and right.isdigit():
            return float(int(right))

    return float(text)


def find_video_file(video_file):
    video_path = Path(str(video_file))

    if video_path.exists():
        return video_path

    for video_dir in VIDEO_DIRS:
        candidate = video_dir / str(video_file)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Video file পাওয়া যায়নি: {video_file}")


def cut_video_segment(input_video, start_sec, end_sec, output_clip):
    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise RuntimeError(f"Video open করা যায়নি: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    if end_frame <= start_frame:
        cap.release()
        raise RuntimeError("Invalid segment time: end_time must be greater than start_time")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_clip), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()

    return output_clip


def predict_clip(clip_path):
    if not PREDICT_SCRIPT.exists():
        raise FileNotFoundError("vision/predict_from_video.py পাওয়া যায়নি।")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--model_dir",
        str(MODEL_DIR),
        "--video",
        str(clip_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )

    output_text = result.stdout + "\n" + result.stderr

    predicted_sentence = ""
    confidence = ""

    for line in output_text.splitlines():
        line = line.strip()

        # Expected line format:
        # 1. আমার চাচা চোর | confidence=0.9999
        if line.startswith("1."):
            temp = line.split(".", 1)[1].strip()
            predicted_sentence = temp.split("|")[0].strip()

            match = re.search(r"confidence\s*=\s*([0-9.]+)", line)
            if match:
                confidence = match.group(1)

            break

    if not predicted_sentence:
        predicted_sentence = "PREDICTION_FAILED"

    return predicted_sentence, confidence, output_text


def join_sentences(sentence_list):
    clean_sentences = []

    for sentence in sentence_list:
        sentence = str(sentence).strip()

        if not sentence or sentence == "PREDICTION_FAILED":
            continue

        sentence = sentence.rstrip("।")

        if sentence:
            clean_sentences.append(sentence)

    if not clean_sentences:
        return ""

    return "। ".join(clean_sentences) + "।"


def auto_classify_sentence(sentence):
    text = str(sentence)

    legal_words = ["থানা", "পুলিশ", "অভিযোগ", "আইনগত", "সাহায্য চাই"]
    crime_words = [
        "চোর", "চুরি", "টাকা", "নিয়ে গেছে", "নিয়ে গেছে",
        "মোবাইল", "ব্যাগ", "হুমকি", "ভয়", "ভয়",
        "আঘাত", "ব্যথা", "মারে", "মারধর"
    ]

    if any(word in text for word in legal_words):
        return "legal_intent", "keep"

    if any(word in text for word in crime_words):
        return "crime", "keep"

    return "unrelated", "remove"


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
    if not filtered_story:
        return "এই long video থেকে আইনগত অভিযোগ তৈরির জন্য যথেষ্ট relevant sentence পাওয়া যায়নি।"

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


# =========================
# Main process
# =========================

print("Using CSV:", INPUT_FILE)

df = pd.read_csv(INPUT_FILE, encoding="utf-8")
prediction_rows = []

for _, row in df.iterrows():
    story_id = str(row["story_id"])
    video_file = str(row["video_file"])
    segment_no = int(row["segment_no"])

    start_time = row["start_time"]
    end_time = row["end_time"]

    start_sec = parse_time_to_seconds(start_time)
    end_sec = parse_time_to_seconds(end_time)

    video_path = find_video_file(video_file)

    clip_name = f"{story_id}_segment_{segment_no}.mp4"
    clip_path = CLIP_DIR / clip_name

    print("=" * 70)
    print(f"Story: {story_id}")
    print(f"Segment: {segment_no}")
    print(f"Time: {start_time} - {end_time}")
    print(f"Cutting clip: {clip_path}")

    cut_video_segment(video_path, start_sec, end_sec, clip_path)

    print("Predicting...")
    predicted_sentence, confidence, raw_output = predict_clip(clip_path)

    auto_type, auto_filter_decision = auto_classify_sentence(predicted_sentence)

    expected_sentence = row["sentence"] if "sentence" in df.columns else ""

    print("Expected Sentence:", expected_sentence)
    print("Predicted Sentence:", predicted_sentence)
    print("Confidence:", confidence)
    print("Auto Type:", auto_type)
    print("Auto Filter:", auto_filter_decision)

    prediction_rows.append({
        "story_id": story_id,
        "video_file": video_file,
        "segment_no": segment_no,
        "start_time": start_time,
        "end_time": end_time,
        "expected_sentence": expected_sentence,
        "predicted_sentence": predicted_sentence,
        "confidence": confidence,
        "auto_type": auto_type,
        "auto_filter_decision": auto_filter_decision,
        "clip_file": str(clip_path)
    })


pred_df = pd.DataFrame(prediction_rows)

predicted_csv = OUTPUT_DIR / "predicted_long_video_segments.csv"
pred_df.to_csv(predicted_csv, index=False, encoding="utf-8-sig")

print("=" * 70)
print("Predicted CSV saved:", predicted_csv)


# =========================
# Generate FIR from predicted sentences
# =========================

for story_id, group in pred_df.groupby("story_id"):
    group = group.sort_values("segment_no")

    original_predicted_story = join_sentences(group["predicted_sentence"].tolist())

    keep_df = group[group["auto_filter_decision"] == "keep"]
    remove_df = group[group["auto_filter_decision"] == "remove"]

    filtered_story = join_sentences(keep_df["predicted_sentence"].tolist())
    clean_fir = build_clean_fir(filtered_story)

    removed_lines = []
    for _, row in remove_df.iterrows():
        removed_lines.append(
            f'- {row["start_time"]}-{row["end_time"]}: {row["predicted_sentence"]}'
        )

    output_text = f"""Story ID:
{story_id}

Original Predicted Story:
{original_predicted_story}

Removed Predicted Sentences:
{chr(10).join(removed_lines) if removed_lines else "None"}

Filtered Predicted Story:
{filtered_story}

Clean FIR:
{clean_fir}
"""

    output_file = FIR_DIR / f"{story_id}_predicted_clean_fir.txt"
    output_file.write_text(output_text, encoding="utf-8")

    print("=" * 70)
    print("Final FIR for:", story_id)
    print(output_text)
    print("Saved:", output_file)


