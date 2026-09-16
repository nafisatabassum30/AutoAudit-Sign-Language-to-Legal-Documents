import os
import sys
import subprocess
import pandas as pd
import re
from pathlib import Path

TEST_DIR = Path("data/Test_data")
LABEL_FILE = TEST_DIR / "sheets" / "Sentence xLabel.xlsx"
VIDEO_ROOT = TEST_DIR / "single_videos"

MODEL_DIR = Path("output/sign_sentence_v2_validated")
PREDICT_SCRIPT = Path("vision/predict_from_video_v2.py")

OUTPUT_FILE = Path("outputs/Test_data_result/single_video_prediction_result_v2.csv")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text):
    text = str(text).strip()
    text = text.replace("?", "")
    text = text.replace("？", "")
    text = text.replace("।", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_video_number(x):
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_video(video_number):
    num = clean_video_number(video_number)
    all_videos = list(VIDEO_ROOT.rglob("*.mp4"))

    possible_names = [
        f"{num}.mp4",
        f"{num} (2).mp4",
        f"{num}_sentence3_withBg.mp4",
    ]

    for name in possible_names:
        for video in all_videos:
            if video.name.strip() == name:
                return video

    for video in all_videos:
        stem = video.stem.strip()
        if stem == num:
            return video
        if stem.startswith(num + " "):
            return video
        if stem.startswith(num + "_"):
            return video

    return None


def predict_video_live(video_path):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--model_dir",
        str(MODEL_DIR),
        "--video",
        str(video_path)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )

    output = result.stdout + "\n" + result.stderr

    predicted_sentence = ""
    confidence = ""

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("1."):
            part = line.replace("1.", "").strip()
            if "| confidence=" in part:
                predicted_sentence, conf_part = part.split("| confidence=")
                predicted_sentence = predicted_sentence.strip()
                confidence = conf_part.strip()
            else:
                predicted_sentence = part.strip()

    return predicted_sentence, confidence


df = pd.read_excel(LABEL_FILE)

rows = []

for _, row in df.iterrows():
    label_class = row.get("Label/Class")
    expected_sentence = str(row.get("Meaning", "")).strip()
    video_number = row.get("Videos")

    video_path = find_video(video_number)

    if video_path is None:
        rows.append({
            "label_class": label_class,
            "video_number": video_number,
            "video_file": "NOT_FOUND",
            "expected_sentence": expected_sentence,
            "predicted_sentence": "",
            "confidence": "",
            "exact_match": "NO",
            "normalized_match": "NO",
            "status": "VIDEO_MISSING"
        })
        continue

    print("=" * 80)
    print("Testing:", video_path)
    print("Expected:", expected_sentence)

    predicted_sentence, confidence = predict_video_live(video_path)

    exact_match = "YES" if predicted_sentence.strip() == expected_sentence.strip() else "NO"
    normalized_match = "YES" if normalize_text(predicted_sentence) == normalize_text(expected_sentence) else "NO"

    print("Predicted:", predicted_sentence)
    print("Confidence:", confidence)
    print("Exact Match:", exact_match)
    print("Normalized Match:", normalized_match)

    rows.append({
        "label_class": label_class,
        "video_number": video_number,
        "video_file": str(video_path),
        "expected_sentence": expected_sentence,
        "predicted_sentence": predicted_sentence,
        "confidence": confidence,
        "exact_match": exact_match,
        "normalized_match": normalized_match,
        "status": "OK"
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nSaved:", OUTPUT_FILE)
print("\nExact match summary:")
print(out_df["exact_match"].value_counts(dropna=False))

print("\nNormalized match summary:")
print(out_df["normalized_match"].value_counts(dropna=False))
