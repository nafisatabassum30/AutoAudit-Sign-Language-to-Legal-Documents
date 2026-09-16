import os
import sys
import re
import subprocess
from pathlib import Path

import cv2
import pandas as pd

VIDEO_FILE = Path(r"data\Test_data\long_videos\story_5_normal_test.mp4")
SEGMENT_FILE = Path(r"data\Test_data\sheets\story_5_long_video_segments.csv")
PREDICT_SCRIPT = Path(r"vision\predict_from_video_v2_exact.py")
MODEL_DIR = Path(r"output\sign_sentence_v2_validated")

OUT_DIR = Path(r"outputs\Test_data_result\story5_segmented")
CLIP_DIR = OUT_DIR / "clips"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)

def time_to_sec(t):
    t = str(t).strip()
    parts = t.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return float(t)

def normalize_text(x):
    x = str(x).strip()
    x = x.replace("।", "").replace(".", "").replace("?", "")
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def cut_clip(start_sec, end_sec, out_file):
    cap = cv2.VideoCapture(str(VIDEO_FILE))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    while True:
        pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if pos_sec > end_sec:
            break

        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)

    cap.release()
    writer.release()

def predict_clip(clip_file):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--model_dir",
        str(MODEL_DIR),
        "--video",
        str(clip_file)
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

    pred = ""
    conf = ""

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("1."):
            part = line.replace("1.", "").strip()
            if "| confidence=" in part:
                pred, conf = part.split("| confidence=")
                pred = pred.strip()
                conf = conf.strip()
            else:
                pred = part.strip()

    return pred, conf

df = pd.read_csv(SEGMENT_FILE)
rows = []

for _, row in df.iterrows():
    seg_no = int(row["segment_no"])
    start_sec = time_to_sec(row["start_time"])
    end_sec = time_to_sec(row["end_time"])
    expected = str(row["sentence"]).strip()

    clip_file = CLIP_DIR / f"story5_segment_{seg_no:02d}.mp4"

    print("=" * 80)
    print(f"Segment {seg_no}: {start_sec}s to {end_sec}s")
    print("Expected:", expected)

    cut_clip(start_sec, end_sec, clip_file)
    predicted, confidence = predict_clip(clip_file)

    match = "YES" if normalize_text(expected) == normalize_text(predicted) else "NO"

    print("Predicted:", predicted)
    print("Confidence:", confidence)
    print("Match:", match)

    rows.append({
        "segment_no": seg_no,
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "expected_sentence": expected,
        "predicted_sentence": predicted,
        "confidence": confidence,
        "type": row["type"],
        "filter_decision": row["filter_decision"],
        "original_index": row["original_index"],
        "match": match,
        "clip_file": str(clip_file)
    })

out = pd.DataFrame(rows)
out_file = OUT_DIR / "story5_segmented_validation_report.csv"
out.to_csv(out_file, index=False, encoding="utf-8-sig")

correct = (out["match"] == "YES").sum()
total = len(out)

print("=" * 80)
print("Story 5 segmented validation")
print("Total:", total)
print("Correct:", correct)
print("Wrong:", total - correct)
print("Accuracy:", round(correct / total, 4) if total else 0)
print("Saved:", out_file)
