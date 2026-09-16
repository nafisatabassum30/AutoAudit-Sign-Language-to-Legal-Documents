import os
import sys
import subprocess
from pathlib import Path

import cv2
import pandas as pd

VIDEO_PATH = Path("data/Test_data/long_videos/longv 1.mp4")
MODEL_DIR = Path("output/sign_sentence_v2_validated")
PREDICT_SCRIPT = Path("vision/predict_from_video_v2_exact.py")

OUT_DIR = Path("outputs/Test_data_result/long_no_segment")
CLIP_DIR = OUT_DIR / "clips"
OUT_CSV = OUT_DIR / "long_video_no_segment_prediction.csv"
OUT_TXT = OUT_DIR / "long_video_predicted_story.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CLIP_DIR.mkdir(parents=True, exist_ok=True)

def predict_video(video_path):
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

def cut_clip(video_path, start_sec, end_sec, clip_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_no = start_frame
    while frame_no < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        frame_no += 1

    writer.release()
    cap.release()

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Long video not found: {VIDEO_PATH}")

cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frames / fps if fps > 0 else 0
cap.release()

print("Long video:", VIDEO_PATH)
print("Duration:", round(duration, 2), "seconds")

window_sec = 5
step_sec = 5

rows = []
start = 0.0
chunk_no = 1

while start < duration:
    end = min(start + window_sec, duration)

    if end - start < 1:
        break

    clip_path = CLIP_DIR / f"chunk_{chunk_no:03d}_{start:.1f}_{end:.1f}.mp4"

    print("=" * 80)
    print(f"Chunk {chunk_no}: {start:.2f}s to {end:.2f}s")

    cut_clip(VIDEO_PATH, start, end, clip_path)

    predicted_sentence, confidence = predict_video(clip_path)

    print("Predicted:", predicted_sentence)
    print("Confidence:", confidence)

    rows.append({
        "chunk_no": chunk_no,
        "start_sec": round(start, 2),
        "end_sec": round(end, 2),
        "clip_file": str(clip_path),
        "predicted_sentence": predicted_sentence,
        "confidence": confidence
    })

    start += step_sec
    chunk_no += 1

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

story_sentences = []
last_sentence = None

for sent in df["predicted_sentence"].tolist():
    sent = str(sent).strip()
    if sent and sent != last_sentence:
        story_sentences.append(sent)
        last_sentence = sent

story = "। ".join([s.rstrip("।") for s in story_sentences])
if story:
    story += "।"

OUT_TXT.write_text(story, encoding="utf-8")

print("\nSaved CSV:", OUT_CSV)
print("Saved story:", OUT_TXT)
print("\nPredicted story:")
print(story)
