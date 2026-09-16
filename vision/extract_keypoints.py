import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21

# pose: 33 landmarks × 4 values = 132
# left hand: 21 landmarks × 3 values = 63
# right hand: 21 landmarks × 3 values = 63
# total per frame = 258
FEATURE_DIM = (POSE_LANDMARKS * 4) + (HAND_LANDMARKS * 3) + (HAND_LANDMARKS * 3)


def flatten_landmarks(landmarks, count, values_per_landmark):
    if landmarks is None:
        return np.zeros(count * values_per_landmark, dtype=np.float32)

    output = []
    for lm in landmarks.landmark:
        output.extend([lm.x, lm.y, lm.z])
        if values_per_landmark == 4:
            output.append(lm.visibility)

    return np.array(output, dtype=np.float32)


def sample_frame_indices(total_frames, num_frames):
    if total_frames <= 0:
        return []

    if total_frames <= num_frames:
        return list(range(total_frames))

    return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()


def extract_video_keypoints(video_path, holistic, num_frames=32):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_frames = set(sample_frame_indices(total_frames, num_frames))

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id in selected_frames:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            pose = flatten_landmarks(results.pose_landmarks, POSE_LANDMARKS, 4)
            left_hand = flatten_landmarks(results.left_hand_landmarks, HAND_LANDMARKS, 3)
            right_hand = flatten_landmarks(results.right_hand_landmarks, HAND_LANDMARKS, 3)

            feature = np.concatenate([pose, left_hand, right_hand])

            if feature.shape[0] != FEATURE_DIM:
                raise ValueError(f"Feature dim mismatch: got {feature.shape[0]}, expected {FEATURE_DIM}")

            frames.append(feature)

        frame_id += 1

    cap.release()

    if len(frames) == 0:
        return None

    # If video has fewer sampled frames, pad with zeros
    while len(frames) < num_frames:
        frames.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    # If somehow more, cut
    frames = frames[:num_frames]

    return np.stack(frames).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/train/video_sentence_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="data/keypoints")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    if args.max_videos is not None:
        df = df.head(args.max_videos)

    mp_holistic = mp.solutions.holistic

    records = []
    failed = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keypoints"):
            index = int(row["index"])
            sentence = str(row["sentence"])
            video_path = Path(row["video_path"])

            output_file = output_dir / f"{index}.npy"

            if output_file.exists():
                records.append({
                    "index": index,
                    "sentence": sentence,
                    "keypoint_path": str(output_file).replace("\\", "/"),
                })
                continue

            keypoints = extract_video_keypoints(video_path, holistic, num_frames=args.num_frames)

            if keypoints is None:
                failed.append(index)
                continue

            np.save(output_file, keypoints)

            records.append({
                "index": index,
                "sentence": sentence,
                "keypoint_path": str(output_file).replace("\\", "/"),
            })

    out_manifest = output_dir / "keypoint_manifest.csv"
    pd.DataFrame(records).to_csv(out_manifest, index=False, encoding="utf-8-sig")

    failed_file = output_dir / "failed_videos.txt"
    failed_file.write_text("\n".join(map(str, failed)), encoding="utf-8")

    print("Saved keypoint manifest:", out_manifest)
    print("Successful videos:", len(records))
    print("Failed videos:", len(failed))
    print("Failed list saved:", failed_file)


if __name__ == "__main__":
    main()