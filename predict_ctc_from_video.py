import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from ctc_model import CTCSignModel, HIDDEN_SIZE, NUM_LSTM_LAYERS, CNN_LAYERS, DROPOUT


NUM_FRAMES = 32
FEATURE_DIM = 258


def load_vocab_from_project():
    candidates = [
        Path("output/word_to_id.json"),
        Path("word_to_id.json"),
    ]
    for word_path in candidates:
        if word_path.exists():
            with open(word_path, "r", encoding="utf-8") as f:
                word_to_id = json.load(f)
            break
    else:
        raise FileNotFoundError("word_to_id.json not found in project root or output/")

    id_candidates = [
        Path("output/id_to_word.json"),
        Path("id_to_word.json"),
    ]
    for id_path in id_candidates:
        if id_path.exists():
            with open(id_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            id_to_word = {int(k): v for k, v in raw.items()}
            return word_to_id, id_to_word

    raise FileNotFoundError("id_to_word.json not found in project root or output/")


def normalize_keypoints(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / (std + 1e-6)
    return x.astype(np.float32)


def extract_landmarks(results):
    pose = np.zeros(33 * 4, dtype=np.float32)
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.pose_landmarks:
        pose_data = []
        for lm in results.pose_landmarks.landmark:
            pose_data.extend([lm.x, lm.y, lm.z, lm.visibility])
        pose = np.array(pose_data, dtype=np.float32)

    if results.left_hand_landmarks:
        left_data = []
        for lm in results.left_hand_landmarks.landmark:
            left_data.extend([lm.x, lm.y, lm.z])
        left_hand = np.array(left_data, dtype=np.float32)

    if results.right_hand_landmarks:
        right_data = []
        for lm in results.right_hand_landmarks.landmark:
            right_data.extend([lm.x, lm.y, lm.z])
        right_hand = np.array(right_data, dtype=np.float32)

    return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)


def extract_keypoints_from_video(video_path, num_frames=NUM_FRAMES):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    mp_holistic = mp.solutions.holistic
    keypoints = []

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                keypoints.append(np.zeros(FEATURE_DIM, dtype=np.float32))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            kp = extract_landmarks(results)
            keypoints.append(kp)

    cap.release()
    keypoints = np.stack(keypoints).astype(np.float32)

    if keypoints.shape != (num_frames, FEATURE_DIM):
        raise RuntimeError(f"Unexpected keypoint shape: {keypoints.shape}")

    return keypoints


def ctc_best_path_decode(log_probs, input_length, id_to_word):
    predictions = torch.argmax(log_probs, dim=-1)
    seq = predictions[:input_length].tolist()

    collapsed = []
    prev = None
    for token_id in seq:
        if token_id != prev:
            collapsed.append(token_id)
        prev = token_id

    words = [id_to_word[int(t)] for t in collapsed if int(t) != 0]
    return words


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Model checkpoint at {model_path} is missing model_state_dict")

    vocab_size = checkpoint.get("vocab_size") or len(load_vocab_from_project()[0])
    model = CTCSignModel(
        input_size=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        vocab_size=vocab_size,
        cnn_layers=CNN_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_sentence_from_video(video_path, model_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    word_to_id, id_to_word = load_vocab_from_project()
    keypoints = extract_keypoints_from_video(video_path, num_frames=NUM_FRAMES)
    keypoints = normalize_keypoints(keypoints)

    x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
    model = load_model(model_path, device)

    with torch.no_grad():
        log_probs = model(x)

    # model returns [time, batch, vocab]
    time_len = log_probs.shape[0]
    decoded_words = ctc_best_path_decode(log_probs[:, 0, :], time_len, id_to_word)
    sentence = " ".join(decoded_words)
    return sentence, decoded_words


def main():
    parser = argparse.ArgumentParser(description="Run CTC sign sentence recognition directly from a video clip.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="output/ctc_sign_model.pt", help="Path to trained CTC checkpoint")
    parser.add_argument("--output_file", type=str, default=None, help="Optional text file to save the predicted sentence")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Video: {args.video}")
    sentence, words = predict_sentence_from_video(args.video, args.model_path, device=device)

    print("\nPredicted sentence:")
    print(sentence if sentence else "<empty>")

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(sentence + "\n", encoding="utf-8")
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
