import argparse
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SignLSTMClassifier(nn.Module):
    def __init__(self, input_dim=258, hidden_dim=256, num_layers=2, num_classes=100, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits


def sample_frame_indices(total_frames, num_frames):
    if total_frames <= 0:
        return []

    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames).astype(int).tolist()

    base = list(range(total_frames))

    while len(base) < num_frames:
        base.append(total_frames - 1)

    return base[:num_frames]


def landmarks_to_array(landmarks, count, dims):
    if landmarks is None:
        return np.zeros(count * dims, dtype=np.float32)

    values = []

    for lm in landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])

        if dims == 4:
            values.append(lm.visibility)

    arr = np.array(values, dtype=np.float32)

    expected_size = count * dims

    if arr.shape[0] != expected_size:
        return np.zeros(expected_size, dtype=np.float32)

    return arr


def extract_video_keypoints(video_path, num_frames=32):
    mp_holistic = mp.solutions.holistic

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    frame_ids = sample_frame_indices(total_frames, num_frames)

    features = []
    last_feature = np.zeros(258, dtype=np.float32)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    ) as holistic:

        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ret, frame = cap.read()

            if not ret:
                features.append(last_feature.copy())
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(image)

            pose = landmarks_to_array(result.pose_landmarks, 33, 4)
            left_hand = landmarks_to_array(result.left_hand_landmarks, 21, 3)
            right_hand = landmarks_to_array(result.right_hand_landmarks, 21, 3)

            one_frame_feature = np.concatenate([pose, left_hand, right_hand]).astype(np.float32)

            if one_frame_feature.shape[0] != 258:
                one_frame_feature = np.zeros(258, dtype=np.float32)

            features.append(one_frame_feature)
            last_feature = one_frame_feature

    cap.release()

    while len(features) < num_frames:
        features.append(last_feature.copy())

    features = np.array(features[:num_frames], dtype=np.float32)

    if features.shape != (num_frames, 258):
        raise RuntimeError(
            f"Wrong feature shape: {features.shape}, expected ({num_frames}, 258)"
        )

    return features


def load_model(model_dir, device):
    model_dir = Path(model_dir)

    checkpoint_path = model_dir / "sign_sentence_lstm_model.pt"
    encoder_path = model_dir / "sentence_label_encoder.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    if not encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_encoder = joblib.load(encoder_path)

    model = SignLSTMClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=2,
        num_classes=checkpoint["num_classes"],
        dropout=0.3,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, label_encoder


def predict_sentence(model, label_encoder, features, device, top_k=5):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        top_k = min(top_k, probs.shape[1])
        top_probs, top_ids = torch.topk(probs, k=top_k, dim=1)

    predictions = []

    for prob, class_id in zip(top_probs[0], top_ids[0]):
        sentence = label_encoder.inverse_transform([class_id.item()])[0]
        predictions.append((sentence, prob.item()))

    return predictions


def print_actual_sentence(video_path, manifest_path):
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return

    try:
        video_index = int(Path(video_path).stem)
        df = pd.read_csv(manifest_path)

        actual = df[df["index"] == video_index]

        if len(actual) > 0:
            print("\nActual sentence:")
            print(str(actual.iloc[0]["sentence"]).strip())

    except Exception:
        pass


def load_or_extract_features(video_path, num_frames=32):
    saved_keypoint_path = Path("data/keypoints_full") / f"{video_path.stem}.npy"

    if saved_keypoint_path.exists():
        print(f"Loading saved keypoints: {saved_keypoint_path}")
        features = np.load(saved_keypoint_path).astype(np.float32)
    else:
        print("Saved keypoints not found.")
        print("Extracting keypoints from video...")
        features = extract_video_keypoints(video_path, num_frames=num_frames)

    if features.shape != (num_frames, 258):
        raise RuntimeError(
            f"Wrong feature shape: {features.shape}, expected ({num_frames}, 258)"
        )

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/keypoints_full/keypoint_manifest.csv",
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    features = load_or_extract_features(
        video_path=video_path,
        num_frames=args.num_frames,
    )

    model, label_encoder = load_model(args.model_dir, device)

    predictions = predict_sentence(
        model=model,
        label_encoder=label_encoder,
        features=features,
        device=device,
        top_k=args.top_k,
    )

    print("\nTop predictions:")

    for rank, (sentence, confidence) in enumerate(predictions, start=1):
        print(f"{rank}. {sentence} | confidence={confidence:.4f}")

    print_actual_sentence(video_path, args.manifest)


if __name__ == "__main__":
    main()