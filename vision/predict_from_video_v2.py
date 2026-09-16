import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


NUM_FRAMES = 32


class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.classifier(last)
        return logits


def clean_index_from_video(video_path):
    return Path(video_path).stem.strip()


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
        lh_data = []
        for lm in results.left_hand_landmarks.landmark:
            lh_data.extend([lm.x, lm.y, lm.z])
        left_hand = np.array(lh_data, dtype=np.float32)

    if results.right_hand_landmarks:
        rh_data = []
        for lm in results.right_hand_landmarks.landmark:
            rh_data.extend([lm.x, lm.y, lm.z])
        right_hand = np.array(rh_data, dtype=np.float32)

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
        min_tracking_confidence=0.5
    ) as holistic:

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                keypoints.append(np.zeros(258, dtype=np.float32))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            kp = extract_landmarks(results)
            keypoints.append(kp)

    cap.release()

    keypoints = np.stack(keypoints).astype(np.float32)

    if keypoints.shape != (num_frames, 258):
        raise RuntimeError(f"Unexpected keypoint shape: {keypoints.shape}")

    return keypoints


def load_keypoints(video_path, use_saved_keypoints=False):
    video_path = Path(video_path)

    if use_saved_keypoints:
        stem = clean_index_from_video(video_path)
        saved_path = Path("data/keypoints_full") / f"{stem}.npy"

        if saved_path.exists():
            print("Loading saved keypoints:", saved_path)
            return np.load(saved_path)

        print("Saved keypoints not found. Extracting live...")

    else:
        print("Using live extraction from actual video.")
        print("This avoids wrong old keypoint loading for external videos.")

    return extract_keypoints_from_video(video_path)


def load_model(model_dir, device):
    model_dir = Path(model_dir)
    model_file = model_dir / "sign_sentence_lstm_model.pt"

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device)

    input_size = checkpoint.get("input_size", checkpoint.get("input_dim"))
    hidden_size = checkpoint.get("hidden_size", checkpoint.get("hidden_dim"))
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.4)
    num_classes = checkpoint.get("num_classes")

    if "label_to_id" in checkpoint:
        label_to_id = checkpoint["label_to_id"]
        id_to_label = {int(v): k for k, v in label_to_id.items()}
    elif "id_to_label" in checkpoint:
        id_to_label = {int(k): v for k, v in checkpoint["id_to_label"].items()}
    else:
        raise KeyError("No label_to_id or id_to_label found in checkpoint")

    if num_classes is None:
        num_classes = len(id_to_label)

    model = SignLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, id_to_label


def predict(model, keypoints, id_to_label, device, top_k=5):
    keypoints = normalize_keypoints(keypoints)

    x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_ids = torch.topk(probs, k=top_k)

    results = []

    for prob, idx in zip(top_probs.cpu().numpy(), top_ids.cpu().numpy()):
        sentence = id_to_label[int(idx)]
        results.append((sentence, float(prob)))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_saved_keypoints", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    keypoints = load_keypoints(
        args.video,
        use_saved_keypoints=args.use_saved_keypoints
    )

    model, id_to_label = load_model(args.model_dir, device)

    results = predict(
        model=model,
        keypoints=keypoints,
        id_to_label=id_to_label,
        device=device,
        top_k=args.top_k
    )

    print("\nTop predictions:")
    for i, (sentence, confidence) in enumerate(results, start=1):
        print(f"{i}. {sentence} | confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
