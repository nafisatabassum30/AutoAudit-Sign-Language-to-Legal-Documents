import argparse
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vision.extract_keypoints import extract_video_keypoints


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
        return self.classifier(last)


def normalize_keypoints(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)

    return ((x - mean) / (std + 1e-6)).astype(np.float32)


def extract_live_exact(video_path, num_frames=32):
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        keypoints = extract_video_keypoints(
            video_path=Path(video_path),
            holistic=holistic,
            num_frames=num_frames
        )

    if keypoints is None:
        raise RuntimeError(f"Could not extract keypoints from: {video_path}")

    return keypoints


def load_saved_keypoints(video_path):
    stem = Path(video_path).stem.strip()
    kp_file = Path("data/keypoints_full") / f"{stem}.npy"

    if not kp_file.exists():
        raise FileNotFoundError(f"Saved keypoint not found: {kp_file}")

    print("Loading saved keypoints:", kp_file)
    return np.load(kp_file)


def load_model(model_dir, device):
    model_file = Path(model_dir) / "sign_sentence_lstm_model.pt"

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    checkpoint = torch.load(model_file, map_location=device)

    input_size = checkpoint.get("input_size", checkpoint.get("input_dim"))
    hidden_size = checkpoint.get("hidden_size", checkpoint.get("hidden_dim"))
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.4)

    if "label_to_id" in checkpoint:
        label_to_id = checkpoint["label_to_id"]
        id_to_label = {int(v): k for k, v in label_to_id.items()}
    elif "id_to_label" in checkpoint:
        id_to_label = {int(k): v for k, v in checkpoint["id_to_label"].items()}
    else:
        raise KeyError("No label_to_id or id_to_label found in checkpoint")

    num_classes = checkpoint.get("num_classes", len(id_to_label))

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
        results.append((id_to_label[int(idx)], float(prob)))

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

    if args.use_saved_keypoints:
        keypoints = load_saved_keypoints(args.video)
    else:
        print("Using EXACT training-time extractor from vision/extract_keypoints.py")
        keypoints = extract_live_exact(args.video, num_frames=32)

    model, id_to_label = load_model(args.model_dir, device)
    results = predict(model, keypoints, id_to_label, device, top_k=args.top_k)

    print("\nTop predictions:")
    for i, (sentence, confidence) in enumerate(results, start=1):
        print(f"{i}. {sentence} | confidence={confidence:.4f}")


if __name__ == "__main__":
    main()
