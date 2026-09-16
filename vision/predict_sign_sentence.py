import argparse
from pathlib import Path

import joblib
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--keypoint_file", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    checkpoint_path = model_dir / "sign_sentence_lstm_model.pt"
    encoder_path = model_dir / "sentence_label_encoder.pkl"
    manifest_path = model_dir / "training_manifest.csv"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_encoder = joblib.load(encoder_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    x = np.load(args.keypoint_file).astype(np.float32)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_ids = torch.topk(probs, k=args.top_k, dim=1)

    print("\nTop predictions:")
    for rank, (prob, class_id) in enumerate(zip(top_probs[0], top_ids[0]), start=1):
        sentence = label_encoder.inverse_transform([class_id.item()])[0]
        print(f"{rank}. {sentence} | confidence={prob.item():.4f}")

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        actual = df[df["keypoint_path"] == args.keypoint_file.replace("\\", "/")]
        if len(actual) > 0:
            print("\nActual sentence:")
            print(actual.iloc[0]["sentence_clean"])


if __name__ == "__main__":
    main()