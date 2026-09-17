import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.cnn1 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.cnn_activation = nn.ReLU()
        self.cnn_dropout = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        lstm_out_size = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.attn_gate = nn.Parameter(torch.tensor(-2.0))
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        cnn_out = self.cnn1(x.transpose(1, 2))
        cnn_out = self.cnn_activation(cnn_out)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = self.cnn2(cnn_out).transpose(1, 2)
        out, _ = self.lstm(x + cnn_out)
        last = out[:, -1, :]
        attention_scores = self.attention(out).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(out * attention_weights.unsqueeze(-1), dim=1)
        gate = torch.sigmoid(self.attn_gate)
        fused = (1.0 - gate) * last + gate * context
        return self.classifier(fused)


def normalize_keypoints(keypoints):
    keypoints = np.nan_to_num(
        keypoints.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    mean = keypoints.mean(axis=0, keepdims=True)
    std = keypoints.std(axis=0, keepdims=True)
    return ((keypoints - mean) / (std + 1e-6)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_path = next(model_dir.glob("*_model.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SignLSTM(
        checkpoint["input_size"],
        checkpoint["hidden_size"],
        checkpoint["num_layers"],
        checkpoint["num_classes"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    keypoints = normalize_keypoints(np.load(args.keypoints))
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(keypoints).unsqueeze(0)), dim=1
        )[0]
        values, indices = torch.topk(probabilities, k=args.top_k)

    with open(model_dir / "sentence_classes.json", encoding="utf-8") as file:
        classes = json.load(file)

    print(f"checkpoint={checkpoint_path}")
    print(f"keypoints={args.keypoints}")
    for rank, (value, index) in enumerate(zip(values, indices), 1):
        print(f"{rank}. {classes[int(index)]} | confidence={float(value):.4f}")


if __name__ == "__main__":
    main()
