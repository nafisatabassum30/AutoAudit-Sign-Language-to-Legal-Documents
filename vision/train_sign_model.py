import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SignKeypointDataset(Dataset):
    def __init__(self, manifest_csv, label_encoder=None, max_samples=None):
        self.df = pd.read_csv(manifest_csv)

        if max_samples is not None:
            self.df = self.df.head(max_samples).copy()

        self.df["index"] = self.df["index"].astype(int)
        self.sentences = self.df["sentence"].astype(str).tolist()

        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.df["index"].astype(str))
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(self.df["index"].astype(str))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        keypoint_path = row["keypoint_path"]

        x = np.load(keypoint_path).astype(np.float32)

        # Shape: 32 frames × 258 features
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


class SignLSTMClassifier(nn.Module):
    def __init__(self, input_dim=258, hidden_dim=256, num_layers=2, num_classes=5010, dropout=0.3):
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
        # x shape: batch × frames × features
        out, _ = self.lstm(x)

        # Use last frame representation
        last = out[:, -1, :]

        logits = self.classifier(last)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    progress = tqdm(loader, desc="Training", leave=False)

    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total_items += x.size(0)

        progress.set_postfix({
            "loss": total_loss / max(total_items, 1),
            "acc": total_correct / max(total_items, 1),
        })

    avg_loss = total_loss / total_items
    avg_acc = total_correct / total_items

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/keypoints_full/keypoint_manifest.csv")
    parser.add_argument("--output_dir", type=str, default="output/sign_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SignKeypointDataset(
        manifest_csv=args.manifest,
        max_samples=args.max_samples,
    )

    num_classes = len(dataset.label_encoder.classes_)
    print("Samples:", len(dataset))
    print("Classes:", num_classes)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = SignLSTMClassifier(
        input_dim=258,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} | train_acc={acc:.4f}")

        history.append({
            "epoch": epoch,
            "loss": loss,
            "train_acc": acc,
        })

        if loss < best_loss:
            best_loss = loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": 258,
                    "hidden_dim": args.hidden_dim,
                    "num_classes": num_classes,
                },
                output_dir / "sign_lstm_model.pt",
            )

            joblib.dump(dataset.label_encoder, output_dir / "label_encoder.pkl")

            dataset.df[["index", "sentence", "keypoint_path"]].to_csv(
                output_dir / "training_manifest.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print("Saved best model.")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("Training complete.")
    print("Model saved to:", output_dir / "sign_lstm_model.pt")
    print("Label encoder saved to:", output_dir / "label_encoder.pkl")


if __name__ == "__main__":
    main()