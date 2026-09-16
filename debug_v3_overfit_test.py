import os
import json
import random
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


SEED = 42
TRAIN_CSV = "data/split_v2/train_manifest.csv"
KEYPOINT_DIR = "data/keypoints_full"

NUM_FRAMES = 32
INPUT_DIM = 258
DEBUG_SAMPLES = 80
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3

OUT_DIR = "outputs/debug_v3_overfit"
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


def clean_index(value):
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def normalize_keypoints(x):
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


class DebugDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.labels = self.le.transform(self.df["sentence"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_index = clean_index(row["index"])
        npy_path = os.path.join(KEYPOINT_DIR, f"{video_index}.npy")

        x = np.load(npy_path).astype(np.float32)
        x = normalize_keypoints(x)

        y = int(self.labels[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TemporalAttention(nn.Module):
    def __init__(self, dim, attn_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x):
        scores = self.net(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context


class SmallCNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = TemporalAttention(256, 128)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)   # [B, F, T]
        x = self.cnn(x)         # [B, C, T]
        x = x.transpose(1, 2)   # [B, T, C]
        x, _ = self.lstm(x)     # [B, T, 256]
        x = self.attn(x)        # [B, 256]
        return self.fc(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(TRAIN_CSV)
    df["index"] = df["index"].apply(clean_index)
    df["sentence"] = df["sentence"].astype(str).str.strip()

    # Take only first DEBUG_SAMPLES, but keep unique classes
    df = df.head(DEBUG_SAMPLES).copy()

    le = LabelEncoder()
    le.fit(df["sentence"].tolist())

    dataset = DebugDataset(df, le)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SmallCNNBiLSTMAttention(INPUT_DIM, len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Debug samples:", len(df))
    print("Debug classes:", len(le.classes_))
    print("Goal: train accuracy should become very high. If not, model/code has issue.")
    print("-" * 70)

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total
        avg_loss = total_loss / total

        history.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {acc:.4f}")

        if acc >= 0.95:
            print("✅ Overfit success. Model can learn small set.")
            break

    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "debug_overfit_history.csv"), index=False)

    print("-" * 70)
    print("Final debug accuracy:", round(history[-1]["accuracy"], 4))
    print("Saved:", os.path.join(OUT_DIR, "debug_overfit_history.csv"))


if __name__ == "__main__":
    main()
