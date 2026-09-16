import os
import json
import random
import time
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# =========================
# CONFIG
# =========================
SEED = 42

TRAIN_CSV = "data/split_v2/train_manifest.csv"
VAL_CSV   = "data/split_v2/val_manifest.csv"
TEST_CSV  = "data/split_v2/test_manifest.csv"
KEYPOINT_DIR = "data/keypoints_full"

OUTPUT_DIR = "output/sign_sentence_v3_safe_validation_only"
RESULT_DIR = "outputs/sign_sentence_v3_safe_validation_only"

MODEL_PATH = os.path.join(OUTPUT_DIR, "sign_sentence_v3_safe_model.pt")

NUM_FRAMES = 32
INPUT_DIM = 258

BATCH_SIZE = 32
NUM_EPOCHS = 80
LR = 0.002
WEIGHT_DECAY = 1e-4

CNN_DIM = 256
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTN_DIM = 128
DROPOUT = 0.20


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


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifest(path):
    df = pd.read_csv(path)
    df["index"] = df["index"].apply(clean_index)
    df["sentence"] = df["sentence"].astype(str).str.strip()
    return df


def normalize_keypoints(x):
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def light_augment(x):
    x = x.copy()

    if random.random() < 0.30:
        x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)

    if random.random() < 0.20:
        scale = np.random.uniform(0.97, 1.03)
        x = x * scale

    return x.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, df, label_encoder, train=False):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.train = train

        self.indices = self.df["index"].tolist()
        self.sentences = self.df["sentence"].tolist()
        self.labels = self.le.transform(self.sentences)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_index = self.indices[idx]
        npy_path = os.path.join(KEYPOINT_DIR, f"{video_index}.npy")

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing keypoint file: {npy_path}")

        x = np.load(npy_path).astype(np.float32)

        if x.shape != (NUM_FRAMES, INPUT_DIM):
            raise ValueError(f"Wrong keypoint shape: {npy_path} has {x.shape}")

        x = normalize_keypoints(x)

        if self.train:
            x = light_augment(x)

        y = int(self.labels[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, H]
        scores = self.attn(x).squeeze(-1)       # [B, T]
        weights = torch.softmax(scores, dim=1)  # [B, T]
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context, weights


class ResidualTemporalCNN(nn.Module):
    def __init__(self, input_dim=258, cnn_dim=256, dropout=0.20):
        super().__init__()

        self.input_proj = nn.Conv1d(input_dim, cnn_dim, kernel_size=1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2)      # [B, F, T]
        x = self.input_proj(x)     # [B, C, T]

        residual = x
        x = self.conv1(x)
        x = x + residual

        residual = x
        x = self.conv2(x)
        x = x + residual

        x = x.transpose(1, 2)      # [B, T, C]
        return x


class CNNBiLSTMAttentionFinal(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.cnn = ResidualTemporalCNN(
            input_dim=input_dim,
            cnn_dim=CNN_DIM,
            dropout=DROPOUT
        )

        self.bilstm = nn.LSTM(
            input_size=CNN_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )

        lstm_out_dim = LSTM_HIDDEN * 2

        self.attention = TemporalAttention(lstm_out_dim, ATTN_DIM)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.cnn(x)
        x, _ = self.bilstm(x)
        context, attn_weights = self.attention(x)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights

        return logits


def run_epoch(model, loader, criterion, optimizer, device, train=False):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs

        pred = torch.argmax(logits, dim=1)
        total_correct += (pred == y).sum().item()

        k = min(5, logits.size(1))
        top5 = torch.topk(logits, k=k, dim=1).indices
        total_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

        total += bs

    return {
        "loss": float(total_loss / max(total, 1)),
        "accuracy": float(total_correct / max(total, 1)),
        "top5_accuracy": float(total_top5 / max(total, 1))
    }


def main():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("AutoAudit v3 SAFE: CNN + BiLSTM + Attention, validation-only checkpoint")
    print("=" * 70)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_df = load_manifest(TRAIN_CSV)
    val_df = load_manifest(VAL_CSV)
    test_df = load_manifest(TEST_CSV)

    print("\nDataset size:")
    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df))

    le = LabelEncoder()
    le.fit(train_df["sentence"].tolist())

    train_classes = set(le.classes_)
    val_missing = set(val_df["sentence"].tolist()) - train_classes
    test_missing = set(test_df["sentence"].tolist()) - train_classes

    if val_missing or test_missing:
        raise RuntimeError("Val/Test has unseen class not present in train.")

    sentence_classes = list(le.classes_)
    num_classes = len(sentence_classes)

    print("Sentence classes:", num_classes)

    save_json(os.path.join(OUTPUT_DIR, "sentence_classes.json"), sentence_classes)
    joblib.dump(le, os.path.join(OUTPUT_DIR, "sentence_label_encoder.pkl"))

    model_config = {
        "architecture": "CNN-BiLSTM-Attention-v3-safe-validation-only",
        "input_dim": int(INPUT_DIM),
        "num_frames": int(NUM_FRAMES),
        "num_classes": int(num_classes),
        "cnn_dim": int(CNN_DIM),
        "lstm_hidden": int(LSTM_HIDDEN),
        "lstm_layers": int(LSTM_LAYERS),
        "bidirectional": True,
        "attention_dim": int(ATTN_DIM),
        "dropout": float(DROPOUT),
        "description": "MediaPipe keypoints -> Residual 1D CNN -> BiLSTM -> Temporal Attention -> Bangla sentence class"
    }

    save_json(os.path.join(OUTPUT_DIR, "model_config.json"), model_config)

    train_ds = SignDataset(train_df, le, train=True)
    val_ds = SignDataset(val_df, le, train=False)
    test_ds = SignDataset(test_df, le, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CNNBiLSTMAttentionFinal(INPUT_DIM, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc = -1.0
    best_train_acc = -1.0
    best_epoch = 0
    history = []

    start = time.time()

    print("\nTraining started...")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        row = {
            "epoch": int(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_top5_accuracy": train_metrics["top5_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "lr": float(optimizer.param_groups[0]["lr"])
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Train Top5: {train_metrics['top5_accuracy']:.4f} | "
            f"Val Top5: {val_metrics['top5_accuracy']:.4f} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )

        pd.DataFrame(history).to_csv(
            os.path.join(RESULT_DIR, "training_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        should_save = False

        # SAFE checkpointing:
        # Save model only when validation accuracy improves.
        # Do not save based on training accuracy, because that can overfit/memorize.
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_train_acc = train_metrics["accuracy"]
            best_epoch = epoch
            should_save = True

        if should_save:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "label_classes": sentence_classes,
                "best_epoch": int(best_epoch),
                "best_val_acc": float(best_val_acc),
                "best_train_acc": float(best_train_acc)
            }
            torch.save(checkpoint, MODEL_PATH)
            print("  ✅ Model checkpoint saved.")

        if epoch == 10 and train_metrics["accuracy"] < 0.01:
            print("\nWARNING: Train accuracy is still very low after 10 epochs.")
            print("This means full 3395-class training is difficult due to sparse class samples.")
            print("Training will continue, but this is mainly architecture/demo evidence.")

    print("-" * 70)
    print("Training finished.")
    print("Best epoch:", best_epoch)
    print("Best val acc:", round(best_val_acc, 4))
    print("Train acc at best validation epoch:", round(best_train_acc, 4))

    if not os.path.exists(MODEL_PATH):
        print("No checkpoint found, saving final model.")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "label_classes": sentence_classes,
            "best_epoch": int(NUM_EPOCHS),
            "best_val_acc": float(best_val_acc),
            "best_train_acc": float(best_train_acc)
        }
        torch.save(checkpoint, MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    final_metrics = {
        "architecture": "CNN-BiLSTM-Attention-v3-safe-validation-only",
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "best_train_accuracy": float(best_train_acc),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_top5_accuracy": float(test_metrics["top5_accuracy"]),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "num_classes": int(num_classes),
        "elapsed_minutes": float((time.time() - start) / 60),
        "model_path": MODEL_PATH
    }

    save_json(os.path.join(RESULT_DIR, "final_metrics.json"), final_metrics)

    print("\nFinal Test Result")
    print("=" * 70)
    print("Test loss     :", round(test_metrics["loss"], 4))
    print("Test accuracy :", round(test_metrics["accuracy"], 4))
    print("Test top-5 acc:", round(test_metrics["top5_accuracy"], 4))
    print("Saved model   :", MODEL_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()

