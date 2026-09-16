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

OUTPUT_DIR = "output/sign_sentence_v3_hybrid_cnn_lstm_attention"
RESULT_DIR = "outputs/sign_sentence_v3_hybrid_cnn_lstm_attention"
MODEL_PATH = os.path.join(OUTPUT_DIR, "sign_sentence_v3_hybrid_model.pt")

NUM_FRAMES = 32
INPUT_DIM = 258

BATCH_SIZE = 32
NUM_EPOCHS = 35
LR = 0.001
WEIGHT_DECAY = 1e-4

CNN_DIM = 256
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTN_DIM = 128
DROPOUT = 0.35


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


def augment_keypoints(x):
    x = x.copy()

    # light noise for generalization
    if random.random() < 0.35:
        x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)

    # light scaling
    if random.random() < 0.25:
        scale = np.random.uniform(0.97, 1.03)
        x = x * scale

    # small temporal shift
    if random.random() < 0.25:
        shift = random.choice([-1, 1])
        if shift > 0:
            x = np.concatenate([x[:1], x[:-1]], axis=0)
        else:
            x = np.concatenate([x[1:], x[-1:]], axis=0)

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
            raise ValueError(f"Wrong keypoint shape for {npy_path}: {x.shape}")

        x = normalize_keypoints(x)

        if self.train:
            x = augment_keypoints(x)

        y = int(self.labels[idx])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class HybridCNNEncoder(nn.Module):
    """
    Dhrubo idea adapted:
    - CNN feature extraction idea
    - Encoder/projection idea

    But for our project:
    - Input is MediaPipe keypoint sequence, not raw RGB frame.
    - So we use 1D CNN over time + raw projection residual path.
    """
    def __init__(self, input_dim=258, cnn_dim=256, dropout=0.35):
        super().__init__()

        # v2-like raw projection path so original keypoint information is not lost
        self.raw_proj = nn.Linear(input_dim, cnn_dim)

        # small temporal CNN path to learn local motion pattern
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(cnn_dim)

    def forward(self, x):
        # x: [B, T, F]
        raw_feat = self.raw_proj(x)          # [B, T, C]

        cnn_feat = x.transpose(1, 2)         # [B, F, T]
        cnn_feat = self.cnn(cnn_feat)        # [B, C, T]
        cnn_feat = cnn_feat.transpose(1, 2)  # [B, T, C]

        fused = raw_feat + cnn_feat          # residual fusion
        fused = self.norm(fused)

        return fused


class TemporalAttention(nn.Module):
    """
    Dhrubo soft temporal attention idea adapted for classification.
    Instead of generating word-by-word caption, we use attention pooling over frames.
    """
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


class HybridCNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.encoder = HybridCNNEncoder(
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
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0
        )

        lstm_out_dim = LSTM_HIDDEN * 2
        self.attention = TemporalAttention(lstm_out_dim, ATTN_DIM)

        # combine attention context with final timestep summaries
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_attention=False):
        x = self.encoder(x)              # [B, T, C]
        lstm_out, _ = self.bilstm(x)     # [B, T, 2H]
        context, attn_weights = self.attention(lstm_out)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
    print("AutoAudit v3 HYBRID: 1D CNN + BiLSTM + Attention")
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
        raise RuntimeError("Validation/Test contains class not present in training.")

    sentence_classes = list(le.classes_)
    num_classes = len(sentence_classes)

    print("Sentence classes:", num_classes)

    model_config = {
        "architecture": "Hybrid-1D-CNN-BiLSTM-Attention",
        "input_dim": int(INPUT_DIM),
        "num_frames": int(NUM_FRAMES),
        "num_classes": int(num_classes),
        "cnn_dim": int(CNN_DIM),
        "lstm_hidden": int(LSTM_HIDDEN),
        "lstm_layers": int(LSTM_LAYERS),
        "bidirectional": True,
        "attention_dim": int(ATTN_DIM),
        "dropout": float(DROPOUT),
        "description": "Dhrubo-inspired CNN feature extraction + temporal attention adapted to MediaPipe keypoint sentence classification."
    }

    save_json(os.path.join(OUTPUT_DIR, "model_config.json"), model_config)
    save_json(os.path.join(OUTPUT_DIR, "sentence_classes.json"), sentence_classes)
    joblib.dump(le, os.path.join(OUTPUT_DIR, "sentence_label_encoder.pkl"))

    train_ds = SignDataset(train_df, le, train=True)
    val_ds = SignDataset(val_df, le, train=False)
    test_ds = SignDataset(test_df, le, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = HybridCNNBiLSTMAttention(INPUT_DIM, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=6
    )

    history = []
    best_val_acc = -1.0
    best_epoch = 0
    best_train_acc_at_val = 0.0

    start_time = time.time()

    print("\nTraining started...")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_m["accuracy"])

        row = {
            "epoch": int(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "train_top5_accuracy": train_m["top5_accuracy"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_top5_accuracy": val_m["top5_accuracy"]
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Acc: {train_m['accuracy']:.4f} | "
            f"Val Acc: {val_m['accuracy']:.4f} | "
            f"Train Top5: {train_m['top5_accuracy']:.4f} | "
            f"Val Top5: {val_m['top5_accuracy']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        pd.DataFrame(history).to_csv(
            os.path.join(RESULT_DIR, "training_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        # validation-only checkpointing to avoid memorization-based selection
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch = epoch
            best_train_acc_at_val = train_m["accuracy"]

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "label_classes": sentence_classes,
                "best_epoch": int(best_epoch),
                "best_val_acc": float(best_val_acc),
                "train_acc_at_best_val": float(best_train_acc_at_val)
            }

            torch.save(checkpoint, MODEL_PATH)
            print("  ✅ Best validation checkpoint saved.")

        if epoch == 10 and train_m["accuracy"] < 0.01:
            print("\nWARNING: Train accuracy is still low after 10 epochs.")
            print("This may indicate that full 3395-class sparse training is difficult.")
            print("But checkpointing remains validation-only.")

    print("-" * 70)
    print("Training finished.")
    print("Best epoch:", best_epoch)
    print("Best val accuracy:", round(best_val_acc, 4))
    print("Train acc at best val:", round(best_train_acc_at_val, 4))

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_m = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    final_metrics = {
        "architecture": "Hybrid-1D-CNN-BiLSTM-Attention",
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "train_acc_at_best_val": float(best_train_acc_at_val),
        "test_loss": float(test_m["loss"]),
        "test_accuracy": float(test_m["accuracy"]),
        "test_top5_accuracy": float(test_m["top5_accuracy"]),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "num_classes": int(num_classes),
        "elapsed_minutes": float((time.time() - start_time) / 60),
        "model_path": MODEL_PATH
    }

    save_json(os.path.join(RESULT_DIR, "final_metrics.json"), final_metrics)

    print("\nFinal Test Result")
    print("=" * 70)
    print("Test loss     :", round(test_m["loss"], 4))
    print("Test accuracy :", round(test_m["accuracy"], 4))
    print("Test top-5 acc:", round(test_m["top5_accuracy"], 4))
    print("Saved model   :", MODEL_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
