import os
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# CONFIG
# =========================
SEED = 42

TRAIN_CSV = "data/split_v2/train_manifest.csv"
VAL_CSV   = "data/split_v2/val_manifest.csv"
TEST_CSV  = "data/split_v2/test_manifest.csv"

KEYPOINT_DIR = "data/keypoints_full"

OUTPUT_DIR = "output/sign_sentence_v3_cnn_bilstm_attention"
RESULT_DIR = "outputs/sign_sentence_v3_cnn_bilstm_attention"

NUM_FRAMES = 32
INPUT_DIM = 258

BATCH_SIZE = 32
NUM_EPOCHS = 35
LR = 1e-3
WEIGHT_DECAY = 1e-4

CNN_CHANNELS = 256
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
ATTN_DIM = 128
DROPOUT = 0.45

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "sign_sentence_cnn_bilstm_attention_model.pt")


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# HELPER
# =========================
def clean_index(value):
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_manifest(path):
    df = pd.read_csv(path)
    required_cols = {"index", "sentence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df["index"] = df["index"].apply(clean_index)
    df["sentence"] = df["sentence"].astype(str).str.strip()
    return df


def normalize_keypoints(x):
    x = x.astype(np.float32)

    # Per-video normalization
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    return x.astype(np.float32)


def augment_keypoints(x):
    x = x.copy()

    # small Gaussian noise
    if random.random() < 0.60:
        noise = np.random.normal(0, 0.015, size=x.shape).astype(np.float32)
        x = x + noise

    # small feature scaling
    if random.random() < 0.45:
        scale = np.random.uniform(0.95, 1.05)
        x = x * scale

    # temporal shift without wrap
    if random.random() < 0.35:
        shift = random.choice([-2, -1, 1, 2])
        if shift > 0:
            pad = np.repeat(x[:1], shift, axis=0)
            x = np.concatenate([pad, x[:-shift]], axis=0)
        else:
            shift_abs = abs(shift)
            pad = np.repeat(x[-1:], shift_abs, axis=0)
            x = np.concatenate([x[shift_abs:], pad], axis=0)

    # random frame masking
    if random.random() < 0.35:
        n_mask = random.randint(1, 3)
        mask_ids = np.random.choice(x.shape[0], size=n_mask, replace=False)
        x[mask_ids] = 0.0

    return x.astype(np.float32)


# =========================
# DATASET
# =========================
class SignSentenceDataset(Dataset):
    def __init__(self, df, label_encoder, keypoint_dir, train=False):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.keypoint_dir = keypoint_dir
        self.train = train

        self.indices = self.df["index"].tolist()
        self.sentences = self.df["sentence"].tolist()
        self.labels = self.label_encoder.transform(self.sentences)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_index = self.indices[idx]
        npy_path = os.path.join(self.keypoint_dir, f"{video_index}.npy")

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Keypoint file missing: {npy_path}")

        x = np.load(npy_path).astype(np.float32)

        if x.shape != (NUM_FRAMES, INPUT_DIM):
            raise ValueError(f"Invalid shape for {npy_path}: {x.shape}, expected {(NUM_FRAMES, INPUT_DIM)}")

        x = normalize_keypoints(x)

        if self.train:
            x = augment_keypoints(x)

        y = int(self.labels[idx])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), video_index, self.sentences[idx]


# =========================
# MODEL
# =========================
class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attn_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, lstm_outputs):
        # lstm_outputs: [B, T, H]
        scores = self.attn(lstm_outputs).squeeze(-1)     # [B, T]
        weights = torch.softmax(scores, dim=1)           # [B, T]
        context = torch.sum(lstm_outputs * weights.unsqueeze(-1), dim=1)
        return context, weights


class CNNBiLSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        cnn_channels=256,
        lstm_hidden=256,
        lstm_layers=2,
        attn_dim=128,
        dropout=0.45
    ):
        super().__init__()

        # 1D CNN learns short-term temporal motion patterns from keypoint sequence
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # BiLSTM learns full temporal sequence in forward and backward directions
        self.bilstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        lstm_out_dim = lstm_hidden * 2

        # Attention focuses on important frames
        self.attention = TemporalAttention(lstm_out_dim, attn_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_attention=False):
        # x: [B, T, F]
        x = x.transpose(1, 2)          # [B, F, T]
        x = self.cnn(x)                # [B, C, T]
        x = x.transpose(1, 2)          # [B, T, C]

        lstm_out, _ = self.bilstm(x)   # [B, T, 2H]
        context, attn_weights = self.attention(lstm_out)

        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights

        return logits


# =========================
# TRAIN / EVAL
# =========================
def run_one_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_top5 = 0

    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            use_amp = device.type == "cuda"

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()

        k = min(5, logits.size(1))
        top5 = torch.topk(logits, k=k, dim=1).indices
        total_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

        total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    top5_acc = total_top5 / max(total_count, 1)

    return avg_loss, acc, top5_acc


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print("AutoAudit v3 Training: CNN + BiLSTM + Attention")
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

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["sentence"].tolist())

    train_classes = set(label_encoder.classes_)
    val_missing = sorted(set(val_df["sentence"].tolist()) - train_classes)
    test_missing = sorted(set(test_df["sentence"].tolist()) - train_classes)

    if val_missing or test_missing:
        print("ERROR: Some validation/test classes are missing from train.")
        print("Val missing:", len(val_missing))
        print("Test missing:", len(test_missing))
        raise SystemExit(1)

    sentence_classes = list(label_encoder.classes_)
    num_classes = len(sentence_classes)

    print("Sentence classes:", num_classes)

    save_json(os.path.join(OUTPUT_DIR, "sentence_classes.json"), sentence_classes)
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "sentence_label_encoder.pkl"))

    model_config = {
        "architecture": "CNN-BiLSTM-Attention",
        "input_dim": INPUT_DIM,
        "num_frames": NUM_FRAMES,
        "num_classes": num_classes,
        "cnn_channels": CNN_CHANNELS,
        "lstm_hidden": LSTM_HIDDEN,
        "lstm_layers": LSTM_LAYERS,
        "bidirectional": True,
        "attention_dim": ATTN_DIM,
        "dropout": DROPOUT,
        "note": "v3 model adapted from CNN feature + LSTM + temporal attention idea; input is MediaPipe keypoint sequence."
    }
    save_json(os.path.join(OUTPUT_DIR, "model_config.json"), model_config)

    train_dataset = SignSentenceDataset(train_df, label_encoder, KEYPOINT_DIR, train=True)
    val_dataset = SignSentenceDataset(val_df, label_encoder, KEYPOINT_DIR, train=False)
    test_dataset = SignSentenceDataset(test_df, label_encoder, KEYPOINT_DIR, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    model = CNNBiLSTMAttentionClassifier(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        cnn_channels=CNN_CHANNELS,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        attn_dim=ATTN_DIM,
        dropout=DROPOUT
    ).to(device)

    # Class-balanced loss: helpful because many sentence classes have very few videos
    train_labels = train_dataset.labels
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / np.sqrt(counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = []
    best_val_acc = 0.0
    best_epoch = 0

    start_time = time.time()

    print("\nTraining started...")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_top5 = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True, scaler=scaler
        )

        val_loss, val_acc, val_top5 = run_one_epoch(
            model, val_loader, criterion, optimizer, device, train=False, scaler=scaler
        )

        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_top5_accuracy": train_top5,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_top5_accuracy": val_top5
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train Top5: {train_top5:.4f} | Val Top5: {val_top5:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "label_classes": sentence_classes,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch
            }

            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"  ✅ Best model saved at epoch {epoch} with val acc {val_acc:.4f}")

        pd.DataFrame(history).to_csv(
            os.path.join(RESULT_DIR, "training_history.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print("-" * 70)
    print("Training finished.")
    print("Best epoch:", best_epoch)
    print("Best validation accuracy:", round(best_val_acc, 4))

    # Load best model for final test
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_top5 = run_one_epoch(
        model, test_loader, criterion, optimizer, device, train=False, scaler=scaler
    )

    elapsed_min = (time.time() - start_time) / 60.0

    final_metrics = {
        "architecture": "CNN-BiLSTM-Attention",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_top5_accuracy": test_top5,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_classes": num_classes,
        "elapsed_minutes": elapsed_min,
        "model_path": BEST_MODEL_PATH
    }

    save_json(os.path.join(RESULT_DIR, "final_metrics.json"), final_metrics)

    print("\nFinal Test Result")
    print("=" * 70)
    print("Test loss     :", round(test_loss, 4))
    print("Test accuracy :", round(test_acc, 4))
    print("Test top-5 acc:", round(test_top5, 4))
    print("Elapsed minute:", round(elapsed_min, 2))
    print("Saved model   :", BEST_MODEL_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
