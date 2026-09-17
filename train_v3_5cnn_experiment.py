"""
train_v3_5cnn_experiment.py

Direct empirical test: does adding CNN depth to the existing v3-continue
model (33-35% test accuracy) help or hurt?

This is the SAME sentence-classification setup as v3/v3-continue
(same split_v2, same dataset class, same LSTM+attention+classifier head)
with ONE change: the CNN front-end goes from 1 residual block (2 conv
layers) to 5 stacked residual blocks (10 conv layers total).

It loads the existing v3-continue checkpoint's LSTM/attention/classifier
weights directly (so training continues from where that model left off,
not from scratch), and only the new CNN blocks start fresh -- zero-init
per block, same safety trick as before, so at epoch 0 this model behaves
almost identically to the checkpoint it's extending.

Run from the project root:
    python v3_5cnn_experiment/train_v3_5cnn_experiment.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SPLIT_DIR = Path("data/split_v2")  # SAME split as original v3/v3-continue -- required for a fair, direct comparison
KEYPOINT_DIR = Path("data/keypoints_full")
OUTPUT_DIR = Path("v3_10cnn_experiment/output")
RESULT_DIR = Path("v3_10cnn_experiment/outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train_manifest.csv"
VAL_FILE = SPLIT_DIR / "val_manifest.csv"
TEST_FILE = SPLIT_DIR / "test_manifest.csv"

# Point this at wherever you moved the v3-continue checkpoint during cleanup.
LOAD_MODEL_PATH = Path(
    "archive_failed_v3_experiments/output/sign_sentence_v3_v2plus_cnn_attention_continue/"
    "sign_sentence_v3_v2plus_cnn_attention_continue_model.pt"
)

NUM_EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.0003  # same as v3-continue, since we're continuing training, not starting fresh
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.4
CNN_BLOCKS = 10  # <-- the variable being tested
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def clean_index(x):
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_keypoint_file(index_value):
    idx = clean_index(index_value)
    candidates = [
        KEYPOINT_DIR / f"{idx}.npy",
        KEYPOINT_DIR / f"keypoints_{idx}.npy",
        KEYPOINT_DIR / "keypoints" / f"{idx}.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(KEYPOINT_DIR.rglob(f"{idx}.npy"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Keypoint file not found for index: {idx}")


def normalize_keypoints(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / (std + 1e-6)
    return x.astype(np.float32)


def augment_keypoints(x):
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.95, 1.05)
        x = x * scale
    if np.random.rand() < 0.3:
        t = np.random.randint(0, x.shape[0])
        x[t] = 0
    return x.astype(np.float32)


class SignDataset(Dataset):
    def __init__(self, manifest_file, label_to_id, is_train=False):
        self.df = pd.read_csv(manifest_file, encoding="utf-8-sig")
        self.df["sentence_clean"] = self.df["sentence"].astype(str).str.strip()
        self.label_to_id = label_to_id
        self.is_train = is_train

        self.items = []
        missing = 0

        for _, row in self.df.iterrows():
            sentence = row["sentence_clean"]
            index_value = row["index"]

            try:
                kp_file = find_keypoint_file(index_value)
            except FileNotFoundError:
                missing += 1
                continue

            if sentence not in label_to_id:
                continue

            self.items.append({
                "keypoint_file": kp_file,
                "label": label_to_id[sentence],
                "sentence": sentence,
                "index": clean_index(index_value),
            })

        if missing > 0:
            print(f"Warning: missing keypoint files: {missing}")
        if len(self.items) == 0:
            raise RuntimeError(f"No usable items found from {manifest_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        x = np.load(item["keypoint_file"])
        x = normalize_keypoints(x)
        if self.is_train:
            x = augment_keypoints(x)
        y = item["label"]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class ResidualCNNBlockV3(nn.Module):
    """
    Same block design as the original v3 CNN (conv -> relu -> dropout -> conv,
    zero-init the second conv so the block starts as a near-identity function).
    Stacking N of these is the "add N CNN layers" experiment.
    """
    def __init__(self, dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout * 0.5)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        # x: [batch, features, time]
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return residual + out


class SignLSTM_5CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout, cnn_blocks=CNN_BLOCKS):
        super().__init__()

        self.cnn_blocks = nn.ModuleList([
            ResidualCNNBlockV3(input_size, dropout) for _ in range(cnn_blocks)
        ])

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
        # x: [batch, time, features]
        cnn_out = x.transpose(1, 2)  # -> [batch, features, time]

        for block in self.cnn_blocks:
            cnn_out = block(cnn_out)

        cnn_out = cnn_out.transpose(1, 2)  # -> [batch, time, features]

        out, _ = self.lstm(cnn_out)

        last = out[:, -1, :]

        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

        gate = torch.sigmoid(self.attn_gate)
        fused = (1.0 - gate) * last + gate * context

        logits = self.classifier(fused)
        return logits


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    train_df["sentence_clean"] = train_df["sentence"].astype(str).str.strip()

    classes = sorted(train_df["sentence_clean"].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(classes)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("Number of classes:", len(classes))

    train_dataset = SignDataset(TRAIN_FILE, label_to_id, is_train=True)
    val_dataset = SignDataset(VAL_FILE, label_to_id, is_train=False)
    test_dataset = SignDataset(TEST_FILE, label_to_id, is_train=False)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    sample_x, _ = train_dataset[0]
    input_size = sample_x.shape[1]
    print("Input size:", input_size)
    print("CNN blocks (layers being tested):", CNN_BLOCKS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SignLSTM_5CNN(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(classes),
        dropout=DROPOUT,
        cnn_blocks=CNN_BLOCKS,
    ).to(device)

    print("Loading previous v3-continue checkpoint (partial load):", LOAD_MODEL_PATH)
    checkpoint = torch.load(LOAD_MODEL_PATH, map_location=device, weights_only=False)
    checkpoint_state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # strict=False: the checkpoint's old cnn1/cnn2 keys won't match the new
    # cnn_blocks.* structure and are skipped -- LSTM/attention/classifier
    # weights (identical shapes) load normally.
    load_result = model.load_state_dict(checkpoint_state, strict=False)
    print("Loaded matching keys. Skipped (new/renamed):", load_result.missing_keys)
    print("Ignored from checkpoint (old CNN shape, expected):", load_result.unexpected_keys)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            train_loss += loss.item() * x.size(0)
            train_correct += (preds == y).sum().item()
            train_total += x.size(0)

        train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "cnn_blocks": CNN_BLOCKS,
                "num_classes": len(classes),
                "dropout": DROPOUT,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }, OUTPUT_DIR / "v3_5cnn_model.pt")
            print("Saved best model.")

    checkpoint = torch.load(OUTPUT_DIR / "v3_5cnn_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("=" * 80)
    print("Final result -- 5-CNN-layer experiment")
    print("Best validation accuracy:", round(best_val_acc, 4))
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("For comparison, v3-continue (1 CNN block) test accuracy was ~0.33-0.35")
    print("=" * 80)

    pd.DataFrame(history).to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "sentence_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "sentence_label_encoder.pkl", "wb") as f:
        pickle.dump({"classes": classes, "label_to_id": label_to_id, "id_to_label": id_to_label}, f)

    summary = {
        "cnn_blocks": CNN_BLOCKS,
        "num_classes": len(classes),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }

    with open(RESULT_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved model:", OUTPUT_DIR)
    print("Saved results:", RESULT_DIR)


if __name__ == "__main__":
    main()
