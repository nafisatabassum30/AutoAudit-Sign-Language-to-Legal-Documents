import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SPLIT_DIR = Path("data/split_v2")
KEYPOINT_DIR = Path("data/keypoints_full")
OUTPUT_DIR = Path("output/sign_sentence_v3_v2plus_cnn_attention")
RESULT_DIR = Path("outputs/sign_sentence_v3_v2plus_cnn_attention")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train_manifest.csv"
VAL_FILE = SPLIT_DIR / "val_manifest.csv"
TEST_FILE = SPLIT_DIR / "test_manifest.csv"

NUM_EPOCHS = 35
BATCH_SIZE = 32
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.4
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

    # Replace NaN/inf safely
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize per video
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / (std + 1e-6)

    return x.astype(np.float32)


def augment_keypoints(x):
    # Small noise
    if np.random.rand() < 0.5:
        x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)

    # Small scale
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.95, 1.05)
        x = x * scale

    # Small temporal masking
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


class SignLSTM(nn.Module):
    """
    v3 model name is kept as SignLSTM_v3_CNN_Attention so the original v2 training script can reuse
    the same training/evaluation/checkpoint code safely.

    Architecture:
    MediaPipe keypoints
    -> Safe residual 1D CNN over time
    -> BiLSTM
    -> Soft temporal attention + last timestep fusion
    -> Classifier

    Design reason:
    - The 1D CNN applies Dhrubo-style CNN feature extraction idea to keypoint sequence.
    - Soft attention applies Dhrubo-style temporal attention idea.
    - The model is initialized to behave close to the validated v2 LSTM at the beginning,
      so the CNN/attention additions should not destroy the working v2 learning path.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()

        # Safe temporal CNN residual block.
        # Input and output dimension remain input_size, so the BiLSTM interface stays same as v2.
        self.cnn1 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(input_size, input_size, kernel_size=3, padding=1)
        self.cnn_activation = nn.ReLU()
        self.cnn_dropout = nn.Dropout(dropout * 0.5)

        # Important safety:
        # Zero-initialize second CNN layer so at the beginning CNN residual is almost zero.
        # Therefore model starts close to the original v2 BiLSTM behavior.
        nn.init.zeros_(self.cnn2.weight)
        nn.init.zeros_(self.cnn2.bias)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        lstm_out_size = hidden_size * 2

        # Soft temporal attention over BiLSTM outputs.
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Learnable gate between v2-style last timestep and attention context.
        # Initialized negative so model initially trusts last timestep more.
        self.attn_gate = nn.Parameter(torch.tensor(-2.0))

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x: [batch, time, features]

        # 1D CNN expects [batch, features, time]
        cnn_in = x.transpose(1, 2)
        cnn_out = self.cnn1(cnn_in)
        cnn_out = self.cnn_activation(cnn_out)
        cnn_out = self.cnn_dropout(cnn_out)
        cnn_out = self.cnn2(cnn_out)
        cnn_out = cnn_out.transpose(1, 2)

        # Residual CNN feature added to original keypoints
        x = x + cnn_out

        out, _ = self.lstm(x)

        # v2 path: last timestep
        last = out[:, -1, :]

        # Dhrubo-inspired soft temporal attention path
        attn_scores = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)

        # Safe fusion: starts close to v2 last timestep, can learn to use attention
        gate = torch.sigmoid(self.attn_gate)
        fused = (1.0 - gate) * last + gate * context

        logits = self.classifier(fused)
        return logits


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_rows = []

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SignLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(classes),
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

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

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }
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
                "num_classes": len(classes),
                "dropout": DROPOUT,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "best_val_acc": best_val_acc,
                "epoch": epoch
            }, OUTPUT_DIR / "sign_sentence_v3_v2plus_cnn_attention_model.pt")

            print("Saved best model.")

    # Load best model for final test
    checkpoint = torch.load(OUTPUT_DIR / "sign_sentence_v3_v2plus_cnn_attention_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print("=" * 80)
    print("Final result")
    print("Best validation accuracy:", round(best_val_acc, 4))
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))
    print("=" * 80)

    pd.DataFrame(history).to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    with open(OUTPUT_DIR / "sentence_classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "sentence_label_encoder.pkl", "wb") as f:
        pickle.dump({
            "classes": classes,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label
        }, f)

    summary = {
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
        "dropout": DROPOUT
    }

    with open(RESULT_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved model:", OUTPUT_DIR)
    print("Saved results:", RESULT_DIR)


if __name__ == "__main__":
    main()

