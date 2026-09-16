import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


SPLIT_DIR = Path("data/split_v2")
KEYPOINT_DIR = Path("data/keypoints_full")
MODEL_DIR = Path("output/sign_sentence_v2_validated")
RESULT_DIR = Path("outputs/sign_sentence_v2_validated")

MODEL_FILE = MODEL_DIR / "sign_sentence_lstm_model.pt"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
        logits = self.classifier(last)
        return logits


def clean_index(x):
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_keypoint(index_value):
    idx = clean_index(index_value)
    path = KEYPOINT_DIR / f"{idx}.npy"

    if path.exists():
        return path

    matches = list(KEYPOINT_DIR.rglob(f"{idx}.npy"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Keypoint not found for index: {idx}")


def normalize_keypoints(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)

    x = (x - mean) / (std + 1e-6)
    return x.astype(np.float32)


def load_model(device):
    checkpoint = torch.load(MODEL_FILE, map_location=device)

    input_size = checkpoint.get("input_size", checkpoint.get("input_dim"))
    hidden_size = checkpoint.get("hidden_size", checkpoint.get("hidden_dim"))
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.4)
    num_classes = checkpoint.get("num_classes")

    if "label_to_id" in checkpoint:
        label_to_id = checkpoint["label_to_id"]
        id_to_label = {int(v): k for k, v in label_to_id.items()}
    elif "id_to_label" in checkpoint:
        id_to_label = {int(k): v for k, v in checkpoint["id_to_label"].items()}
        label_to_id = {v: k for k, v in id_to_label.items()}
    else:
        raise KeyError("No label_to_id or id_to_label found")

    if num_classes is None:
        num_classes = len(id_to_label)

    model = SignLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, label_to_id, id_to_label


def evaluate_split(split_name, model, label_to_id, id_to_label, device):
    manifest_file = SPLIT_DIR / f"{split_name}_manifest.csv"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing split file: {manifest_file}")

    df = pd.read_csv(manifest_file, encoding="utf-8-sig")

    rows = []
    correct = 0
    total = 0

    for _, row in df.iterrows():
        index_value = row["index"]
        expected = str(row["sentence"]).strip()

        kp_file = find_keypoint(index_value)
        x = np.load(kp_file)
        x = normalize_keypoints(x)

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_ids = torch.topk(probs, k=5)

        top_ids = top_ids.cpu().numpy()
        top_probs = top_probs.cpu().numpy()

        predicted = id_to_label[int(top_ids[0])]
        confidence = float(top_probs[0])

        is_correct = expected == predicted

        if is_correct:
            correct += 1

        total += 1

        rows.append({
            "split": split_name,
            "index": index_value,
            "video_path": row["video_path"],
            "expected_sentence": expected,
            "predicted_sentence": predicted,
            "confidence": round(confidence, 4),
            "correct": "YES" if is_correct else "NO",
            "top1": id_to_label[int(top_ids[0])],
            "top1_conf": round(float(top_probs[0]), 4),
            "top2": id_to_label[int(top_ids[1])],
            "top2_conf": round(float(top_probs[1]), 4),
            "top3": id_to_label[int(top_ids[2])],
            "top3_conf": round(float(top_probs[2]), 4),
            "top4": id_to_label[int(top_ids[3])],
            "top4_conf": round(float(top_probs[3]), 4),
            "top5": id_to_label[int(top_ids[4])],
            "top5_conf": round(float(top_probs[4]), 4)
        })

    result_df = pd.DataFrame(rows)

    out_file = RESULT_DIR / f"{split_name}_prediction_report.csv"
    result_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    acc = correct / max(total, 1)

    print("=" * 80)
    print(f"{split_name.upper()} RESULT")
    print("Total:", total)
    print("Correct:", correct)
    print("Wrong:", total - correct)
    print("Accuracy:", round(acc, 4))
    print("Saved:", out_file)

    return {
        "split": split_name,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": acc
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, label_to_id, id_to_label = load_model(device)

    summaries = []

    for split_name in ["val", "test"]:
        summaries.append(
            evaluate_split(split_name, model, label_to_id, id_to_label, device)
        )

    summary_file = RESULT_DIR / "validation_test_prediction_summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Summary saved:", summary_file)


if __name__ == "__main__":
    main()
