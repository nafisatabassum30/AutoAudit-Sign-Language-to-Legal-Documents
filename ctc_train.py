"""
ctc_train.py
Full training loop for the v4 CNN+BiLSTM+CTC sign-sentence recognizer,
evaluated with Word Error Rate (WER) instead of exact-sentence accuracy.

Run from the project root:
    python sign_sentence_v4_ctc/ctc_train.py
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ctc_dataset import (
    CTCSignDataset,
    ctc_collate_fn,
    load_vocab,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
)
from ctc_model import CTCSignModel, HIDDEN_SIZE, NUM_LSTM_LAYERS, CNN_LAYERS, DROPOUT

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RESULT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)


def ctc_greedy_decode(log_probs, input_lengths, id_to_word):
    """
    Best-path CTC decoding: argmax each frame, collapse consecutive
    repeats, then drop blanks (id 0).
    log_probs: [time, batch, vocab]  ->  returns list of word lists, one per sample.
    """
    predictions = torch.argmax(log_probs, dim=-1)  # [time, batch]
    predictions = predictions.transpose(0, 1)  # [batch, time]

    decoded_batch = []
    for i in range(predictions.shape[0]):
        seq = predictions[i, : input_lengths[i]].tolist()

        collapsed = []
        prev = None
        for token_id in seq:
            if token_id != prev:
                collapsed.append(token_id)
            prev = token_id

        words = [id_to_word[t] for t in collapsed if t != 0]  # drop blank
        decoded_batch.append(words)

    return decoded_batch


def word_edit_distance(ref_words, hyp_words):
    """Standard Levenshtein distance at the word level."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def evaluate(model, loader, criterion, id_to_word, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_edits = 0
    total_ref_words = 0
    exact_matches = 0

    with torch.no_grad():
        for padded_keypoints, targets, input_lengths, target_lengths in loader:
            padded_keypoints = padded_keypoints.to(device)
            targets = targets.to(device)
            input_lengths_dev = input_lengths.to(device)
            target_lengths_dev = target_lengths.to(device)

            log_probs = model(padded_keypoints)
            loss = criterion(log_probs, targets, input_lengths_dev, target_lengths_dev)

            total_loss += loss.item() * padded_keypoints.size(0)
            total_samples += padded_keypoints.size(0)

            decoded_batch = ctc_greedy_decode(log_probs, input_lengths, id_to_word)

            # Split the concatenated targets back into per-sample word lists.
            targets_cpu = targets.cpu().tolist()
            offset = 0
            for i, t_len in enumerate(target_lengths.tolist()):
                ref_ids = targets_cpu[offset: offset + t_len]
                offset += t_len
                ref_words = [id_to_word[t] for t in ref_ids]
                hyp_words = decoded_batch[i]

                total_edits += word_edit_distance(ref_words, hyp_words)
                total_ref_words += len(ref_words)
                if ref_words == hyp_words:
                    exact_matches += 1

    avg_loss = total_loss / max(total_samples, 1)
    wer = total_edits / max(total_ref_words, 1)
    sentence_acc = exact_matches / max(total_samples, 1)

    return avg_loss, wer, sentence_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    word_to_id, id_to_word = load_vocab()
    vocab_size = len(word_to_id)
    print("Vocab size (incl. blank):", vocab_size)

    train_dataset = CTCSignDataset(TRAIN_FILE, word_to_id, is_train=True)
    val_dataset = CTCSignDataset(VAL_FILE, word_to_id, is_train=False)
    test_dataset = CTCSignDataset(TEST_FILE, word_to_id, is_train=False)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ctc_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ctc_collate_fn)

    sample_kp, _ = train_dataset[0]
    feature_dim = sample_kp.shape[1]

    model = CTCSignModel(
        input_size=feature_dim,
        hidden_size=HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        vocab_size=vocab_size,
        cnn_layers=CNN_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Lower WER is better, so we track improvement with mode="min".
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    best_val_wer = float("inf")
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for padded_keypoints, targets, input_lengths, target_lengths in train_loader:
            padded_keypoints = padded_keypoints.to(device)
            targets = targets.to(device)
            input_lengths_dev = input_lengths.to(device)
            target_lengths_dev = target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(padded_keypoints)
            loss = criterion(log_probs, targets, input_lengths_dev, target_lengths_dev)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * padded_keypoints.size(0)
            train_samples += padded_keypoints.size(0)

        train_loss = train_loss / max(train_samples, 1)

        val_loss, val_wer, val_sentence_acc = evaluate(model, val_loader, criterion, id_to_word, device)

        scheduler.step(val_wer)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_sentence_acc": val_sentence_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_wer={val_wer:.4f} val_sentence_acc={val_sentence_acc:.4f}"
        )

        if val_wer < best_val_wer:
            best_val_wer = val_wer

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_size": feature_dim,
                "hidden_size": HIDDEN_SIZE,
                "num_lstm_layers": NUM_LSTM_LAYERS,
                "cnn_layers": CNN_LAYERS,
                "dropout": DROPOUT,
                "vocab_size": vocab_size,
                "word_to_id": word_to_id,
                "id_to_word": id_to_word,
                "best_val_wer": best_val_wer,
                "epoch": epoch,
            }, OUTPUT_DIR / "ctc_sign_model.pt")

            print("Saved best model (val_wer improved).")

    # Load best checkpoint for final test evaluation.
    checkpoint = torch.load(OUTPUT_DIR / "ctc_sign_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_wer, test_sentence_acc = evaluate(model, test_loader, criterion, id_to_word, device)

    print("=" * 80)
    print("Final result")
    print("Best validation WER:", round(best_val_wer, 4))
    print("Test loss:", round(test_loss, 4))
    print("Test WER:", round(test_wer, 4))
    print("Test sentence accuracy (exact match):", round(test_sentence_acc, 4))
    print("=" * 80)

    pd.DataFrame(history).to_csv(RESULT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    summary = {
        "vocab_size": vocab_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_wer": float(best_val_wer),
        "test_loss": float(test_loss),
        "test_wer": float(test_wer),
        "test_sentence_acc": float(test_sentence_acc),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "cnn_layers": CNN_LAYERS,
        "dropout": DROPOUT,
    }

    with open(RESULT_DIR / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved model:", OUTPUT_DIR)
    print("Saved results:", RESULT_DIR)


if __name__ == "__main__":
    main()
