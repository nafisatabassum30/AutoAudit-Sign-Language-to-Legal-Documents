"""
ctc_model.py
CNN + BiLSTM encoder with a CTC output head for the v4 sign-sentence
recognizer. Same CNN-residual + BiLSTM backbone spirit as your v3 model,
but the output is a per-frame distribution over the word vocabulary
(+blank) instead of a single softmax over sentence classes.

Run directly to sanity-check a forward pass + one CTC loss step on a
real batch from ctc_dataset.py.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ctc_dataset import CTCSignDataset, ctc_collate_fn, load_vocab, TRAIN_FILE

HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
CNN_LAYERS = 3  # residual CNN blocks -- this is your "increase CNN layers" lever, now applied safely


class ResidualCNNBlock(nn.Module):
    """
    One residual 1D-conv block over the time axis. Input/output feature
    dim stays the same, so any number of these can be stacked without
    changing the BiLSTM's expected input size.
    """

    def __init__(self, feature_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(feature_dim)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(feature_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Zero-init the second conv so each block starts as a near-identity
        # function -- same "safe residual" trick your v3 model used, applied
        # per-block so stacking more blocks doesn't destabilize early training.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        # x: [batch, features, time]
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return residual + out


class CTCSignModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, vocab_size,
                 cnn_layers=CNN_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.cnn_blocks = nn.ModuleList([
            ResidualCNNBlock(input_size, dropout) for _ in range(cnn_layers)
        ])

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out_size = hidden_size * 2

        # Per-frame classifier over the word vocabulary + blank.
        # vocab_size already includes the blank token at id 0 (see build_vocab.py).
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, x):
        # x: [batch, time, features]
        cnn_in = x.transpose(1, 2)  # -> [batch, features, time]

        cnn_out = cnn_in
        for block in self.cnn_blocks:
            cnn_out = block(cnn_out)

        cnn_out = cnn_out.transpose(1, 2)  # -> [batch, time, features]

        lstm_out, _ = self.lstm(cnn_out)  # [batch, time, hidden*2]

        logits = self.classifier(lstm_out)  # [batch, time, vocab_size]

        # CTCLoss expects [time, batch, vocab_size] log-probabilities.
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)

        return log_probs


def main():
    word_to_id, id_to_word = load_vocab()
    vocab_size = len(word_to_id)

    train_dataset = CTCSignDataset(TRAIN_FILE, word_to_id, is_train=True)
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=ctc_collate_fn)

    padded_keypoints, targets, input_lengths, target_lengths = next(iter(loader))

    feature_dim = padded_keypoints.shape[2]
    print("Feature dim:", feature_dim)
    print("Vocab size (incl. blank):", vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = CTCSignModel(
        input_size=feature_dim,
        hidden_size=HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        vocab_size=vocab_size,
        cnn_layers=CNN_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    padded_keypoints = padded_keypoints.to(device)
    targets = targets.to(device)
    input_lengths = input_lengths.to(device)
    target_lengths = target_lengths.to(device)

    log_probs = model(padded_keypoints)
    print("log_probs shape [time, batch, vocab]:", tuple(log_probs.shape))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    print("CTC loss on one random batch (untrained model):", loss.item())

    loss.backward()
    print("Backward pass OK -- gradients computed successfully.")


if __name__ == "__main__":
    main()
