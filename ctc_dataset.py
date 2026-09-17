"""
ctc_dataset.py
Dataset + collate function for the v4 CTC sign-sentence recognizer.

Unlike v2/v3 (one sentence = one class label), each sample here returns:
  - keypoints: [T, F] float32 tensor (T = frames, F = keypoint features)
  - word_ids:  [L] long tensor (L = number of words in the sentence)

CTCLoss aligns the T-length keypoint sequence to the L-length word sequence
on its own during training -- no frame-level word boundaries needed.

Run directly to sanity-check one batch:
    python sign_sentence_v4_ctc/ctc_dataset.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT_DIR / "data" / "split_v4_no_leak"
KEYPOINT_DIR = ROOT_DIR / "data" / "keypoints_full_64"
VOCAB_DIR = Path(__file__).resolve().parent / "output"

TRAIN_FILE = SPLIT_DIR / "train_manifest.csv"
VAL_FILE = SPLIT_DIR / "val_manifest.csv"
TEST_FILE = SPLIT_DIR / "test_manifest.csv"


def load_vocab():
    with open(VOCAB_DIR / "word_to_id.json", "r", encoding="utf-8") as f:
        word_to_id = json.load(f)
    with open(VOCAB_DIR / "id_to_word.json", "r", encoding="utf-8") as f:
        id_to_word = {int(k): v for k, v in json.load(f).items()}
    return word_to_id, id_to_word


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
    if np.random.rand() < 0.6:
        x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        x = x * scale

    # Time-masking: zero out a contiguous chunk of frames (not just one),
    # forcing the model to rely on surrounding context instead of
    # memorizing exact frame-by-frame patterns. Key fix for overfitting
    # with few training sentences.
    if np.random.rand() < 0.5:
        mask_len = np.random.randint(2, max(3, x.shape[0] // 6))
        start = np.random.randint(0, max(1, x.shape[0] - mask_len))
        x[start:start + mask_len] = 0

    # Random joint dropout: zero random individual keypoint columns across
    # all frames, simulating occlusion/tracking failure rather than a
    # missing timestep.
    if np.random.rand() < 0.4:
        num_joints_to_drop = max(1, x.shape[1] // 20)
        joint_idx = np.random.choice(x.shape[1], num_joints_to_drop, replace=False)
        x[:, joint_idx] = 0

    return x.astype(np.float32)


class CTCSignDataset(Dataset):
    def __init__(self, manifest_file, word_to_id, is_train=False):
        self.df = pd.read_csv(manifest_file, encoding="utf-8-sig")
        self.df["sentence_clean"] = self.df["sentence"].astype(str).str.strip()
        self.word_to_id = word_to_id
        self.is_train = is_train

        self.items = []
        missing_keypoints = 0
        missing_words = 0

        for _, row in self.df.iterrows():
            sentence = row["sentence_clean"]
            index_value = row["index"]

            try:
                kp_file = find_keypoint_file(index_value)
            except FileNotFoundError:
                missing_keypoints += 1
                continue

            tokens = sentence.split()
            try:
                word_ids = [self.word_to_id[t] for t in tokens]
            except KeyError:
                missing_words += 1
                continue

            self.items.append({
                "keypoint_file": kp_file,
                "word_ids": word_ids,
                "sentence": sentence,
                "index": clean_index(index_value),
            })

        if missing_keypoints > 0:
            print(f"Warning: missing keypoint files: {missing_keypoints}")
        if missing_words > 0:
            print(f"Warning: sentences with out-of-vocab words: {missing_words}")

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

        keypoints = torch.tensor(x, dtype=torch.float32)          # [T, F]
        word_ids = torch.tensor(item["word_ids"], dtype=torch.long)  # [L]

        return keypoints, word_ids


def ctc_collate_fn(batch):
    """
    Pads keypoint sequences to the batch max length and concatenates
    word-id targets into a single 1D tensor, as expected by nn.CTCLoss.
    """
    keypoints_list, word_ids_list = zip(*batch)

    input_lengths = torch.tensor([kp.shape[0] for kp in keypoints_list], dtype=torch.long)
    target_lengths = torch.tensor([w.shape[0] for w in word_ids_list], dtype=torch.long)

    feature_dim = keypoints_list[0].shape[1]
    max_time = int(input_lengths.max().item())

    padded_keypoints = torch.zeros(len(batch), max_time, feature_dim, dtype=torch.float32)
    for i, kp in enumerate(keypoints_list):
        padded_keypoints[i, :kp.shape[0], :] = kp

    concatenated_targets = torch.cat(word_ids_list)  # 1D, required shape for CTCLoss

    return padded_keypoints, concatenated_targets, input_lengths, target_lengths


def main():
    word_to_id, id_to_word = load_vocab()
    print("Vocab size:", len(word_to_id))

    train_dataset = CTCSignDataset(TRAIN_FILE, word_to_id, is_train=True)
    val_dataset = CTCSignDataset(VAL_FILE, word_to_id, is_train=False)
    test_dataset = CTCSignDataset(TEST_FILE, word_to_id, is_train=False)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    kp, word_ids = train_dataset[0]
    print("\nSingle sample check:")
    print("  keypoints shape:", tuple(kp.shape))
    print("  word_ids:", word_ids.tolist())
    print("  decoded words:", [id_to_word[i] for i in word_ids.tolist()])

    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=ctc_collate_fn)
    padded_keypoints, targets, input_lengths, target_lengths = next(iter(loader))

    print("\nBatch check (batch_size=4):")
    print("  padded_keypoints shape:", tuple(padded_keypoints.shape))
    print("  concatenated targets shape:", tuple(targets.shape))
    print("  input_lengths:", input_lengths.tolist())
    print("  target_lengths:", target_lengths.tolist())


if __name__ == "__main__":
    main()
