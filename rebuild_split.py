"""
rebuild_split.py
Rebuilds train/val/test manifests so that every SENTENCE (not just every
video) appears in exactly one split. Your existing data/split_v2 manifests
allowed the same sentence's multiple video recordings to land in different
splits, which leaks information from train into val/test.

Reads the union of your existing split_v2 manifests (so we only use
indices that already have extracted keypoints -- no re-extraction needed),
regroups by sentence, and writes a new leak-free split.

Run from the project root:
    python sign_sentence_v4_ctc/rebuild_split.py
"""

import random
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
OLD_SPLIT_DIR = ROOT_DIR / "data" / "split_v2"
NEW_SPLIT_DIR = ROOT_DIR / "data" / "split_v4_no_leak"
NEW_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
# Target video counts, matching the original split's sizes for a fair comparison.
TARGET_VAL = 752
TARGET_TEST = 751


def has_keypoints(index_value):
    index_text = str(index_value).strip()
    if index_text.endswith(".0"):
        index_text = index_text[:-2]

    candidates = [
        Path("data/keypoints_full") / f"{index_text}.npy",
        Path("data/keypoints_full") / f"keypoints_{index_text}.npy",
        Path("data/keypoints_full") / "keypoints" / f"{index_text}.npy",
    ]
    return any(path.exists() for path in candidates)


def main():
    random.seed(RANDOM_SEED)

    train = pd.read_csv(OLD_SPLIT_DIR / "train_manifest.csv", encoding="utf-8-sig")
    val = pd.read_csv(OLD_SPLIT_DIR / "val_manifest.csv", encoding="utf-8-sig")
    test = pd.read_csv(OLD_SPLIT_DIR / "test_manifest.csv", encoding="utf-8-sig")

    all_rows = pd.concat([train, val, test], ignore_index=True)
    all_rows["sentence"] = all_rows["sentence"].astype(str).str.strip()
    all_rows = all_rows.drop_duplicates(subset="index")

    before_filter = len(all_rows)
    all_rows = all_rows[all_rows["index"].map(has_keypoints)].copy()
    print("Rows without extracted keypoints skipped:", before_filter - len(all_rows))

    print("Total available videos:", len(all_rows))
    print("Total unique sentences:", all_rows["sentence"].nunique())

    # Group video indices by sentence.
    sentence_to_rows = {}
    for _, row in all_rows.iterrows():
        sentence_to_rows.setdefault(row["sentence"], []).append(row)

    sentences = list(sentence_to_rows.keys())
    random.shuffle(sentences)

    val_rows, test_rows, train_rows = [], [], []
    val_count, test_count = 0, 0

    for sentence in sentences:
        rows = sentence_to_rows[sentence]

        if val_count < TARGET_VAL:
            val_rows.extend(rows)
            val_count += len(rows)
        elif test_count < TARGET_TEST:
            test_rows.extend(rows)
            test_count += len(rows)
        else:
            train_rows.extend(rows)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    print("\nNew split sizes:")
    print("  train:", len(train_df), "videos,", train_df["sentence"].nunique(), "unique sentences")
    print("  val:  ", len(val_df), "videos,", val_df["sentence"].nunique(), "unique sentences")
    print("  test: ", len(test_df), "videos,", test_df["sentence"].nunique(), "unique sentences")

    train_sents = set(train_df["sentence"])
    val_sents = set(val_df["sentence"])
    test_sents = set(test_df["sentence"])

    print("\nLeakage check (should all be 0):")
    print("  train & val overlap:", len(train_sents & val_sents))
    print("  train & test overlap:", len(train_sents & test_sents))
    print("  val & test overlap:", len(val_sents & test_sents))

    train_df.to_csv(NEW_SPLIT_DIR / "train_manifest.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(NEW_SPLIT_DIR / "val_manifest.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(NEW_SPLIT_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")

    print("\nSaved new leak-free split to:", NEW_SPLIT_DIR)


if __name__ == "__main__":
    main()
