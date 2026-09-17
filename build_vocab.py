"""
build_vocab.py
Builds the word-level vocabulary for the v4 CTC sign-sentence recognizer.

Reads data/raw/FinalSheet2.xlsx (columns: Names, Index), tokenizes each
Bangla sentence into words, and saves a word_to_id / id_to_word mapping.
Index 0 is reserved for the CTC "blank" token — never a real word.

Run from the project root:
    python sign_sentence_v4_ctc/build_vocab.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT_DIR / "data" / "raw" / "FinalSheet2.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CTC_BLANK_TOKEN = "<blank>"


def tokenize(sentence: str):
    return str(sentence).strip().split()


def main():
    print("Loading:", RAW_FILE)
    df = pd.read_excel(RAW_FILE)

    if "Names" not in df.columns or "Index" not in df.columns:
        raise ValueError(f"Expected columns 'Names' and 'Index', got: {list(df.columns)}")

    print("Total rows (videos):", len(df))
    print("Unique sentences:", df["Names"].nunique())

    word_counter = Counter()
    sentence_lengths = []

    for sentence in df["Names"].astype(str):
        tokens = tokenize(sentence)
        sentence_lengths.append(len(tokens))
        word_counter.update(tokens)

    vocab_words = sorted(word_counter.keys())

    # Reserve id 0 for CTC blank; real words start at 1.
    word_to_id = {CTC_BLANK_TOKEN: 0}
    for i, word in enumerate(vocab_words, start=1):
        word_to_id[word] = i

    id_to_word = {i: w for w, i in word_to_id.items()}

    print("Vocabulary size (including blank):", len(word_to_id))
    print("Max sentence length (words):", max(sentence_lengths))
    print("Avg sentence length (words):", sum(sentence_lengths) / len(sentence_lengths))

    with open(OUTPUT_DIR / "word_to_id.json", "w", encoding="utf-8") as f:
        json.dump(word_to_id, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "id_to_word.json", "w", encoding="utf-8") as f:
        json.dump(id_to_word, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "word_frequencies.json", "w", encoding="utf-8") as f:
        json.dump(word_counter.most_common(), f, ensure_ascii=False, indent=2)

    print("Saved vocab files to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
