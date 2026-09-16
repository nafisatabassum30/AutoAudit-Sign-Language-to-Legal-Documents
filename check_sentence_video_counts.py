import pandas as pd
from pathlib import Path

MANIFEST = Path("data/train/video_sentence_manifest.csv")
OUT_DIR = Path("outputs/dataset_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST, encoding="utf-8-sig")

print("Columns:", list(df.columns))
print("Total rows:", len(df))

df["sentence_clean"] = df["sentence"].astype(str).str.strip()

counts = df["sentence_clean"].value_counts().reset_index()
counts.columns = ["sentence", "video_count"]

counts.to_csv(OUT_DIR / "sentence_video_count.csv", index=False, encoding="utf-8-sig")

print("\nTotal unique sentences:", len(counts))
print("\nVideo count distribution:")
print(counts["video_count"].value_counts().sort_index())

print("\nTop 20 sentences with most videos:")
print(counts.head(20).to_string(index=False))

print("\nHow many sentences can be used for proper train/val/test?")
print("Sentences with >= 3 videos:", (counts["video_count"] >= 3).sum())
print("Sentences with >= 5 videos:", (counts["video_count"] >= 5).sum())
print("Sentences with >= 10 videos:", (counts["video_count"] >= 10).sum())

print("\nSaved:")
print(OUT_DIR / "sentence_video_count.csv")
