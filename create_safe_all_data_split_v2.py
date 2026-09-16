import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/train/video_sentence_manifest.csv")
OUT_DIR = Path("data/split_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
df["sentence_clean"] = df["sentence"].astype(str).str.strip()
df["video_path_clean"] = df["video_path"].astype(str).str.strip()

df = df.drop_duplicates(subset=["video_path_clean"]).copy()
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

train_rows = []
remaining_rows = []

# First rule: every sentence/class must have at least one video in train
for sentence, group in df.groupby("sentence_clean"):
    group = group.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train_rows.append(group.iloc[0])

    if len(group) > 1:
        for _, r in group.iloc[1:].iterrows():
            remaining_rows.append(r)

train_df = pd.DataFrame(train_rows)
remaining_df = pd.DataFrame(remaining_rows)

total = len(df)
target_train = int(round(total * TRAIN_RATIO))
target_val = int(round(total * VAL_RATIO))

# Add extra videos to train until train becomes about 70%
need_more_train = max(0, target_train - len(train_df))

remaining_df = remaining_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

extra_train_df = remaining_df.iloc[:need_more_train].copy()
rest_df = remaining_df.iloc[need_more_train:].copy()

train_df = pd.concat([train_df, extra_train_df], ignore_index=True)

rest_df = rest_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

val_df = rest_df.iloc[:target_val].copy()
test_df = rest_df.iloc[target_val:].copy()

keep_cols = ["index", "sentence", "video_path"]

train_df[keep_cols].to_csv(OUT_DIR / "train_manifest.csv", index=False, encoding="utf-8-sig")
val_df[keep_cols].to_csv(OUT_DIR / "val_manifest.csv", index=False, encoding="utf-8-sig")
test_df[keep_cols].to_csv(OUT_DIR / "test_manifest.csv", index=False, encoding="utf-8-sig")

all_split_df = pd.concat([
    train_df.assign(split="train"),
    val_df.assign(split="val"),
    test_df.assign(split="test")
], ignore_index=True)

all_split_df[["index", "sentence", "video_path", "split"]].to_csv(
    OUT_DIR / "all_data_split_manifest.csv",
    index=False,
    encoding="utf-8-sig"
)

split_summary = all_split_df.groupby(["sentence", "split"]).size().unstack(fill_value=0).reset_index()
split_summary.to_csv(OUT_DIR / "sentence_split_summary.csv", index=False, encoding="utf-8-sig")

print("Safe split created.")
print("=" * 60)
print("Total videos:", total)
print("Train videos:", len(train_df), f"({len(train_df)/total*100:.2f}%)")
print("Validation videos:", len(val_df), f"({len(val_df)/total*100:.2f}%)")
print("Test videos:", len(test_df), f"({len(test_df)/total*100:.2f}%)")
print("=" * 60)
print("Train unique classes:", train_df["sentence"].astype(str).str.strip().nunique())
print("Validation unique classes:", val_df["sentence"].astype(str).str.strip().nunique())
print("Test unique classes:", test_df["sentence"].astype(str).str.strip().nunique())
print("=" * 60)
print("Saved in:", OUT_DIR)
