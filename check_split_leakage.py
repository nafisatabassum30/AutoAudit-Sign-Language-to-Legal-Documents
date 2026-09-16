import pandas as pd
from pathlib import Path

SPLIT_DIR = Path("data/split_v2")

train_file = SPLIT_DIR / "train_manifest.csv"
val_file = SPLIT_DIR / "val_manifest.csv"
test_file = SPLIT_DIR / "test_manifest.csv"

for f in [train_file, val_file, test_file]:
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

train = pd.read_csv(train_file, encoding="utf-8-sig")
val = pd.read_csv(val_file, encoding="utf-8-sig")
test = pd.read_csv(test_file, encoding="utf-8-sig")

train_videos = set(train["video_path"].astype(str).str.strip())
val_videos = set(val["video_path"].astype(str).str.strip())
test_videos = set(test["video_path"].astype(str).str.strip())

train_classes = set(train["sentence"].astype(str).str.strip())
val_classes = set(val["sentence"].astype(str).str.strip())
test_classes = set(test["sentence"].astype(str).str.strip())

print("Train videos:", len(train))
print("Validation videos:", len(val))
print("Test videos:", len(test))
print("Total:", len(train) + len(val) + len(test))

print("\nDuplicate video leakage check:")
print("Train ∩ Validation:", len(train_videos & val_videos))
print("Train ∩ Test:", len(train_videos & test_videos))
print("Validation ∩ Test:", len(val_videos & test_videos))

print("\nClass availability check:")
print("Validation classes missing from train:", len(val_classes - train_classes))
print("Test classes missing from train:", len(test_classes - train_classes))

if (
    len(train_videos & val_videos) == 0
    and len(train_videos & test_videos) == 0
    and len(val_videos & test_videos) == 0
    and len(val_classes - train_classes) == 0
    and len(test_classes - train_classes) == 0
):
    print("\nRESULT: SPLIT IS SAFE")
else:
    print("\nRESULT: SPLIT HAS PROBLEM")
