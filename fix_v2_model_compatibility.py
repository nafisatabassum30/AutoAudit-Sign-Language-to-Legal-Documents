import torch
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path("output/sign_sentence_v2_validated")
MODEL_FILE = MODEL_DIR / "sign_sentence_lstm_model.pt"
BACKUP_FILE = MODEL_DIR / "sign_sentence_lstm_model_before_compat_fix.pt"
ENCODER_FILE = MODEL_DIR / "sentence_label_encoder.pkl"

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

checkpoint = torch.load(MODEL_FILE, map_location="cpu")

# backup original checkpoint
if not BACKUP_FILE.exists():
    torch.save(checkpoint, BACKUP_FILE)
    print("Backup saved:", BACKUP_FILE)

# Add old prediction-code compatible keys
if "input_dim" not in checkpoint and "input_size" in checkpoint:
    checkpoint["input_dim"] = checkpoint["input_size"]

if "hidden_dim" not in checkpoint and "hidden_size" in checkpoint:
    checkpoint["hidden_dim"] = checkpoint["hidden_size"]

# Make sure label classes exist
if "label_to_id" in checkpoint:
    label_to_id = checkpoint["label_to_id"]
    classes = [None] * len(label_to_id)

    for label, idx in label_to_id.items():
        classes[int(idx)] = label

elif "id_to_label" in checkpoint:
    id_to_label = checkpoint["id_to_label"]
    classes = [id_to_label[i] for i in range(len(id_to_label))]

else:
    raise KeyError("No label_to_id or id_to_label found in checkpoint")

# Save compatible checkpoint
torch.save(checkpoint, MODEL_FILE)
print("Checkpoint compatibility fixed.")
print("Added/checked keys: input_dim, hidden_dim")

# Save sklearn LabelEncoder because old predict_from_video.py may expect it
le = LabelEncoder()
le.classes_ = __import__("numpy").array(classes)

with open(ENCODER_FILE, "wb") as f:
    pickle.dump(le, f)

print("Compatible LabelEncoder saved:", ENCODER_FILE)
print("Total classes:", len(classes))
