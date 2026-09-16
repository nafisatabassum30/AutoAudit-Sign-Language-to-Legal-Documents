import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm.dataset_utils import load_dataset_frame

try:
    df = load_dataset_frame('data/raw/sign_language_conflict.xlsx')
    print(f"✅ Success! Loaded {len(df)} rows")
    print(f"📋 Columns: {df.columns.tolist()}")
    print("\n📝 First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("❌ File not found! Make sure sign_language_conflict.xlsx is in data/raw/")
    print("   Current location: data/raw/sign_language_conflict.xlsx")
except Exception as e:
    print(f"❌ Error: {e}")