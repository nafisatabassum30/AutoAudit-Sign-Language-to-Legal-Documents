import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from llm.dataset_utils import load_dataset_frame

path = Path('data/raw/sign_language_conflict.xlsx')
print('exists=', path.exists())
if path.exists():
    df = load_dataset_frame(path)
    print('shape=', df.shape)
    print('columns=', list(df.columns))
    print(json.dumps(df.head(10).to_dict(orient='records'), ensure_ascii=False, indent=2))
else:
    print('Workbook not found at expected path')
