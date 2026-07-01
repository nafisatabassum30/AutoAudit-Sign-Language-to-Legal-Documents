import pandas as pd
from pathlib import Path
import json

path = Path('data/raw/FinalSheet2.xlsx')
print('exists=', path.exists())
xl = pd.ExcelFile(path)
print('sheets=', xl.sheet_names)
for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print('---', sheet, 'shape=', df.shape)
    print(json.dumps(df.head(10).to_dict(orient='records'), ensure_ascii=False, indent=2))
