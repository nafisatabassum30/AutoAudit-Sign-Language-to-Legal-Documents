# convert_excel_to_csv.py
import pandas as pd

# Load Excel file
df = pd.read_excel('../data/raw/FinalSheet2.xlsx')

# Save as CSV
df.to_csv('../data/raw/FinalSheet2.csv', index=False, encoding='utf-8')

print("✅ Converted FinalSheet2.xlsx to FinalSheet2.csv")
print(f"📋 Rows: {len(df)}, Columns: {df.columns.tolist()}")