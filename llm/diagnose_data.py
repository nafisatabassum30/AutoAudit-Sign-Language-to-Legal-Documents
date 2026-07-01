# diagnose_data.py
import pandas as pd

# Load the CSV
df = pd.read_csv('../data/raw/FinalSheet2.csv')

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check what the "Names" column contains
print(f"\nSample Names values:")
print(df['Names'].head(10).tolist())

# Check if any rows contain crime-related keywords
keywords = ['চুরি', 'মামলা', 'পুলিশ', 'হামলা', 'মারামারি', 'সংঘর্ষ', 'লুঠ', 'ডাকাতি', 'ছিনতাই', 'অপহরণ', 'হত্যা']

print(f"\nSearching for keywords: {keywords}")
count = 0
for idx, row in df.iterrows():
    text = str(row['Names'])
    if any(kw in text for kw in keywords):
        count += 1
        if count <= 5:
            print(f"  Found: {text}")

print(f"\nTotal rows containing keywords: {count}")