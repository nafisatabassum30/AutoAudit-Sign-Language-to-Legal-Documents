import pandas as pd

# Try to load the Excel file
try:
    df = pd.read_excel('data/raw/FinalSheet2.xlsx')
    print(f"✅ Success! Loaded {len(df)} rows")
    print(f"📋 Columns: {df.columns.tolist()}")
    print("\n📝 First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("❌ File not found! Make sure FinalSheet2.xlsx is in data/raw/")
    print("   Current location: data/raw/FinalSheet2.xlsx")
except Exception as e:
    print(f"❌ Error: {e}")