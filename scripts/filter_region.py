import pandas as pd
import os

# === CONFIG ===
INPUT_PATH = "data/processed_firms_fire_data.csv"  # Adjust if needed
OUTPUT_PATH = "data/india_firms_fire_data.csv"

# === BOUNDING BOX FOR INDIA ===
LAT_MIN, LAT_MAX = 6.0, 37.0
LON_MIN, LON_MAX = 68.0, 97.0

def filter_india_fires(df):
    return df[
        (df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
        (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)
    ]

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    try:
        df = pd.read_csv(INPUT_PATH)
        india_df = filter_india_fires(df)
        india_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Filtered fire data saved to: {OUTPUT_PATH}")
        print(f"📊 Total records: {len(india_df)}")
    except Exception as e:
        print(f"⚠️ Error during filtering: {e}")

if __name__ == "__main__":
    main()
