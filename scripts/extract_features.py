import pandas as pd
import os

INPUT_PATH = "data/india_firms_fire_data.csv"
OUTPUT_PATH = "data/fire_features.csv"

def extract_fire_counts(df):
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['region'] = pd.cut(df['longitude'], bins=[68, 75, 82, 89, 97], labels=['West', 'Central', 'East', 'Far East'])
    grouped = df.groupby(['acq_date', 'region']).size().reset_index(name='fire_count')
    return grouped

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    try:
        df = pd.read_csv(INPUT_PATH)
        features_df = extract_fire_counts(df)
        features_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Fire features saved to: {OUTPUT_PATH}")
        print(features_df.head())
    except Exception as e:
        print(f"⚠️ Error during feature extraction: {e}")

if __name__ == "__main__":
    main()
