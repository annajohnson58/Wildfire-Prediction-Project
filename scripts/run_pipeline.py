import sqlite3
import pandas as pd

# Paths
DB_PATH = "data/wildfire_grid.db"
CSV_PATH = "data/grid_features_realtime.csv"

def add_columns():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for col in ["elevation", "slope", "aspect"]:
        try:
            cur.execute(f"ALTER TABLE grid_daily_features_labeled ADD COLUMN {col} REAL;")
            print(f"✅ Added column: {col}")
        except sqlite3.OperationalError:
            print(f"ℹ️ Column {col} already exists, skipping.")

    conn.commit()
    conn.close()

def backfill_from_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Load enriched CSV with terrain features
    df = pd.read_csv(CSV_PATH)
    df = df[["grid_id", "elevation", "slope", "aspect"]].drop_duplicates()

    # Update all rows for each grid_id with static terrain values
    for _, row in df.iterrows():
        cur.execute("""
            UPDATE grid_daily_features_labeled
            SET elevation = ?, slope = ?, aspect = ?
            WHERE grid_id = ?;
        """, (row["elevation"], row["slope"], row["aspect"], row["grid_id"]))

    conn.commit()
    conn.close()
    print("✅ Backfill complete: terrain features updated for all historical rows")

def main():
    print("📥 Adding terrain columns...")
    add_columns()
    print("📥 Backfilling terrain values from CSV...")
    backfill_from_csv()

if __name__ == "__main__":
    main()
