import sqlite3
import pandas as pd

DB_PATH = "data/wildfire_grid.db"

def main():
    print("🔄 Connecting to database...")
    conn = sqlite3.connect(DB_PATH)

    print("📥 Loading grid features...")
    features = pd.read_sql_query("SELECT * FROM grid_daily_features", conn)
    print(f"✅ Loaded {len(features)} grid-day records")

    print("🔥 Loading fire labels from grid_labels...")
    labels = pd.read_sql_query("""
        SELECT grid_id, date, fire_Dplus2 AS fire_label, severity_Dplus2 AS severity_label
        FROM grid_labels
    """, conn)

    print("🔗 Merging labels into features...")
    merged = features.merge(
        labels,
        on=["grid_id", "date"],
        how="left"
    )
    merged["fire_label"] = merged["fire_label"].fillna(0).astype(int)
    merged["severity_label"] = merged["severity_label"].fillna(0).astype(int)

    print("💾 Writing labeled data to new table: grid_daily_features_labeled")
    merged.to_sql("grid_daily_features_labeled", conn, if_exists="replace", index=False)

    print("✅ Done. You can now train models using grid_daily_features_labeled")
    conn.close()

if __name__ == "__main__":
    main()
