import sqlite3
import pandas as pd
import numpy as np
import os

DB_PATH = "data/wildfire_grid.db"
DISTRICT_MATRIX_PATH = "data/preprocessed/training_matrix_xgb_preprocessed.csv"

def rule_based_severity(row):
    score = 0
    if row["temperature"] > 32:
        score += 1
    if row["wind"] > 12:
        score += 1
    if row["rainfall"] < 2:
        score += 1
    if row["ndvi"] < 0.3:
        score += 1

    if score >= 3:
        return 2
    if score == 2:
        return 1
    return 0

def main():
    if not os.path.exists(DISTRICT_MATRIX_PATH):
        raise FileNotFoundError(DISTRICT_MATRIX_PATH)

    # Load district-level matrix
    df = pd.read_csv(DISTRICT_MATRIX_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["district"] = df["district"].str.title()
    df = df.sort_values(["district", "date"]).reset_index(drop=True)

    # Compute same-day severity
    df["severity_label"] = df.apply(rule_based_severity, axis=1)

    # Create D+2 labels
    df["severity_label_Dplus2"] = df.groupby("district")["severity_label"].shift(-2)
    df["fire_Dplus2"] = df.groupby("district")["surge_label"].shift(-2)

    df = df.dropna(subset=["severity_label_Dplus2", "fire_Dplus2"]).reset_index(drop=True)
    df["severity_label_Dplus2"] = df["severity_label_Dplus2"].astype(int)
    df["fire_Dplus2"] = df["fire_Dplus2"].astype(int)

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Load grid cells
    grid_cells = pd.read_sql_query("SELECT grid_id, district FROM grid_cells", conn)

    # Process district by district
    districts = df["district"].unique()
    print("Processing districts:", districts)

    for district in districts:
        print(f"\n➡ Processing district: {district}")

        # Grid cells in this district
        grid_subset = grid_cells[grid_cells["district"] == district]
        if grid_subset.empty:
            print(f"No grid cells found for {district}, skipping.")
            continue

        # District-level rows
        df_dist = df[df["district"] == district]

        # Insert features + labels for each date
        for _, row in df_dist.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")

            # Features
            features_tuple = (
                row.get("ndvi", None),
                row.get("temperature", None),
                row.get("rainfall", None),
                row.get("wind", None),
                row.get("dryness_index", None),
                int(row.get("thermal_flag", 0)) if "thermal_flag" in row else 0
            )

            # Labels
            fire_label = int(row["fire_Dplus2"])
            sev_label = int(row["severity_label_Dplus2"])

            # Insert for each grid cell in this district
            for _, g in grid_subset.iterrows():
                grid_id = g["grid_id"]

                cur.execute("""
                    INSERT OR REPLACE INTO grid_daily_features
                    (grid_id, date, ndvi, temperature, rainfall, wind, dryness_index, firs_hotspots)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (grid_id, date_str, *features_tuple))

                cur.execute("""
                    INSERT OR REPLACE INTO grid_labels
                    (grid_id, date, fire_Dplus2, severity_Dplus2)
                    VALUES (?, ?, ?, ?)
                """, (grid_id, date_str, fire_label, sev_label))

        conn.commit()
        print(f"✅ Finished district: {district}")

    conn.close()
    print("\n✅ All grid training data inserted successfully.")

if __name__ == "__main__":
    main()
