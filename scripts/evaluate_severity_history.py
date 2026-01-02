import sqlite3
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

DB_PATH = "data/wildfire_grid.db"


def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            g.grid_id, g.lat, g.lon, g.district,
            f.date,
            f.ndvi, f.temperature, f.rainfall, f.wind,
            f.dryness_index, f.firs_hotspots,
            l.fire_Dplus2, l.severity_Dplus2
        FROM grid_daily_features f
        JOIN grid_labels l
          ON f.grid_id = l.grid_id AND f.date = l.date
        JOIN grid_cells g
          ON g.grid_id = f.grid_id
        ORDER BY f.date, g.grid_id
    """, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def add_features(df):
    df = df.sort_values(["grid_id", "date"]).copy()

    df["rain_7d_sum"] = df.groupby("grid_id")["rainfall"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["temp_7d_mean"] = df.groupby("grid_id")["temperature"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["dry_7d_mean"] = df.groupby("grid_id")["dryness_index"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    df["hotspots_lag1"] = df.groupby("grid_id")["firs_hotspots"].shift(1).fillna(0)
    df["dry_lag1"] = df.groupby("grid_id")["dryness_index"].shift(1).fillna(0)

    df["temp_anom"] = df["temperature"] - df["temp_7d_mean"]
    df["dry_anom"] = df["dryness_index"] - df["dry_7d_mean"]

    return df


def main():
    df = load_data()
    df = add_features(df)

    # ✅ Focus only on years where High severity exists
    train_year = 2022
    test_year = 2023

    train_df = df[df["year"] == train_year].copy()
    test_df = df[df["year"] == test_year].copy()

    print(f"[INFO] Training on {train_year}: {len(train_df)} rows")
    print(f"[INFO] Testing on {test_year}: {len(test_df)} rows")
    print(f"[INFO] High severity in test: {(test_df['severity_Dplus2'] == 2).sum()}")

    feature_cols = [
        "ndvi", "temperature", "rainfall", "wind", "dryness_index",
        "firs_hotspots",
        "rain_7d_sum", "temp_7d_mean", "dry_7d_mean",
        "hotspots_lag1", "dry_lag1",
        "temp_anom", "dry_anom",
    ]

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)

    y_train = train_df["severity_Dplus2"].replace({1: 0, 2: 1}).astype(int)
    y_test = test_df["severity_Dplus2"].replace({1: 0, 2: 1}).astype(int)

    # ✅ Class weights to emphasize High severity
    n_mod = (y_train == 0).sum()
    n_high = (y_train == 1).sum()
    total = n_mod + n_high
    w_mod = total / (2 * max(1, n_mod))
    w_high = total / (2 * max(1, n_high))
    sample_weight = np.where(y_train == 0, w_mod, w_high)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=2,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining severity model on historical data...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba = model.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)

    print("\nHistorical Severity Evaluation:")
    print(classification_report(
        y_test,
        pred,
        labels=[0, 1],
        target_names=["Moderate", "High"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        pred,
        labels=[0, 1]
    ))


if __name__ == "__main__":
    main()
