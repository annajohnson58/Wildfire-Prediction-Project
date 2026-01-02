import os
import sqlite3
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

DB_PATH = "data/wildfire_grid.db"


# ------------------------------------------------------------
# LOAD TRAINING DATA
# ------------------------------------------------------------
def load_training_data() -> pd.DataFrame:
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
    return df


# ------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["grid_id", "date"]).copy()

    # Rolling features
    df["rain_3d_sum"] = df.groupby("grid_id")["rainfall"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    df["rain_7d_sum"] = df.groupby("grid_id")["rainfall"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["temp_7d_mean"] = df.groupby("grid_id")["temperature"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["dry_7d_mean"] = df.groupby("grid_id")["dryness_index"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["ndvi_7d_mean"] = df.groupby("grid_id")["ndvi"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    # Lagged features
    df["hotspots_lag1"] = df.groupby("grid_id")["firs_hotspots"].shift(1).fillna(0)
    df["dry_lag1"] = df.groupby("grid_id")["dryness_index"].shift(1).fillna(0)
    df["temp_lag1"] = df.groupby("grid_id")["temperature"].shift(1).fillna(0)

    # Anomalies
    df["temp_anom"] = df["temperature"] - df["temp_7d_mean"]
    df["dry_anom"] = df["dryness_index"] - df["dry_7d_mean"]

    # District-level fire history
    df = df.sort_values(["district", "date"])
    df["fire_flag"] = (df["fire_Dplus2"] > 0).astype(int)

    df["dist_fire_7d"] = df.groupby("district")["fire_flag"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    df["dist_fire_30d"] = df.groupby("district")["fire_flag"].transform(
        lambda s: s.rolling(30, min_periods=1).sum()
    )

    df.drop(columns=["fire_flag"], inplace=True)
    return df


# ------------------------------------------------------------
# TIME-AWARE TRAIN/TEST SPLIT
# ------------------------------------------------------------
def time_aware_split(df: pd.DataFrame, test_frac: float = 0.2):
    df_sorted = df.sort_values(["date", "grid_id"]).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main():
    df = load_training_data()
    df = add_temporal_features(df)

    feature_cols = [
        "ndvi", "temperature", "rainfall", "wind", "dryness_index",
        "firs_hotspots",
        "rain_3d_sum", "rain_7d_sum",
        "temp_7d_mean", "dry_7d_mean", "ndvi_7d_mean",
        "hotspots_lag1", "dry_lag1", "temp_lag1",
        "temp_anom", "dry_anom",
        "dist_fire_7d", "dist_fire_30d",
    ]

    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    train_df, test_df = time_aware_split(df)

    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)

    y_fire_train = train_df["fire_Dplus2"].astype(int)
    y_fire_test = test_df["fire_Dplus2"].astype(int)

    y_sev_train = train_df["severity_Dplus2"].astype(int)
    y_sev_test = test_df["severity_Dplus2"].astype(int)

    # ------------------------------------------------------------
    # FIRE OCCURRENCE MODEL
    # ------------------------------------------------------------
    print("Training D+2 fire occurrence model (grid-based)...")

    n_neg = (y_fire_train == 0).sum()
    n_pos = (y_fire_train == 1).sum()
    scale_pos_weight = n_neg / max(1, n_pos)

    fire_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    fire_model.fit(X_train, y_fire_train)

    fire_proba = fire_model.predict_proba(X_test)[:, 1]
    fire_pred = (fire_proba > 0.3).astype(int)

    print("\nFire D+2 Model Evaluation (threshold=0.3):")
    print(classification_report(y_fire_test, fire_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_fire_test, fire_pred))

    # ------------------------------------------------------------
    # SEVERITY MODEL (FIRE-ONLY, BINARY: MODERATE vs HIGH)
    # ------------------------------------------------------------
    print("\nTraining D+2 severity model (fire-only)...")

    fire_mask_train = y_fire_train == 1

    # Initialize defaults
    sev_pred = np.zeros(len(X_test), dtype=int)
    sev_proba = np.zeros((len(X_test), 2))

    if fire_mask_train.sum() == 0:
        print("No fire-positive samples in training set. Skipping severity model.")
    else:
        # Remap severity: 1→0 (Moderate), 2→1 (High)
        y_sev_train_fire = y_sev_train[fire_mask_train].replace({1: 0, 2: 1})
        X_sev_train = X_train[fire_mask_train]

        sev_model = XGBClassifier(
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
        sev_model.fit(X_sev_train, y_sev_train_fire)

        # Predict on ALL test rows
        sev_proba = sev_model.predict_proba(X_test)

        # ✅ FIX: convert one-hot predictions to class indices
        sev_pred = np.argmax(sev_proba, axis=1)

    # ------------------------------------------------------------
    # SEVERITY EVALUATION (FIRE-POSITIVE ONLY)
    # ------------------------------------------------------------
    fire_mask_test = y_fire_test == 1

    if fire_mask_test.sum() > 0:
        print("\nSeverity D+2 Model Evaluation (fire-positive test rows):")

        # Remap true labels BEFORE filtering
        y_true_sev = y_sev_test.replace({1: 0, 2: 1})

        # Convert both to clean NumPy int arrays
        y_true_sev = y_true_sev[fire_mask_test].astype(int).to_numpy()
        y_pred_sev = sev_pred[fire_mask_test].astype(int)

        print(classification_report(
            y_true_sev,
            y_pred_sev,
            labels=[0, 1],
            target_names=["Moderate", "High"],
            zero_division=0,
        ))

        print("Severity Confusion Matrix:\n",
              confusion_matrix(
                  y_true_sev,
                  y_pred_sev,
                  labels=[0, 1]
              ))
    else:
        print("No fire-positive samples in test set for severity evaluation.")

    # ------------------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------------------
    print("\nSaving predictions...")

    df_test = test_df.copy()
    df_test["fire_prob"] = fire_proba

    inv_map = {0: "Moderate", 1: "High"}
    df_test["severity_pred"] = [inv_map.get(int(p), "Moderate") for p in sev_pred]

    df_test["prob_moderate"] = sev_proba[:, 0]
    df_test["prob_high"] = sev_proba[:, 1]
    df_test["prob_low"] = 0.0  # no low class in dataset

    df_test["forecast_date"] = df_test["date"] + pd.Timedelta(days=2)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    rows_pred = []
    for _, row in df_test.iterrows():
        rows_pred.append((
            row["grid_id"],
            row["date"].strftime("%Y-%m-%d"),
            row["forecast_date"].strftime("%Y-%m-%d"),
            float(row["fire_prob"]),
            row["severity_pred"],
            float(row["prob_low"]),
            float(row["prob_moderate"]),
            float(row["prob_high"]),
            "xgb_grid_v2",
        ))

    cur.executemany("""
        INSERT OR REPLACE INTO grid_predictions
        (grid_id, date, forecast_date, fire_prob, severity_pred,
         prob_low, prob_moderate, prob_high, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows_pred)

    conn.commit()
    conn.close()
    print("✅ Predictions saved to grid_predictions table.")

    # Export CSV
    os.makedirs("data/exports", exist_ok=True)
    export_df = df_test[[
        "grid_id", "lat", "lon", "district",
        "date", "forecast_date",
        "fire_prob", "severity_pred",
        "prob_low", "prob_moderate", "prob_high",
    ]]
    export_df.to_csv("data/exports/grid_predictions_with_severity_Dplus2_v2.csv", index=False)
    print("✅ Exported CSV for dashboard.")


if __name__ == "__main__":
    main()
