import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Paths
DB_PATH = "data/wildfire_grid.db"
FIRE_MODEL_PATH = "models/fire_model_v4.pkl"
SEVERITY_MODEL_PATH = "models/severity_model_v4.pkl"

os.makedirs("models", exist_ok=True)

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            f.grid_id, f.date,
            f.ndvi, f.temperature, f.rainfall, f.wind,
            f.dryness_index, f.firs_hotspots,
            f.elevation, f.slope, f.aspect,
            f.fire_label, f.severity_label
        FROM grid_daily_features_labeled f
        WHERE f.date BETWEEN '2022-01-01' AND '2024-12-31'
    """, conn)
    conn.close()
    return df

def preprocess(df):
    df = df.sort_values(["grid_id", "date"])
    df["dry_lag1"] = df.groupby("grid_id")["dryness_index"].shift(1).fillna(0)
    df["ndvi_lag1"] = df.groupby("grid_id")["ndvi"].shift(1).fillna(df["ndvi"].median())
    df["fire_label"] = df["fire_label"].astype(int)
    df["severity_label"] = df["severity_label"].fillna(1).astype(int)
    return df

def train_models(df):
    feature_cols = [
        "ndvi", "ndvi_lag1", "temperature", "rainfall", "wind",
        "dryness_index", "dry_lag1", "firs_hotspots",
        "elevation", "slope", "aspect"
    ]

    train_df = df[df["date"] < "2024-01-01"]
    test_df = df[df["date"] >= "2024-01-01"]

    X_train = train_df[feature_cols].fillna(0)
    y_fire_train = train_df["fire_label"]
    y_sev_train = train_df["severity_label"]

    X_test = test_df[feature_cols].fillna(0)
    y_fire_test = test_df["fire_label"]
    y_sev_test = test_df["severity_label"]

    # 🔥 Train Fire Model with imbalance handling
    pos_weight = (len(y_fire_train) - y_fire_train.sum()) / max(1, y_fire_train.sum())
    fire_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=pos_weight,
        n_jobs=-1
    )
    fire_model.fit(X_train, y_fire_train)

    fire_proba = fire_model.predict_proba(X_test)[:, 1]
    fire_pred = (fire_proba >= 0.3).astype(int)

    print("\n🔥 Fire D+2 Model Evaluation (threshold=0.3):")
    print(classification_report(y_fire_test, fire_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_fire_test, fire_pred))

    joblib.dump(fire_model, FIRE_MODEL_PATH)
    print(f"✅ Fire model saved to {FIRE_MODEL_PATH}")

    # ⚠️ Train Severity Model (fire-only)
    fire_mask_train = y_fire_train == 1
    fire_mask_test = y_fire_test == 1

    if fire_mask_train.sum() == 0:
        print("❌ No fire-positive samples in training set. Skipping severity model.")
        severity_model = None
    else:
        y_sev_train_fire = y_sev_train[fire_mask_train].replace({1: 0, 2: 1}).astype(int)
        X_sev_train = X_train[fire_mask_train]

        # Balance severity classes
        n_mod = (y_sev_train_fire == 0).sum()
        n_high = (y_sev_train_fire == 1).sum()
        total = n_mod + n_high
        w_mod = total / (2 * max(1, n_mod))
        w_high = total / (2 * max(1, n_high))
        sev_sample_weight = np.where(y_sev_train_fire == 0, w_mod, w_high)

        severity_model = XGBClassifier(
            objective="multi:softprob",
            num_class=2,
            n_estimators=400,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        severity_model.fit(X_sev_train, y_sev_train_fire, sample_weight=sev_sample_weight)

        # ✅ Predict severity probabilities and convert to class indices
        sev_proba = severity_model.predict_proba(X_test)
        sev_pred = np.argmax(sev_proba, axis=1)

        if fire_mask_test.sum() > 0:
            print("\n⚠️ Severity D+2 Model Evaluation (fire-positive test rows):")
            y_true_sev = y_sev_test.replace({1: 0, 2: 1})[fire_mask_test].astype(int).to_numpy()
            y_pred_sev = sev_pred[fire_mask_test].astype(int)

            print(classification_report(
                y_true_sev,
                y_pred_sev,
                labels=[0, 1],
                target_names=["Moderate", "High"],
                zero_division=0,
            ))
            print("Severity Confusion Matrix:\n", confusion_matrix(y_true_sev, y_pred_sev, labels=[0, 1]))
        else:
            print("⚠️ No fire-positive samples in test set for severity evaluation.")

        joblib.dump(severity_model, SEVERITY_MODEL_PATH)
        print(f"✅ Severity model saved to {SEVERITY_MODEL_PATH}")

def main():
    print("📥 Loading labeled data...")
    df = load_data()
    df = preprocess(df)
    train_models(df)

if __name__ == "__main__":
    main()
