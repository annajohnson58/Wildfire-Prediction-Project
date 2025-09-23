from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

# ---------------------- Load and Preprocess Dataset ----------------------
df = pd.read_csv("data/training_matrix.csv")
df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)

# ---------------------- Feature Engineering ----------------------
df["ndvi_lag1"] = df["ndvi"].shift(1)
df["ndvi_lag2"] = df["ndvi"].shift(2)
df["thermal_lag1"] = df["thermal_count"].shift(1)
df["thermal_lag3"] = df["thermal_count"].shift(3)
df["rh_lag1"] = df["rh"].shift(1)
df["rh_lag5"] = df["rh"].shift(5)

df["ndvi_roll3"] = df["ndvi"].rolling(3).mean()
df["ndvi_roll5"] = df["ndvi"].rolling(5).mean()
df["thermal_roll3"] = df["thermal_count"].rolling(3).mean()
df["thermal_ema3"] = df["thermal_count"].ewm(span=3).mean()

df["dry_heat_index"] = df["rh_anomaly"] * df["temperature"]
df["veg_stress_index"] = df["ndvi_drop"] * df["wind"]

df.dropna(inplace=True)

# ---------------------- Create Target Column ----------------------
df["fire_flag"] = (df["thermal_spike"] > 0).astype(int)

# ---------------------- Define Features and Target ----------------------
features = [
    "ndvi", "ndvi_drop", "temperature", "rh", "rh_anomaly", "wind", "rainfall",
    "ndvi_lag1", "ndvi_lag2", "thermal_lag1", "thermal_lag3", "rh_lag1", "rh_lag5",
    "ndvi_roll3", "ndvi_roll5", "thermal_roll3", "thermal_ema3",
    "dry_heat_index", "veg_stress_index"
]
target = "fire_flag"

missing = set(features) - set(df.columns)
if missing:
    print("Missing features:", missing)
    exit()

# ---------------------- Rolling Forecast Simulation ----------------------
initial_window = 60
results = []

for i in range(initial_window, len(df) - 1):
    train_df = df.iloc[:i]
    test_df = df.iloc[i:i+1]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Skip iteration if training set has only one class
    if len(np.unique(y_train)) < 2:
        continue

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    counts = np.bincount(y_train)
    if len(counts) == 1:
        neg, pos = counts[0], 0
        ratio = 1
    else:
        neg, pos = counts
        ratio = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=ratio,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba > 0.3).astype(int)

    results.append({
        "date": test_df["date"].values[0],
        "actual": int(y_test.values[0]),
        "predicted": int(y_pred[0]),
        "probability": float(y_proba[0])
    })

# ---------------------- Save Results ----------------------
results_df = pd.DataFrame(results)
results_df.to_csv("data/rolling_predictions.csv", index=False)
print("✅ Rolling forecast simulation completed and saved.")
