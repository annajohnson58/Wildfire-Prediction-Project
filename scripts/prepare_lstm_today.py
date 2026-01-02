import pandas as pd
import numpy as np
import os

# ---------------------- Config ----------------------
N_DAYS = 14
FEATURES = [
    "temperature", "rainfall", "wind", "ndvi", "thermal_count",
    "thermal_scaled", "temperature_norm", "rainfall_norm",
    "wind_norm", "ndvi_norm", "thermal_count_norm",
    "month", "dayofweek"
]
SOURCE = "data/preprocessed/training_matrix_xgb_preprocessed.csv"

# ---------------------- Load Historical Matrix ----------------------
df = pd.read_csv(SOURCE)
df["date"] = pd.to_datetime(df["date"])
df["district"] = df["district"].str.strip().str.title()
df = df.sort_values(["district", "date"])

# ---------------------- Load Today's Districts ----------------------
today_df = pd.read_csv("data/today_features.csv")
today_df.columns = today_df.columns.str.strip().str.lower()

if "district" not in today_df.columns:
    raise ValueError("❌ 'district' column missing in today_features.csv")

today_df["district"] = today_df["district"].str.strip().str.title()
districts_today = today_df["district"].unique()

print("\n📋 Districts in today_features.csv:")
print(districts_today)

print("\n📋 Districts in historical matrix:")
print(df["district"].unique())

# ---------------------- Build LSTM Input ----------------------
X_today = []

for district in districts_today:
    district_df = df[df["district"] == district].copy()
    district_df = district_df.sort_values("date")

    if len(district_df) < N_DAYS:
        print(f"⚠️ Skipping {district}: not enough history")
        continue

    last_n = district_df.iloc[-N_DAYS:][FEATURES].values
    if last_n.shape != (N_DAYS, len(FEATURES)):
        print(f"⚠️ Skipping {district}: shape mismatch {last_n.shape}")
        continue

    X_today.append(last_n)
    print(f"✅ {district}: using last {N_DAYS} days")

X_today = np.array(X_today)

print(f"\n✅ Shape of X_lstm_today: {X_today.shape}")
np.save("data/X_lstm_today.npy", X_today)
print("💾 Saved: data/X_lstm_today.npy")
