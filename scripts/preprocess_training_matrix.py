import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# 📂 Load fused matrix
df = pd.read_csv("data/training_matrix_xgb.csv")
df["date"] = pd.to_datetime(df["date"])
df["district"] = df["district"].str.title()

# ---------------------- 🧹 Clip Outliers ----------------------
def clip_outliers(df, cols, lower=0.01, upper=0.99):
    for col in cols:
        low = df[col].quantile(lower)
        high = df[col].quantile(upper)
        df[col] = df[col].clip(low, high)
    return df

clip_cols = ["rainfall", "thermal_count", "temperature", "wind"]
df = clip_outliers(df, clip_cols)

# ---------------------- 🔁 Normalize Within District ----------------------
norm_cols = ["ndvi", "rainfall", "temperature", "wind", "thermal_count"]
for col in norm_cols:
    df[col + "_norm"] = df.groupby("district")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

# ---------------------- 📊 Scale Engineered Features ----------------------
scale_cols = [
    "ndvi_drop", "thermal_lag", "rainfall_deficit", "wind_surge",
    "dry_heat_index", "veg_stress_index", "thermal_scaled",
    "rainfall_3d", "wind_3d", "ndvi_3d"
]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ---------------------- 🔁 Add Lag and Rolling Features ----------------------
df["ndvi_lag1"] = df.groupby("district")["ndvi"].shift(1)
df["thermal_lag1"] = df.groupby("district")["thermal_count"].shift(1)
df["rainfall_lag1"] = df.groupby("district")["rainfall"].shift(1)

df["ndvi_roll3"] = df.groupby("district")["ndvi"].rolling(3).mean().reset_index(level=0, drop=True)
df["thermal_roll3"] = df.groupby("district")["thermal_count"].rolling(3).mean().reset_index(level=0, drop=True)
df["thermal_ema3"] = df.groupby("district")["thermal_count"].ewm(span=3).mean().reset_index(drop=True)

# ---------------------- 🧹 Final Cleanup ----------------------
df.dropna(inplace=True)

# ---------------------- 💾 Save Preprocessed Matrix ----------------------
os.makedirs("data/preprocessed", exist_ok=True)
df.to_csv("data/preprocessed/training_matrix_xgb_preprocessed.csv", index=False)
print("✅ Preprocessed matrix saved to data/preprocessed/training_matrix_xgb_preprocessed.csv")