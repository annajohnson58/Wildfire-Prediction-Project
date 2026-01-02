
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ----------------------  Load Raw Sources ----------------------
era5_path = "data/daily_climate_district.csv"
ndvi_path = "data/ndvi/Kerala_District_Daily_NDVI_MODIS.csv"
thermal_path = "data/daily_predictions_full.csv"

climate = pd.read_csv(era5_path)
ndvi = pd.read_csv(ndvi_path)
thermal = pd.read_csv(thermal_path)

# ----------------------  Clean Columns ----------------------
climate["date"] = pd.to_datetime(climate["date"])
climate["district"] = climate["district"].str.strip().str.title()
climate["wind"] = np.sqrt(climate["wind_u"]**2 + climate["wind_v"]**2)

ndvi.columns = ndvi.columns.str.strip().str.lower()
ndvi = ndvi.rename(columns={"mean": "ndvi", "region": "district"} if "region" in ndvi.columns else {"mean": "ndvi"})
ndvi["date"] = pd.to_datetime(ndvi["date"])
ndvi["district"] = ndvi["district"].str.strip().str.title()

thermal.columns = thermal.columns.str.strip().str.lower()
thermal["date"] = pd.to_datetime(thermal["date"])
thermal["district"] = thermal["district"].str.strip().str.title()

# ----------------------  Merge All Sources ----------------------
df = climate.merge(ndvi[["date", "district", "ndvi"]], on=["date", "district"], how="outer")
df = df.merge(thermal[["date", "district", "thermal_count"]], on=["date", "district"], how="outer")
df = df.sort_values(["district", "date"])

# ----------------------  Fill Missing Values ----------------------
df = df.groupby("district").apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
df = df.dropna(subset=["district", "date"])

# ----------------------  Engineer Features ----------------------
df["ndvi_drop"] = df.groupby("district")["ndvi"].diff().fillna(0)
df["wind_surge"] = df.groupby("district")["wind"].diff().fillna(0)
df["rainfall_7d"] = df.groupby("district")["rainfall"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["rainfall_deficit"] = df["rainfall_7d"] - df["rainfall"]
df["thermal_lag"] = df.groupby("district")["thermal_count"].shift(1).fillna(0)

df["rainfall_3d"] = df.groupby("district")["rainfall"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["wind_3d"] = df.groupby("district")["wind"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["ndvi_3d"] = df.groupby("district")["ndvi"].transform(lambda x: x.rolling(3, min_periods=1).mean())

df["thermal_scaled"] = df.groupby("district")["thermal_count"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

df["surge_label"] = df.groupby("district")["thermal_count"].shift(-1).fillna(0)
df["surge_label"] = (df["surge_label"] >= df.groupby("district")["thermal_count"].transform("mean") + 2).astype(int)

df["dry_heat_index"] = df["temperature"] * df["rainfall_deficit"]
df["veg_stress_index"] = df["ndvi_drop"] * df["wind"]

# ----------------------  Clip Outliers ----------------------
clip_cols = ["rainfall", "thermal_count", "temperature", "wind"]
for col in clip_cols:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = df[col].clip(low, high)

# ----------------------  Normalize Within District ----------------------
norm_cols = ["ndvi", "rainfall", "temperature", "wind", "thermal_count"]
for col in norm_cols:
    df[col + "_norm"] = df.groupby("district")[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

# ----------------------  Scale Engineered Features ----------------------
scale_cols = [
    "ndvi_drop", "thermal_lag", "rainfall_deficit", "wind_surge",
    "dry_heat_index", "veg_stress_index", "thermal_scaled",
    "rainfall_3d", "wind_3d", "ndvi_3d"
]
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ----------------------  Add Lag and Rolling Features ----------------------
df["ndvi_lag1"] = df.groupby("district")["ndvi"].shift(1)
df["thermal_lag1"] = df.groupby("district")["thermal_count"].shift(1)
df["rainfall_lag1"] = df.groupby("district")["rainfall"].shift(1)

df["ndvi_roll3"] = df.groupby("district")["ndvi"].rolling(3).mean().reset_index(level=0, drop=True)
df["thermal_roll3"] = df.groupby("district")["thermal_count"].rolling(3).mean().reset_index(level=0, drop=True)
df["thermal_ema3"] = df.groupby("district")["thermal_count"].ewm(span=3).mean().reset_index(drop=True)

# ----------------------  Add Calendar Features ----------------------
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

# ----------------------  Final Cleanup ----------------------
df.dropna(inplace=True)

# ----------------------  District Coverage Check ----------------------
expected_districts = [
    "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha",
    "Kottayam", "Idukki", "Ernakulam", "Thrissur",
    "Palakkad", "Malappuram", "Kozhikode", "Wayanad",
    "Kannur", "Kasaragod"
]
actual_districts = sorted(df["district"].unique())
missing = sorted(set(expected_districts) - set(actual_districts))

print(f"\n Districts in matrix: {len(actual_districts)} → {actual_districts}")
if missing:
    print(f" Missing districts: {missing}")
else:
    print(" All 14 districts are present in the training matrix.")

# ----------------------  Inject Missing Districts ----------------------
for district in missing:
    print(f" Injecting placeholder for {district}")
    template = df[df["district"] == "Thrissur"].tail(14).copy()
    template["district"] = district
    template["thermal_count"] = 0
    template["surge_label"] = 0
    template[norm_cols + scale_cols] = 0
    df = pd.concat([df, template], ignore_index=True)

# ----------------------  Save Output ----------------------
os.makedirs("data/preprocessed", exist_ok=True)
df.to_csv("data/preprocessed/training_matrix_xgb_preprocessed.csv", index=False)
print(" Saved: data/preprocessed/training_matrix_xgb_preprocessed.csv")
