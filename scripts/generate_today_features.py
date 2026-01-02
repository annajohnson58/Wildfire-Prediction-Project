import pandas as pd
import numpy as np
import os

# ---------------------- Load Raw Data ----------------------
era5 = pd.read_csv("data/daily_climate_district.csv")
ndvi = pd.read_csv("data/ndvi/Kerala_District_Daily_NDVI_MODIS.csv")
thermal = pd.read_csv("data/daily_predictions_full.csv")

# ---------------------- Clean Column Names ----------------------
for df in [era5, ndvi, thermal]:
    df.columns = df.columns.str.strip().str.lower()

# ---------------------- Fix NDVI Columns ----------------------
if "region" in ndvi.columns:
    ndvi = ndvi.rename(columns={"region": "district"})
elif "district" not in ndvi.columns:
    raise ValueError("❌ NDVI file missing 'district' or 'region' column.")

if "mean" in ndvi.columns:
    ndvi = ndvi.rename(columns={"mean": "ndvi"})
elif "ndvi" not in ndvi.columns:
    raise ValueError("❌ NDVI file missing 'mean' or 'ndvi' column.")

# ---------------------- Standardize District Names ----------------------
for df in [era5, ndvi, thermal]:
    df["district"] = df["district"].str.strip().str.title()

# ---------------------- Convert Dates ----------------------
for df in [era5, ndvi, thermal]:
    df["date"] = pd.to_datetime(df["date"])

# ---------------------- Find Latest Common Date ----------------------
common_dates = set(era5["date"]).intersection(ndvi["date"]).intersection(thermal["date"])
if not common_dates:
    raise ValueError("❌ No common date found across ERA5, NDVI, and thermal datasets.")

latest_common_date = max(common_dates)
print(f"\n📅 Using latest common date: {latest_common_date.date()}")

era5_today = era5[era5["date"] == latest_common_date]
ndvi_today = ndvi[ndvi["date"] == latest_common_date]
thermal_today = thermal[thermal["date"] == latest_common_date]

print(f"\n📊 Row counts on {latest_common_date.date()}:")
print("ERA5:", len(era5_today))
print("NDVI:", len(ndvi_today))
print("Thermal:", len(thermal_today))

print("\n📍 Districts per source:")
print("ERA5:", sorted(era5_today["district"].unique()))
print("NDVI:", sorted(ndvi_today["district"].unique()))
print("Thermal:", sorted(thermal_today["district"].unique()))

# ---------------------- Merge Sources ----------------------
df = era5_today.merge(ndvi_today[["date", "district", "ndvi"]], on=["date", "district"], how="outer")
df = df.merge(thermal_today[["date", "district", "thermal_count"]], on=["date", "district"], how="outer")

print(f"\n🧮 Rows after merge: {df.shape[0]}")
print("🧪 Sample rows:")
print(df.head())

# ---------------------- Fill Missing ERA5 Values ----------------------
era5_means = era5.groupby("district")[["temperature", "rainfall", "wind_u", "wind_v"]].mean()

for district in df["district"].unique():
    if district in era5_means.index:
        for col in ["temperature", "rainfall", "wind_u", "wind_v"]:
            df.loc[df["district"] == district, col] = df.loc[df["district"] == district, col].fillna(era5_means.loc[district, col])
    else:
        print(f"⚠️ ERA5 missing for {district} — using global mean")
        for col in ["temperature", "rainfall", "wind_u", "wind_v"]:
            df[col] = df[col].fillna(df[col].mean())

print("\n✅ Filled missing ERA5 values for districts with NDVI and thermal data.")

# ---------------------- Engineer Features ----------------------
df["wind"] = (df["wind_u"]**2 + df["wind_v"]**2)**0.5
df["thermal_scaled"] = (df["thermal_count"] - df["thermal_count"].min()) / (df["thermal_count"].max() - df["thermal_count"].min() + 1e-6)

# Global normalization across districts
for col in ["temperature", "rainfall", "wind", "ndvi", "thermal_count"]:
    norm_col = col + "_norm"
    df[norm_col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

# Add calendar features
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

# ---------------------- Final Cleanup ----------------------
df = df.sort_values(["district"])

# Check if any valid rows exist before dropping
if df[["district", "date"]].dropna().empty:
    raise ValueError("❌ Final merged DataFrame is empty. Check source coverage and district overlap.")

# Drop rows missing critical inputs only
df.dropna(subset=["temperature", "rainfall", "wind_u", "wind_v", "ndvi", "thermal_count"], inplace=True)

# ---------------------- Save Output ----------------------
os.makedirs("data", exist_ok=True)
df.to_csv("data/today_features.csv", index=False)
print("✅ Saved: data/today_features.csv")
