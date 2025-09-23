import pandas as pd
import numpy as np

# ---------------------- Load NDVI Data ----------------------
ndvi = pd.read_csv("data/ndvi/Sentinel2_NDVI_Kerala_2022_2025.csv")
ndvi.columns = ndvi.columns.str.strip().str.lower()
ndvi.rename(columns={"month": "date"}, inplace=True)
ndvi["date"] = pd.to_datetime(ndvi["date"])
ndvi["district"] = ndvi["district"].str.title()
ndvi = ndvi[["date", "district", "ndvi"]]

# ---------------------- Load FIRMS VIIRS Data ----------------------
firms = pd.read_csv("data/processed/firms_with_districts.csv")
firms.columns = firms.columns.str.strip().str.lower()
firms["acq_date"] = pd.to_datetime(firms["acq_date"])
firms["district"] = firms["district"].str.title()

# Aggregate fire detections per district per day
thermal = firms.groupby(["acq_date", "district"]).size().reset_index(name="thermal_count")
thermal.rename(columns={"acq_date": "date"}, inplace=True)

# ---------------------- Load ERA5 Weather Data ----------------------
df_weather = pd.read_csv("data/processed/era5_with_districts.csv")
df_weather.columns = df_weather.columns.str.strip().str.lower()

# Rename columns to match expected schema
df_weather.rename(columns={
    "time": "date",
    "t2m": "temperature",
    "d2m": "rh",
    "tp": "rainfall"
}, inplace=True)

# Compute wind magnitude from u10 and v10
df_weather["wind"] = np.sqrt(df_weather["u10"]**2 + df_weather["v10"]**2)

# Format date and district
df_weather["date"] = pd.to_datetime(df_weather["date"])
df_weather["district"] = df_weather["district"].str.title()

# Ensure numeric columns before aggregation
numeric_cols = ["temperature", "rh", "wind", "rainfall"]
df_weather[numeric_cols] = df_weather[numeric_cols].apply(pd.to_numeric, errors="coerce")
df_weather = df_weather.dropna(subset=["date", "district"])

# Aggregate weather variables
weather = df_weather.groupby(["date", "district"])[numeric_cols].mean().reset_index()

# ---------------------- Merge All Datasets ----------------------
df = ndvi.merge(thermal, on=["date", "district"], how="outer")
df = df.merge(weather, on=["date", "district"], how="outer")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["district", "date"])

# ---------------------- Fill Missing Values ----------------------
df = df.groupby("district").apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

# ---------------------- Feature Engineering ----------------------
df["ndvi_7day_avg"] = df.groupby("district")["ndvi"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["ndvi_drop"] = df["ndvi"] - df["ndvi_7day_avg"]

df["thermal_3day_avg"] = df.groupby("district")["thermal_count"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["thermal_spike"] = df["thermal_count"] - df["thermal_3day_avg"]

df["rh_30day_avg"] = df.groupby("district")["rh"].transform(lambda x: x.rolling(30, min_periods=1).mean())
df["rh_anomaly"] = df["rh"] - df["rh_30day_avg"]

# ---------------------- Final Cleanup ----------------------
df = df.fillna(0)
df = df.drop(columns=["ndvi_7day_avg", "thermal_3day_avg", "rh_30day_avg"])

# ---------------------- Save Output ----------------------
df.to_csv("data/training_matrix.csv", index=False)
print(" training_matrix.csv created successfully.")
