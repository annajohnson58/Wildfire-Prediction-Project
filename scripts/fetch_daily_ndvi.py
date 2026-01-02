import pandas as pd

ndvi_raw = pd.read_csv("data/Kerala_District_Daily_NDVI_MODIS.csv")

# Rename and format
ndvi = ndvi_raw.rename(columns={
    "DISTRICT": "district",
    "date": "date",
    "mean": "ndvi"
})

# Convert date to datetime
ndvi["date"] = pd.to_datetime(ndvi["date"], format="%Y-%m-%d")

# Drop unnecessary columns
ndvi = ndvi[["date", "district", "ndvi"]]
# Check daily coverage per district
coverage = ndvi.groupby("district")["date"].nunique()
print("🧠 Days covered per district:\n", coverage)

# Optional: Pivot to wide format for dashboard
ndvi_pivot = ndvi.pivot(index="date", columns="district", values="ndvi")
ndvi.to_csv("data/ndvi_daily_district.csv", index=False)
