import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import os
import ee
import cdsapi
import xarray as xr
import zipfile

# ---------------------- CONFIG ----------------------
mode = "historical"  # change to "forecast" for future prediction
TODAY = datetime.utcnow().date()
NDVI_DATE = datetime(2025, 10, 10).date()
ERA5_DATE = datetime(2025, 10, 7).date()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------- Load Kerala Districts ----------------------
shapefile_path = os.path.join(DATA_DIR, "shapefiles", "kerala_districts.shp")
districts = gpd.read_file(shapefile_path)

# ---------------------- Initialize Earth Engine ----------------------
ee.Initialize()

def fetch_ndvi(geometry, date):
    try:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(str(date - timedelta(days=10)), str(date)) \
            .filterBounds(geometry) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
            .median()

        band_names = s2.bandNames().getInfo()
        if "B8" in band_names and "B4" in band_names:
            ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
            stats = ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            )
            result = stats.getInfo()
            ndvi_value = result.get("NDVI")
            if ndvi_value is None:
                print(f"⚠️ Sentinel-2 NDVI missing for geometry: {geometry.getInfo()}")
                return 0.0
            return ndvi_value
        else:
            print("⚠️ Sentinel-2 bands missing")
            return 0.0

    except Exception as e:
        print(f"⚠️ NDVI fetch failed: {e}")
        return 0.0

# ---------------------- Fetch ERA5 Historical (GRIB inside ZIP) ----------------------
def fetch_era5_historical():
    safe_hours = ['00:00']
    print(f"📅 Using ERA5 safe date: {ERA5_DATE} with hour(s): {safe_hours}")
    try:
        c = cdsapi.Client()
        zip_path = os.path.join(DATA_DIR, "era5_today.zip")
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature',
                    'total_precipitation',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind'
                ],
                'year': str(ERA5_DATE.year),
                'month': str(ERA5_DATE.month).zfill(2),
                'day': str(ERA5_DATE.day).zfill(2),
                'time': safe_hours,
                'area': [12.5, 74.8, 8.0, 78.5],
                'format': 'zip'
            },
            zip_path
        )

        if not zipfile.is_zipfile(zip_path):
            print("❌ File is not a valid ZIP archive.")
            raise SystemExit("🚫 Aborting: ERA5 file is not a ZIP.")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            all_files = zip_ref.namelist()
            grib_files = [f for f in all_files if f.endswith(".grib")]

            if not grib_files:
                print("❌ No .grib file found in ZIP.")
                print("📦 ZIP contents:", all_files)
                raise SystemExit("🚫 Aborting: ERA5 ZIP missing GRIB file.")

            grib_path = os.path.join(DATA_DIR, grib_files[0])

        ds = xr.open_dataset(grib_path, engine="cfgrib")
        return ds
    except Exception as e:
        print(f"❌ ERA5 fetch failed: {e}")
        raise SystemExit("🚫 Aborting: ERA5 fetch failed.")

# ---------------------- Fetch ECMWF Forecast ----------------------
def fetch_ecmwf_forecast():
    print(f"🔮 Using ECMWF forecast for {TODAY}")
    try:
        c = cdsapi.Client()
        output_path = os.path.join(DATA_DIR, "ecmwf_forecast.nc")
        c.retrieve(
            'ecmwf-forecast',
            {
                'variable': ['2m_temperature', 'total_precipitation', '10m_u_component_of_wind'],
                'product_type': 'ensemble_mean',
                'year': str(TODAY.year),
                'month': str(TODAY.month).zfill(2),
                'day': str(TODAY.day).zfill(2),
                'time': '00:00',
                'leadtime_hour': ['24'],
                'area': [12.5, 74.8, 8.0, 78.5],
                'format': 'netcdf'
            },
            output_path
        )
        ds = xr.open_dataset(output_path, engine="netcdf4")
        return ds
    except Exception as e:
        print(f"❌ ECMWF forecast fetch failed: {e}")
        return None

# ---------------------- Select Source ----------------------
weather_ds = fetch_ecmwf_forecast() if mode == "forecast" else fetch_era5_historical()

# ---------------------- Feature Engineering ----------------------
rows = []

for _, row in districts.iterrows():
    geom = row["geometry"]
    name = row["DISTRICT"]

    coords = list(geom.exterior.coords)
    ndvi = fetch_ndvi(ee.Geometry.Polygon([coords]), NDVI_DATE)

    if weather_ds is not None:
        temp = float(weather_ds["t2m"].mean().values) - 273.15
        rain = float(weather_ds["tp"].mean().values) * 1000
        wind_u = float(weather_ds["u10"].mean().values)
        wind_v = float(weather_ds["v10"].mean().values) if "v10" in weather_ds else 0.0
        wind = np.sqrt(wind_u**2 + wind_v**2)
    else:
        temp = 30.0
        rain = 5.0
        wind = 3.0
        print(f"⚠️ Using fallback weather values for {name}")

    row_data = {
        "district": name,
        "date": TODAY,
        "ndvi_mean": ndvi,
        "ndvi_std": 0.01,
        "ndvi_max": ndvi + 0.02,
        "ndvi_min": ndvi - 0.02,
        "ndvi_range": 0.04,
        "thermal_mean": temp,
        "thermal_std": 1.5,
        "thermal_max": temp + 2,
        "thermal_min": temp - 2,
        "thermal_range": 4,
        "rainfall_mean": rain,
        "rainfall_std": 2.0,
        "rainfall_max": rain + 5,
        "rainfall_min": max(0, rain - 5),
        "rainfall_range": 10,
        "wind_mean": wind,
        "wind_std": 1.0,
        "wind_max": wind + 2,
        "wind_min": max(0, wind - 2),
        "wind_range": 4,
        "thermal_lag": 0.0,
        "rainfall_lag": 0.0,
        "wind_lag": 0.0,
        "ndvi_lag": 0.0,
        "thermal_roll3": 0.0,
        "rainfall_roll3": 0.0,
        "wind_roll3": 0.0,
        "ndvi_roll3": 0.0,
        "thermal_ema3": temp
    }

    rows.append(row_data)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, "today_features.csv"), index=False)
print(f"✅ Feature matrix saved to {DATA_DIR}/today_features.csv using mode: {mode}")
