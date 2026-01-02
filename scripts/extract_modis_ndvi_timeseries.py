import ee
import json
import pandas as pd
from datetime import datetime, timedelta

ee.Initialize()

# Load grid GeoJSON
with open("data/geojson/kerala_forest_grid.geojson", encoding="utf-8") as f:

    grid_geojson = json.load(f)

# Convert to Earth Engine Features
features = [ee.Feature(ee.Geometry(f["geometry"]), {"grid_id": f["properties"]["grid_id"]}) for f in grid_geojson["features"]]
ee_features = ee.FeatureCollection(features)

# Define MODIS NDVI collection
ndvi_collection = ee.ImageCollection("MODIS/006/MOD13Q1").select("NDVI").map(
    lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"])
)

# Define time range
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 1, 16)

# Generate 16-day intervals
dates = []
d = start_date
while d <= end_date:
    dates.append(d)
    d += timedelta(days=16)

# Extract NDVI per grid cell per date
records = []
for date in dates:
    date_str = date.strftime("%Y-%m-%d")
    img = ndvi_collection.filterDate(date_str, (date + timedelta(days=16)).strftime("%Y-%m-%d")).mean()
    for f in grid_geojson["features"]:
        grid_id = f["properties"]["grid_id"]
        geom = ee.Geometry(f["geometry"])
        ndvi = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=250,
            maxPixels=1e9
        ).getInfo()
        records.append({
            "date": date_str,
            "grid_id": grid_id,
            "ndvi": ndvi.get("NDVI", None)
        })

# Save raw NDVI
df = pd.DataFrame(records)
df.to_csv("data/features/raw_ndvi_grid.csv", index=False)
print("✅ Raw NDVI saved: raw_ndvi_grid.csv")
