# import os
# import pandas as pd
# import geopandas as gpd
# import requests
# from shapely.geometry import Point
# from datetime import datetime, timedelta
# import ee

# # === CONFIG ===
# FIRMS_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InJpeWFyb3NlbWoiLCJleHAiOjE3NzIyOTE5MjcsImlhdCI6MTc2NzEwNzkyNywiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.jz8MNb0lqJa5ax97Z17Wt3NXyQ2K6jhIErqoGTjuDQMQHfGxY51UthpYFd1017Ey6jxKjDuuYx7boCcxudp0nz5YFUfu_BYss4b3yJds047NHdwHRLHsQ9UASQ0Fb-zbfL6ArjeABvfJvl2yW3MBliy3v_j-2HectUfx-yyLfhEJ36ZuOJ0csQB19KNz8zd8pPfrukbLU05Mj1cywMEvWvywSHWtKCiB0Kqdoahbb7yj3zzDMcKBsM5FStbEtxCjPJbwX_m8l7lYY5-JkAaxOOfoffAowpsfnkfKGUAahw6oWTbYvqGl9d3qdSlv-xQxGjritvn3VOYstdOLOM5Dqg"  # Replace with your NASA FIRMS token
# FIRMS_COUNTRY = "IND"  # ISO country code
# FIRMS_DAYS = 3
# GRID_CSV = "data/grid_locations.csv"
# OUTPUT_CSV = "data/grid_features_realtime.csv"

# # === Load Grid Locations ===
# df = pd.read_csv(GRID_CSV)
# df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
# grids = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# # === Fetch FIRMS Fire Hotspots ===
# print("🔥 Fetching FIRMS fire detections...")
# firms_url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_TOKEN}/VIIRS_SNPP_NRT/{FIRMS_COUNTRY}/{FIRMS_DAYS}"
# fires = pd.read_csv(firms_url)
# fires["geometry"] = fires.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
# fires_gdf = gpd.GeoDataFrame(fires, geometry="geometry", crs="EPSG:4326")

# # === Count Hotspots per Grid ===
# print("📍 Counting fire hotspots per grid...")
# joined = gpd.sjoin(grids, fires_gdf, how="left", predicate="contains")
# fire_counts = joined.groupby("grid_id").size().reset_index(name="firs_hotspots")
# grids = grids.merge(fire_counts, on="grid_id", how="left")
# grids["firs_hotspots"] = grids["firs_hotspots"].fillna(0).astype(int)

# # === Fetch NDVI from Google Earth Engine ===
# print("🌿 Fetching NDVI from GEE...")
# ee.Initialize()
# ndvi_img = ee.ImageCollection("MODIS/006/MOD13Q1").select("NDVI").sort("system:time_start", False).first()

# def get_ndvi(lat, lon):
#     try:
#         point = ee.Geometry.Point([lon, lat])
#         val = ndvi_img.reduceRegion(
#             reducer=ee.Reducer.mean(),
#             geometry=point,
#             scale=250
#         ).get("NDVI")
#         return val.getInfo()
#     except:
#         return None

# grids["ndvi"] = grids.apply(lambda row: get_ndvi(row["lat"], row["lon"]), axis=1)

# # === Final Cleanup and Save ===
# grids[["grid_id", "lat", "lon", "ndvi", "firs_hotspots"]].to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Real-time features saved to {OUTPUT_CSV}")

# import os
# import pandas as pd
# import geopandas as gpd
# import requests
# from shapely.geometry import Point
# from datetime import datetime, timedelta
# import ee

# # === CONFIG ===
# FIRMS_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6InJpeWFyb3NlbWoiLCJleHAiOjE3NzIyOTE5MjcsImlhdCI6MTc2NzEwNzkyNywiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.jz8MNb0lqJa5ax97Z17Wt3NXyQ2K6jhIErqoGTjuDQMQHfGxY51UthpYFd1017Ey6jxKjDuuYx7boCcxudp0nz5YFUfu_BYss4b3yJds047NHdwHRLHsQ9UASQ0Fb-zbfL6ArjeABvfJvl2yW3MBliy3v_j-2HectUfx-yyLfhEJ36ZuOJ0csQB19KNz8zd8pPfrukbLU05Mj1cywMEvWvywSHWtKCiB0Kqdoahbb7yj3zzDMcKBsM5FStbEtxCjPJbwX_m8l7lYY5-JkAaxOOfoffAowpsfnkfKGUAahw6oWTbYvqGl9d3qdSlv-xQxGjritvn3VOYstdOLOM5Dqg"  # Replace with your actual FIRMS token
# FIRMS_COUNTRY = "IND"  # ISO country code
# FIRMS_DAYS = 3
# GRID_CSV = "data/grid_locations.csv"
# OUTPUT_CSV = "data/grid_features_realtime.csv"

# # === Load Grid Locations ===
# df = pd.read_csv(GRID_CSV)
# df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
# grids = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# # === Fetch FIRMS Fire Hotspots ===
# print("🔥 Fetching FIRMS fire detections...")
# firms_url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_TOKEN}/VIIRS_SNPP_NRT/{FIRMS_COUNTRY}/{FIRMS_DAYS}"
# fires = pd.read_csv(firms_url)
# fires["geometry"] = fires.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
# fires_gdf = gpd.GeoDataFrame(fires, geometry="geometry", crs="EPSG:4326")

# # === Count Hotspots per Grid with Accurate Buffering ===
# print("📍 Counting fire hotspots per grid...")

# # Reproject to UTM Zone 43N for Kerala (EPSG:32643)
# grids_proj = grids.to_crs(epsg=32643)
# fires_proj = fires_gdf.to_crs(grids_proj.crs)

# # Buffer each grid point by 1000 meters (1 km)
# grids_proj["geometry"] = grids_proj["geometry"].buffer(1000)

# # Spatial join: which fire points fall within each grid buffer
# joined = gpd.sjoin(fires_proj, grids_proj, how="left", predicate="within")

# # Count fires per grid
# fire_counts = joined.groupby("grid_id").size().reset_index(name="firs_hotspots")

# # Merge counts back to original grid
# grids = grids.merge(fire_counts, on="grid_id", how="left")
# grids["firs_hotspots"] = grids["firs_hotspots"].fillna(0).astype(int)

# # === Fetch NDVI from Google Earth Engine ===
# print("🌿 Fetching NDVI from GEE...")
# ee.Initialize()
# ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI").sort("system:time_start", False).first()

# def get_ndvi(lat, lon):
#     try:
#         point = ee.Geometry.Point([lon, lat])
#         val = ndvi_img.reduceRegion(
#             reducer=ee.Reducer.mean(),
#             geometry=point,
#             scale=250
#         ).get("NDVI")
#         return val.getInfo()
#     except:
#         return None

# grids["ndvi"] = grids.apply(lambda row: get_ndvi(row["lat"], row["lon"]), axis=1)

# # === Final Cleanup and Save ===
# grids[["grid_id", "lat", "lon", "ndvi", "firs_hotspots"]].to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Real-time features saved to {OUTPUT_CSV}")

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import ee

# === CONFIG ===
FIRMS_TOKEN = "YOUR_FIRMS_TOKEN"  # Replace with your actual FIRMS token
FIRMS_COUNTRY = "IND"  # ISO country code
FIRMS_DAYS = 3
GRID_CSV = "data/grid_locations.csv"
OUTPUT_CSV = "data/grid_features_realtime.csv"

# === Load Grid Locations ===
df = pd.read_csv(GRID_CSV)
df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
grids = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# === Fetch FIRMS Fire Hotspots ===
print("🔥 Fetching FIRMS fire detections...")
firms_url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{FIRMS_TOKEN}/VIIRS_SNPP_NRT/{FIRMS_COUNTRY}/{FIRMS_DAYS}"
fires = pd.read_csv(firms_url)
fires["geometry"] = fires.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
fires_gdf = gpd.GeoDataFrame(fires, geometry="geometry", crs="EPSG:4326")

# === Count Hotspots per Grid with Accurate Buffering ===
print("📍 Counting fire hotspots per grid...")
grids_proj = grids.to_crs(epsg=32643)
fires_proj = fires_gdf.to_crs(grids_proj.crs)
grids_proj["geometry"] = grids_proj["geometry"].buffer(1000)  # 1 km buffer
joined = gpd.sjoin(fires_proj, grids_proj, how="left", predicate="within")
fire_counts = joined.groupby("grid_id").size().reset_index(name="firs_hotspots")
grids = grids.merge(fire_counts, on="grid_id", how="left")
grids["firs_hotspots"] = grids["firs_hotspots"].fillna(0).astype(int)

# === Fetch NDVI + Terrain Features from Google Earth Engine (Chunked) ===
print("🌿 Fetching NDVI and terrain features from GEE...")
ee.Initialize()

# NDVI (MODIS)
ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI").sort("system:time_start", False).first()

# Terrain (SRTM DEM)
dem = ee.Image("USGS/SRTMGL1_003").resample('bilinear')
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)

combined_img = ndvi_img.addBands(dem.rename('elevation')) \
                       .addBands(slope.rename('slope')) \
                       .addBands(aspect.rename('aspect'))

def fetch_chunk(chunk_df):
    features = []
    for _, row in chunk_df.iterrows():
        # Buffer each point by 150m to avoid NoData artifacts
        point = ee.Geometry.Point([row["lon"], row["lat"]]).buffer(150)
        features.append(ee.Feature(point, {"grid_id": row["grid_id"]}))
    fc = ee.FeatureCollection(features)
    results_fc = combined_img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=90, tileScale=4)
    results_list = results_fc.getInfo()["features"]
    return {
        f["properties"]["grid_id"]: {
            "NDVI": f["properties"].get("NDVI"),
            "elevation": f["properties"].get("elevation"),
            "slope": f["properties"].get("slope"),
            "aspect": f["properties"].get("aspect")
        }
        for f in results_list
    }

# Process in chunks of 1000
chunk_size = 1000
all_results = {}
for i in range(0, len(grids), chunk_size):
    chunk = grids.iloc[i:i+chunk_size]
    print(f"➡️ Processing chunk {i}–{i+len(chunk)}...")
    chunk_results = fetch_chunk(chunk)
    all_results.update(chunk_results)

# Map back to dataframe
grids["ndvi"] = grids["grid_id"].map(lambda gid: all_results.get(gid, {}).get("NDVI"))
grids["elevation"] = grids["grid_id"].map(lambda gid: all_results.get(gid, {}).get("elevation"))
grids["slope"] = grids["grid_id"].map(lambda gid: all_results.get(gid, {}).get("slope"))
grids["aspect"] = grids["grid_id"].map(lambda gid: all_results.get(gid, {}).get("aspect"))

# Normalize NDVI
grids["ndvi"] = pd.to_numeric(grids["ndvi"], errors="coerce") / 10000.0
grids["ndvi"] = grids["ndvi"].fillna(0.0)

# Handle missing terrain: use median instead of 0 to avoid bias
for col in ["elevation", "slope", "aspect"]:
    grids[col] = pd.to_numeric(grids[col], errors="coerce")
    med = grids[col].median() if pd.notna(grids[col].median()) else 0.0
    grids[col] = grids[col].fillna(med)

# Optional: zero aspect where slope is near-flat
grids.loc[grids["slope"] < 0.5, "aspect"] = 0.0

# === Save ===
cols = ["grid_id", "lat", "lon", "ndvi", "firs_hotspots", "elevation", "slope", "aspect"]
grids[cols].to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Real-time features saved to {OUTPUT_CSV}")
