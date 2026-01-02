# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# import os

# # Load VIIRS fire detections
# fire_df = pd.read_csv("data/fire_archive_SV-C2_671063.csv")
# fire_df["date"] = pd.to_datetime(fire_df["acq_date"])
# fire_df["geometry"] = fire_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
# fire_gdf = gpd.GeoDataFrame(fire_df, geometry="geometry", crs="EPSG:4326")

# # Load forest grid
# grid_gdf = gpd.read_file("data/geojson/kerala_forest_grid.geojson")

# # Spatial join: assign grid_id to each fire point
# joined = gpd.sjoin(fire_gdf, grid_gdf[["grid_id", "geometry"]], how="inner", predicate="intersects")

# # Aggregate: count fires per grid per day
# thermal_daily = joined.groupby(["grid_id", "date"]).size().reset_index(name="thermal_count")

# # Save
# os.makedirs("data/features", exist_ok=True)
# thermal_daily.to_csv("data/features/thermal_grid_daily.csv", index=False)
# print("✅ Thermal data saved: thermal_grid_daily.csv")
import pandas as pd
import os

# Load sparse fire detections
thermal_df = pd.read_csv("data/features/thermal_grid_daily.csv")
thermal_df["date"] = pd.to_datetime(thermal_df["date"])  # auto-detects ISO format

# Create full date range
full_dates = pd.date_range("2022-01-01", "2024-12-31")

# Get all grid_ids
grid_ids = thermal_df["grid_id"].unique()

# Create full grid-date matrix
full_index = pd.MultiIndex.from_product([grid_ids, full_dates], names=["grid_id", "date"])
full_df = pd.DataFrame(index=full_index).reset_index()

# Merge with detections
thermal_df["thermal_flag"] = 1
merged = pd.merge(full_df, thermal_df[["grid_id", "date", "thermal_flag"]], on=["grid_id", "date"], how="left")
merged["thermal_flag"] = merged["thermal_flag"].fillna(0).astype(int)

# Save
os.makedirs("data/features", exist_ok=True)
merged.to_csv("data/features/thermal_flag_grid_daily.csv", index=False)
print("✅ Thermal flags saved: thermal_flag_grid_daily.csv")
