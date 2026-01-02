import geopandas as gpd
import pandas as pd
import os

# Load grid GeoJSON
gdf = gpd.read_file("data/geojson/kerala_forest_grid.geojson")

# Compute centroids
gdf["lon"] = gdf.geometry.centroid.x
gdf["lat"] = gdf.geometry.centroid.y

# Keep only grid_id, lat, lon
centroids_df = gdf[["grid_id", "lat", "lon"]]

# Save to CSV
os.makedirs("data/geojson", exist_ok=True)
centroids_df.to_csv("data/geojson/grid_centroids.csv", index=False)
print("✅ Grid centroids saved: grid_centroids.csv")
