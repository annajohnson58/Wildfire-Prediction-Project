import geopandas as gpd
import os

# Load the grid shapefile
grid = gpd.read_file("data/shapefiles/kerala_forest_grid.shp")

# Convert to WGS84 (lat/lon)
grid = grid.to_crs(epsg=4326)

# Save as GeoJSON
os.makedirs("data/geojson", exist_ok=True)
grid.to_file("data/geojson/kerala_forest_grid.geojson", driver="GeoJSON")

print("✅ Grid converted to GeoJSON and saved.")
