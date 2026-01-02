import geopandas as gpd
from shapely.geometry import box
import numpy as np
import os

# Load forest shapefile
forest = gpd.read_file("data/shapefiles/kerala_osm_forests.shp")
forest = forest.to_crs(epsg=32643)  # UTM zone for Kerala

# Get bounding box
xmin, ymin, xmax, ymax = forest.total_bounds
grid_size = 1000  # 1km in meters

# Generate grid cells
grid_cells = []
for x in np.arange(xmin, xmax, grid_size):
    for y in np.arange(ymin, ymax, grid_size):
        cell = box(x, y, x + grid_size, y + grid_size)
        grid_cells.append(cell)

grid = gpd.GeoDataFrame(geometry=grid_cells, crs=forest.crs)

# Clip grid to forest polygons
grid_clipped = gpd.overlay(grid, forest, how="intersection")
grid_clipped["grid_id"] = range(len(grid_clipped))

# Save grid shapefile
os.makedirs("data/shapefiles", exist_ok=True)
grid_clipped.to_file("data/shapefiles/kerala_forest_grid.shp")
print("✅ Grid saved: kerala_forest_grid.shp")
