import sqlite3
import numpy as np
import geopandas as gpd
import os

DB_PATH = "data/wildfire_grid.db"
SHAPEFILE_PATH = "data/shapefiles/kerala_districts.shp"

# Approximate grid spacing in degrees for ~1km (0.01 ≈ 1.1km)
LAT_STEP = 0.01
LON_STEP = 0.01

def main():
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"Shapefile not found at {SHAPEFILE_PATH}")

    districts = gpd.read_file(SHAPEFILE_PATH)
    districts["DISTRICT"] = districts["DISTRICT"].str.title()
    districts = districts.to_crs(epsg=4326)  # ensure lat/lon

    # Get bounding box for Kerala
    minx, miny, maxx, maxy = districts.total_bounds
    print("Kerala bounds:", minx, miny, maxx, maxy)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    grid_id_counter = 0
    points_to_insert = []

    lats = np.arange(miny, maxy, LAT_STEP)
    lons = np.arange(minx, maxx, LON_STEP)

    print("Generating grid points...")
    for lat in lats:
        for lon in lons:
            point = gpd.points_from_xy([lon], [lat], crs="EPSG:4326")[0]

            # Check if this point lies inside any Kerala district
            mask = districts.contains(point)
            if not mask.any():
                continue

            district_name = districts.loc[mask, "DISTRICT"].values[0]
            forest_name = None  # You can later overlay forest polygons if you have them

            grid_id = f"GRID_{grid_id_counter:06d}"
            grid_id_counter += 1

            points_to_insert.append((grid_id, lat, lon, district_name, forest_name))

    print(f"Total grid cells inside Kerala: {len(points_to_insert)}")

    cur.executemany("""
        INSERT OR REPLACE INTO grid_cells (grid_id, lat, lon, district, forest)
        VALUES (?, ?, ?, ?, ?)
    """, points_to_insert)

    conn.commit()
    conn.close()
    print("✅ Grid cells saved to grid_cells table.")

if __name__ == "__main__":
    main()
