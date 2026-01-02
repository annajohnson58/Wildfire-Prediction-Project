import sqlite3
import numpy as np
import rasterio
from rasterio.transform import rowcol
import os

DB_PATH = "data/wildfire_grid.db"
RASTER_PATH = "data/shapefiles/Kerala_Forest_Mask_4326.tif"

LAT_STEP = 0.01   # ~1.1 km
LON_STEP = 0.01

def main():
    if not os.path.exists(RASTER_PATH):
        raise FileNotFoundError("Forest raster not found.")

    # Load raster
    src = rasterio.open(RASTER_PATH)
    transform = src.transform

    # ✅ Read raster ONCE (critical)
    band = src.read(1)

    # Get raster bounds
    minx, miny, maxx, maxy = src.bounds

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM grid_cells")

    grid_id_counter = 0
    points_to_insert = []

    lats = np.arange(miny, maxy, LAT_STEP)
    lons = np.arange(minx, maxx, LON_STEP)

    print("Generating forest-only grid...")

    for lat in lats:
        for lon in lons:
            # Convert lat/lon to raster row/col
            row, col = rowcol(transform, lon, lat)

            # Skip if outside raster
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                continue

            # ✅ Fast lookup
            value = band[row, col]

            # Forest mask = 1
            if value != 1:
                continue

            grid_id = f"GRID_{grid_id_counter:06d}"
            grid_id_counter += 1

            points_to_insert.append((grid_id, lat, lon, None, None))

    print(f"✅ Total forest grid cells: {len(points_to_insert)}")

    cur.executemany("""
        INSERT INTO grid_cells (grid_id, lat, lon, district, forest)
        VALUES (?, ?, ?, ?, ?)
    """, points_to_insert)

    conn.commit()
    conn.close()
    print("✅ Forest grid saved to grid_cells table.")

if __name__ == "__main__":
    main()
