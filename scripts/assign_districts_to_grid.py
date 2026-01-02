import sqlite3
import geopandas as gpd
from shapely.geometry import Point

DB = "data/wildfire_grid.db"
DISTRICTS = "data/shapefiles/kerala_districts.shp"   # <-- use your actual filename

def main():
    print("Loading forest grid cells...")

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Load grid cells
    rows = cur.execute("SELECT grid_id, lat, lon FROM grid_cells").fetchall()
    grid = gpd.GeoDataFrame(
        rows,
        columns=["grid_id", "lat", "lon"],
        geometry=[Point(lon, lat) for _, lat, lon in rows],
        crs="EPSG:4326"
    )

    print("Loading Kerala district polygons...")
    districts = gpd.read_file(DISTRICTS).to_crs(epsg=4326)

    # Identify the district name column
    name_col = None
    for col in districts.columns:
        if col.lower() in ["district", "name", "name_2", "dist_name"]:
            name_col = col
            break

    if name_col is None:
        raise ValueError("Could not find district name column in shapefile.")

    print(f"Using district name column: {name_col}")

    print("Performing spatial join...")
    joined = gpd.sjoin(grid, districts, how="left", predicate="within")

    print("Updating database...")
    updates = [(row[name_col], row["grid_id"]) for _, row in joined.iterrows()]

    cur.executemany("UPDATE grid_cells SET district=? WHERE grid_id=?", updates)
    conn.commit()
    conn.close()

    print("✅ Districts assigned to forest grid cells.")

if __name__ == "__main__":
    main()
