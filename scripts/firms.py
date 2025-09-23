import pandas as pd
import geopandas as gpd

# ---------------------- Load FIRMS Data ----------------------
firms = pd.read_csv("data/raw/viirs_india_2022_2025.csv")
firms.columns = firms.columns.str.strip().str.lower()

# ---------------------- Convert to GeoDataFrame ----------------------
firms_gdf = gpd.GeoDataFrame(
    firms,
    geometry=gpd.points_from_xy(firms["longitude"], firms["latitude"]),
    crs="EPSG:4326"
)

# ---------------------- Load Kerala District Boundaries ----------------------
kerala_districts = gpd.read_file("data/shapefiles/kerala_districts.shp")  # or .shp
kerala_districts.rename(columns={"DISTRICT": "district"}, inplace=True)
kerala_districts["district"] = kerala_districts["district"].str.title()

# ---------------------- Spatial Join ----------------------
firms_tagged = gpd.sjoin(
    firms_gdf,
    kerala_districts[["district", "geometry"]],
    how="inner",
    predicate="intersects"
)

# ---------------------- Clean and Save ----------------------
firms_tagged = firms_tagged.drop(columns=["geometry", "index_right"])
firms_tagged.to_csv("data/processed/firms_with_districts.csv", index=False)
print("✅ FIRMS fire detections tagged with Kerala districts.")
