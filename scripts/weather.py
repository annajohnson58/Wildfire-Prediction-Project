import xarray as xr
import pandas as pd
import geopandas as gpd

# ---------------------- Load ERA5 Dataset ----------------------
df_weather = pd.read_csv("data/processed/era5_with_districts.csv")

df_weather = ds.to_dataframe().reset_index()

# ---------------------- Convert to GeoDataFrame ----------------------
weather_gdf = gpd.GeoDataFrame(
    df_weather,
    geometry=gpd.points_from_xy(df_weather["longitude"], df_weather["latitude"]),
    crs="EPSG:4326"
)

# ---------------------- Load Kerala District Boundaries ----------------------
kerala_districts = gpd.read_file("data/shapefiles/kerala_districts.shp")
kerala_districts.rename(columns={"DISTRICT": "district"}, inplace=True)
kerala_districts["district"] = kerala_districts["district"].str.title()

# ---------------------- Spatial Join ----------------------
weather_tagged = gpd.sjoin(
    weather_gdf,
    kerala_districts[["district", "geometry"]],
    how="inner",
    predicate="intersects"
)

# ---------------------- Clean and Save ----------------------
weather_tagged = weather_tagged.drop(columns=["geometry", "index_right"])
weather_tagged.to_csv("data/processed/era5_with_districts.csv", index=False)
print("✅ ERA5 weather data tagged with Kerala districts.")