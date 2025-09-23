import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
import os

# 📅 Set date string
date_str = "2025-09-11"

# 📂 Path to extracted NetCDF file
nc_path = f"data/weather/era5_{date_str}/data_0.nc"

# 📍 Load ERA5 dataset
ds = xr.open_dataset(nc_path, engine="netcdf4")

# 🌀 Compute wind speed
u = ds['u10']
v = ds['v10']
wind_speed = np.sqrt(u**2 + v**2)

# 💧 Estimate relative humidity from temp & dewpoint
T = ds['t2m'] - 273.15  # Convert to °C
Td = ds['d2m'] - 273.15
rh = 100 * (np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T)))

# 🌧️ Precipitation in mm
precip = ds['tp'] * 1000

# 🧮 Aggregate over valid_time
mean_temp = T.mean(dim='valid_time')
mean_rh = rh.mean(dim='valid_time')
mean_wind = wind_speed.mean(dim='valid_time')
total_precip = precip.sum(dim='valid_time')

# 🧭 Create full lat/lon grid
lats = ds.latitude.values
lons = ds.longitude.values
lon_grid, lat_grid = np.meshgrid(lons, lats)

# 📦 Flatten and build DataFrame
df = pd.DataFrame({
    'lat': lat_grid.flatten(),
    'lon': lon_grid.flatten(),
    'temp': mean_temp.values.flatten(),
    'rh': mean_rh.values.flatten(),
    'wind': mean_wind.values.flatten(),
    'precip': total_precip.values.flatten()
})

# 🗺️ Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

# 📍 Load Kerala district shapefile
districts = gpd.read_file("data/shapefiles/kerala_districts.shp")
districts = districts.to_crs("EPSG:4326")

# 🔗 Spatial join
joined = gpd.sjoin(gdf, districts, how='inner', predicate='intersects')

# 📊 Aggregate per district
district_col = 'DISTRICT' if 'DISTRICT' in joined.columns else joined.columns[-1]
weather = joined.groupby(district_col)[['temp', 'rh', 'wind', 'precip']].mean().reset_index()
weather['date'] = date_str

# 💾 Save to CSV
output_path = f"data/weather/district_weather_{date_str}.csv"
weather.to_csv(output_path, index=False)
print(f"✅ District-level weather saved to {output_path}")
