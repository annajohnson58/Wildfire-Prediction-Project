import xarray as xr
import geopandas as gpd
import pandas as pd
import os
from rasterstats import zonal_stats
import rioxarray  # activates .rio accessor

# Load district shapefile
gdf = gpd.read_file('data/kerala_districts.shp')
gdf = gdf[gdf['DISTRICT'].isin(['Thrissur', 'Palakkad', 'Idukki'])]

# Load ERA5-Land NetCDF file
ds = xr.open_dataset('data/era5land_kerala_2024/data_stream-mnth.nc', engine='netcdf4')

# Create output folder
os.makedirs('data/climate_features', exist_ok=True)

# Loop through each month
for i in range(len(ds.valid_time)):
    month_str = pd.to_datetime(ds.valid_time.values[i]).strftime('%Y-%m')
    print(f"📦 Extracting {month_str}")

    rows = []
    for var in ['t2m', 'tp', 'u10', 'v10']:
        da = ds[var].isel(valid_time=i).squeeze()
        da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        da.rio.write_crs("EPSG:4326", inplace=True)
        tif_path = f'temp_{var}.tif'
        da.rio.to_raster(tif_path)

        stats = zonal_stats(gdf, tif_path, stats="mean", geojson_out=True)
        for s in stats:
            district = s['properties']['DISTRICT']
            val = s['properties']['mean']
            rows.append({'month': month_str, 'district': district, 'variable': var, 'value': val})

        os.remove(tif_path)

    df = pd.DataFrame(rows)
    df_pivot = df.pivot_table(index=['month', 'district'], columns='variable', values='value').reset_index()
    df_pivot.to_csv(f'data/climate_features/climate_{month_str}.csv', index=False)
