# import cdsapi
# import xarray as xr
# import pandas as pd

# # ✅ Use latest available ERA5 date
# latest_date = '2025-09-10'  # You can automate this later via metadata query

# # 🔧 Request ERA5-Land hourly data
# c = cdsapi.Client()
# c.retrieve(
#     'reanalysis-era5-land',
#     {
#         'variable': ['2m_temperature', 'total_precipitation', '10m_u_component_of_wind', '10m_v_component_of_wind'],
#         'year': latest_date[:4],
#         'month': latest_date[5:7],
#         'day': latest_date[8:],
#         'time': [f'{h:02d}:00' for h in range(24)],
#         'format': 'netcdf',
#         'area': [10.8, 75.8, 8.8, 77.2],  # Kerala bounding box: N, W, S, E
#     },
#     'data/daily_climate.nc'
# )

# # 📊 Load and aggregate
# ds = xr.open_dataset('data/daily_climate/data_0.nc', engine='netcdf4')

# df = ds.to_dataframe().reset_index()
# print(df.columns)
# daily_means = df.groupby('valid_time').mean().reset_index()

# print(df.dtypes)

# # 💾 Save to CSV
# daily_means['date'] = '2025-09-10' 
# daily_means.to_csv('data/daily_climate.csv', index=False)
# print("✅ Daily climate data saved.")



import cdsapi
import xarray as xr
import pandas as pd
import zipfile
import os

# ✅ Use latest available ERA5 date
latest_date = '2025-09-10'

# 🔧 Request ERA5-Land hourly data as ZIP
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-land',
    {
        'variable': [
            '2m_temperature',
            'total_precipitation',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ],
        'year': latest_date[:4],
        'month': latest_date[5:7],
        'day': latest_date[8:],
        'time': [f'{h:02d}:00' for h in range(24)],
        'format': 'netcdf',
        'area': [10.8, 75.8, 8.8, 77.2],  # Kerala bounding box
    },
    'data/daily_climate.zip'
)

# 🧩 Unzip the NetCDF file
with zipfile.ZipFile('data/daily_climate.zip', 'r') as zip_ref:
    zip_ref.extractall('data/daily_climate_extracted')

# 📂 Find the extracted .nc file
nc_files = [f for f in os.listdir('data/daily_climate_extracted') if f.endswith('.nc')]
nc_path = os.path.join('data/daily_climate_extracted', nc_files[0])

# 📊 Load and aggregate
ds = xr.open_dataset(nc_path, engine='netcdf4')
df = ds.to_dataframe().reset_index()

# ✅ Keep only numeric columns
numeric_cols = ['valid_time', 't2m', 'tp', 'u10', 'v10']
df_clean = df[numeric_cols].copy()

# ✅ Group by valid_time and compute mean
daily_means = df_clean.groupby('valid_time').mean().reset_index()

# ✅ Add date column and save
daily_means['date'] = latest_date
daily_means.to_csv('data/daily_climate.csv', index=False)
print("✅ Daily climate data saved.")
