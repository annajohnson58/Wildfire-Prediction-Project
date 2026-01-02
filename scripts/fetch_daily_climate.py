

# import cdsapi
# import xarray as xr
# import pandas as pd
# import zipfile
# import os
# from datetime import datetime, timedelta

# # 📅 Define date range
# start_date = datetime(2022, 1, 1)
# end_date = datetime(2024, 12, 31)

# # 📂 Create folders if needed
# os.makedirs("data/era5_raw", exist_ok=True)
# output_path = 'data/daily_climate_long.csv'

# # 🧠 Generate list of dates
# date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d")
#              for i in range((end_date - start_date).days + 1)]

# # 🧠 Load already processed dates
# if os.path.exists(output_path):
#     processed = pd.read_csv(output_path)
#     done_dates = set(pd.to_datetime(processed['date']).dt.strftime("%Y-%m-%d"))
# else:
#     done_dates = set()

# # 🔁 Loop through each date
# c = cdsapi.Client()
# for date in date_list:
#     if date in done_dates:
#         print(f"⏭️ Skipping {date} (already processed)")
#         continue

#     year, month, day = date.split("-")
#     print(f"📦 Fetching ERA5 for {date}...")

#     # 🔧 Download hourly ERA5-Land data
#     zip_path = f"data/era5_raw/{date}.zip"
#     try:
#         c.retrieve(
#             'reanalysis-era5-land',
#             {
#                 'variable': [
#                     '2m_temperature',
#                     'total_precipitation',
#                     '10m_u_component_of_wind',
#                     '10m_v_component_of_wind'
#                 ],
#                 'year': year,
#                 'month': month,
#                 'day': day,
#                 'time': [f'{h:02d}:00' for h in range(24)],
#                 'format': 'netcdf',
#                 'area': [10.8, 75.8, 8.8, 77.2],  # Kerala bounding box
#             },
#             zip_path
#         )
#     except Exception as e:
#         print(f"❌ Failed to fetch {date}: {e}")
#         continue

#     # 🧩 Unzip and extract NetCDF
#     extract_path = f"data/era5_raw/{date}"
#     os.makedirs(extract_path, exist_ok=True)
#     try:
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extract_path)
#     except Exception as e:
#         print(f"❌ Failed to unzip {date}: {e}")
#         continue

#     # 📊 Load and aggregate
#     nc_files = [f for f in os.listdir(extract_path) if f.endswith('.nc')]
#     if not nc_files:
#         print(f"⚠️ No NetCDF file found for {date}")
#         continue

#     nc_path = os.path.join(extract_path, nc_files[0])
#     try:
#         ds = xr.open_dataset(nc_path, engine='netcdf4')
#         df = ds.to_dataframe().reset_index()
#     except Exception as e:
#         print(f"❌ Failed to load NetCDF for {date}: {e}")
#         continue

#     # ✅ Keep only numeric columns
#     numeric_cols = ['valid_time', 't2m', 'tp', 'u10', 'v10']
#     df_clean = df[numeric_cols].copy()

#     # ✅ Group by valid_time and compute mean
#     daily_mean = df_clean.groupby('valid_time').mean().reset_index()
#     daily_mean['date'] = date

#     # ✅ Append to master file
#     daily_mean.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
#     print(f"✅ Saved {date} to daily_climate_long.csv")

# print("🎉 ERA5 fetch resumed and completed.")
import cdsapi
import xarray as xr
import geopandas as gpd
import pandas as pd
import zipfile
import os
from datetime import datetime, timedelta

# 📅 Define date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)

# 📂 Paths
districts_path = "data/shapefiles/kerala_districts.shp"  # Your district shapefile
output_path = "data/daily_climate_district.csv"
os.makedirs("data/era5_raw", exist_ok=True)

# 🧠 Load districts
districts = gpd.read_file(districts_path)
districts = districts.to_crs("EPSG:4326")  # Ensure lat/lon

# 🧠 Generate list of dates
date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range((end_date - start_date).days + 1)]

# 🧠 Load already processed dates
if os.path.exists(output_path):
    processed = pd.read_csv(output_path)
    done_dates = set(pd.to_datetime(processed['date']).dt.strftime("%Y-%m-%d"))
else:
    done_dates = set()

# 🔁 Loop through each date
c = cdsapi.Client()
for date in date_list:
    if date in done_dates:
        print(f"⏭️ Skipping {date} (already processed)")
        continue

    year, month, day = date.split("-")
    print(f"📦 Fetching ERA5 for {date}...")

    zip_path = f"data/era5_raw/{date}.zip"
    try:
        c.retrieve(
            'reanalysis-era5-land',
            {
                'variable': [
                    '2m_temperature',
                    'total_precipitation',
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind'
                ],
                'year': year,
                'month': month,
                'day': day,
                'time': [f'{h:02d}:00' for h in range(24)],
                'format': 'netcdf',
                'area': [10.8, 75.8, 8.8, 77.2],  # Kerala bounding box
            },
            zip_path
        )
    except Exception as e:
        print(f"❌ Failed to fetch {date}: {e}")
        continue

    # 🧩 Unzip and extract NetCDF
    extract_path = f"data/era5_raw/{date}"
    os.makedirs(extract_path, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except Exception as e:
        print(f"❌ Failed to unzip {date}: {e}")
        continue

    # 📊 Load and aggregate
    nc_files = [f for f in os.listdir(extract_path) if f.endswith('.nc')]
    if not nc_files:
        print(f"⚠️ No NetCDF file found for {date}")
        continue

    nc_path = os.path.join(extract_path, nc_files[0])
    try:
        ds = xr.open_dataset(nc_path)
        df = ds.to_dataframe().reset_index()
    except Exception as e:
        print(f"❌ Failed to load NetCDF for {date}: {e}")
        continue

    # ✅ Compute daily mean per district
    results = []
    for _, row in districts.iterrows():
        name = row["DISTRICT"]
        geom = row["geometry"]
        mask = (df["latitude"] >= geom.bounds[1]) & (df["latitude"] <= geom.bounds[3]) & \
               (df["longitude"] >= geom.bounds[0]) & (df["longitude"] <= geom.bounds[2])
        df_district = df[mask]
        if df_district.empty:
            continue
        temp = df_district["t2m"].mean()
        rain = df_district["tp"].mean()
        u10 = df_district["u10"].mean()
        v10 = df_district["v10"].mean()
        results.append({
            "date": date,
            "district": name,
            "temperature": temp,
            "rainfall": rain,
            "wind_u": u10,
            "wind_v": v10
        })

    # ✅ Append to master file
    if results:
        pd.DataFrame(results).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        print(f"✅ Saved {date} to daily_climate_district.csv")

print("🎉 District-wise ERA5 fetch completed.")
