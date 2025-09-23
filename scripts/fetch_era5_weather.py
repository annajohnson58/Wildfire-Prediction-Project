# import cdsapi
# import datetime
# import os

# # 📅 Use latest available ERA5 date (typically 4–5 days behind)
# latest_available_date = datetime.date.today() - datetime.timedelta(days=5)
# date_str = latest_available_date.strftime('%Y-%m-%d')
# print(f"📅 Requesting ERA5 weather for {date_str}")

# # 📁 Output directory
# output_dir = "data/weather"
# os.makedirs(output_dir, exist_ok=True)

# # 🌍 CDS API client
# c = cdsapi.Client()

# # 📦 ERA5-Land data request
# c.retrieve(
#     'reanalysis-era5-land',
#     {
#         'variable': [
#             '2m_temperature',
#             '2m_dewpoint_temperature',
#             'total_precipitation',
#             '10m_u_component_of_wind',
#             '10m_v_component_of_wind'
#         ],
#         'year': str(latest_available_date.year),
#         'month': str(latest_available_date.month).zfill(2),
#         'day': str(latest_available_date.day).zfill(2),
#         'time': [f"{hour:02d}:00" for hour in range(0, 24)],
#         'format': 'netcdf',
#         'area': [12.5, 74.5, 8.0, 78.5],  # Kerala bounding box: N, W, S, E
#     },
#     f"{output_dir}/era5_{date_str}.nc"
# )

# print(f"✅ ERA5 weather data saved to {output_dir}/era5_{date_str}.nc")

import cdsapi
import os
from datetime import datetime, timedelta

# Create output folder
os.makedirs("data/weather/era5_daily", exist_ok=True)

# Define years and variables
years = ["2022", "2023", "2024", "2025"]
variables = ["2m_temperature", "2m_dewpoint_temperature", "surface_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation"]

# Define Kerala bounding box (approx): [North, West, South, East]
kerala_bbox = [12.8, 74.8, 8.2, 77.5]

c = cdsapi.Client()

for year in years:
    print(f"🔄 Fetching ERA5 data for {year}...")
    c.retrieve(
        "reanalysis-era5-land",
        {
            "variable": variables,
            "year": year,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00"],
            "area": kerala_bbox,  # [N, W, S, E]
            "format": "netcdf",
        },
        f"data/weather/era5_daily/kerala_era5_{year}.nc"
    )
    print(f"✅ Saved: kerala_era5_{year}.nc")
