import cdsapi
import pandas as pd
import os

# Load grid centroids
grid_df = pd.read_csv("data/geojson/grid_centroids.csv")

# Create output folder
os.makedirs("data/features/weather", exist_ok=True)

# Initialize CDS API
c = cdsapi.Client()

# Loop over each grid cell
for _, row in grid_df.iterrows():
    grid_id = row["grid_id"]
    lat = row["lat"]
    lon = row["lon"]
    output_path = f"data/features/weather/grid_{grid_id}_era5.nc"

    # ✅ Skip if already downloaded
    if os.path.exists(output_path):
        print(f"⏭️ Skipping grid {grid_id} — already downloaded")
        continue

    print(f"⬇️ Downloading ERA5 for grid {grid_id}...")

    try:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["2m_temperature", "2m_dewpoint_temperature"],
                "year": [str(y) for y in range(2022, 2025)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00"],
                "format": "netcdf",
                "area": [lat + 0.05, lon - 0.05, lat - 0.05, lon + 0.05],
            },
            output_path
        )
        print(f"✅ Finished grid {grid_id}")
    except Exception as e:
        print(f"❌ Failed grid {grid_id}: {e}")

print("🎉 All available ERA5 downloads complete.")
