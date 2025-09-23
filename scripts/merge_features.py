import xarray as xr
import os

# Define input folder and output path
input_folder = "data/weather/era5_daily"
output_path = "data/weather/kerala_era5_2022_2025.nc"

# List all yearly files
files = [
    "kerala_era5_2022.nc",
    "kerala_era5_2023.nc",
    "kerala_era5_2024.nc",
    "kerala_era5_2025.nc"
]

# Load and concatenate datasets
datasets = [xr.open_dataset(os.path.join(input_folder, f)) for f in files]
merged = xr.concat(datasets, dim="time")

# Save merged file
merged.to_netcdf(output_path)
print(f"✅ Merged ERA5 data saved to: {output_path}")
