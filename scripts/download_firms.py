import pandas as pd
import requests
from io import StringIO

# Step 1: Download MODIS fire data (last 7 days, global)
url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv"

response = requests.get(url)
if response.status_code == 200:
    fire_data = pd.read_csv(StringIO(response.text))
else:
    raise Exception("Failed to download data")

# Step 2: Basic preprocessing
# Keep only relevant columns
columns_to_keep = ['latitude', 'longitude', 'brightness', 'scan', 'track',
                   'acq_date', 'acq_time', 'confidence', 'version', 'type']
# Only keep columns that actually exist in the dataset
available_columns = [col for col in columns_to_keep if col in fire_data.columns]
fire_data = fire_data[available_columns]

# Convert date and time to datetime
fire_data['timestamp'] = pd.to_datetime(fire_data['acq_date'] + ' ' + fire_data['acq_time'].astype(str).str.zfill(4),
                                        format='%Y-%m-%d %H%M')

# Drop rows with low confidence
fire_data = fire_data[fire_data['confidence'] != 'low']

# Save to CSV
fire_data.to_csv("data/processed_firms_fire_data.csv", index=False)
print("✅ Fire data downloaded and saved.")
