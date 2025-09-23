import pandas as pd

# 🔄 Load NDVI and climate data
ndvi_df = pd.read_csv('data/daily_ndvi.csv')
climate_df = pd.read_csv('data/daily_climate.csv')

# 🧠 Aggregate climate to district level (mean across Kerala)
climate_mean = climate_df[['t2m', 'tp', 'u10', 'v10']].mean().to_dict()

# 🧬 Merge climate into NDVI per district
for key in climate_mean:
    ndvi_df[key] = climate_mean[key]

# ✅ Final input for prediction
input_df = ndvi_df[['district', 'ndvi', 't2m', 'tp', 'u10', 'v10']]
