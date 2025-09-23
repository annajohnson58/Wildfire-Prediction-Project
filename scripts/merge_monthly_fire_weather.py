import pandas as pd
import numpy as np
import os

# Base path
base = 'C:/Users/annac/OneDrive/Desktop/projects/Foresight-for-Forests/EarthEngineExports'

# Load daily fire dataset
fire_df = pd.read_csv(os.path.join(base, 'merged_ndvi_firms_lst_thrissur_2024.csv'))

# Aggregate fire activity by month
fire_df['month'] = pd.to_datetime(fire_df['date']).dt.to_period('M')
monthly_fire = fire_df.groupby('month')['fire_label'].sum().reset_index()
monthly_fire['month'] = monthly_fire['month'].astype(str)

# Load ERA5-Land monthly weather
weather_df = pd.read_csv(os.path.join(base, 'era5land_monthly_thrissur_2024.csv'))
weather_df['month'] = weather_df['date']  # Rename for merge
weather_df['temp_C'] = weather_df['temp_2m'] - 273.15

# Merge fire + weather
merged = pd.merge(monthly_fire, weather_df, on='month', how='left')

# Compute wind magnitude
merged['wind_speed'] = np.sqrt(merged['u_wind_10m']**2 + merged['v_wind_10m']**2)

# Preview
print("\n🔥 Final monthly fire + weather dataset:")
print(merged[['month', 'fire_label', 'temp_C', 'precip_m', 'wind_speed']].head())

# Save output
merged.to_csv(os.path.join(base, 'merged_monthly_fire_weather_thrissur_2024.csv'), index=False)
print("\n✅ Final dataset saved to EarthEngineExports/merged_monthly_fire_weather_thrissur_2024.csv")
