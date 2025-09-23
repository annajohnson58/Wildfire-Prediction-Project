# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import os

# # Load merged monthly dataset
# base = 'C:/Users/annac/OneDrive/Desktop/projects/Foresight-for-Forests/EarthEngineExports'
# df = pd.read_csv(os.path.join(base, 'merged_monthly_fire_weather_thrissur_2024.csv'))

# # Convert temperature from Kelvin to Celsius (already done, but safe to reapply)
# df['temp_C'] = df['temp_2m'] - 273.15

# # Convert precipitation to mm
# df['precip_mm'] = df['precip_m'] * 1000

# # Compute wind magnitude
# df['wind_speed'] = np.sqrt(df['u_wind_10m']**2 + df['v_wind_10m']**2)

# # Create lag features
# df['fire_label_t-1'] = df['fire_label'].shift(1)
# df['temp_C_t-1'] = df['temp_C'].shift(1)
# df['precip_mm_t-1'] = df['precip_mm'].shift(1)
# df['wind_speed_t-1'] = df['wind_speed'].shift(1)

# # Drop first row (NaNs from lag)
# df = df.dropna().reset_index(drop=True)

# # Select features for modeling
# features = ['temp_C', 'precip_mm', 'wind_speed', 'temp_C_t-1', 'precip_mm_t-1', 'wind_speed_t-1', 'fire_label_t-1']
# target = 'fire_label'

# X = df[features]
# y = df[target]

# # Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Save preprocessed data
# preprocessed = pd.DataFrame(X_scaled, columns=features)
# preprocessed['fire_label'] = y.values
# preprocessed['month'] = df['month']

# # Preview
# print("\n✅ Preprocessed wildfire dataset:")
# print(preprocessed.head())

# # Save output
# preprocessed.to_csv(os.path.join(base, 'preprocessed_fire_dataset_thrissur_2024.csv'), index=False)
# print("\n📦 Saved to EarthEngineExports/preprocessed_fire_dataset_thrissur_2024.csv")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Load daily merged dataset
base = 'C:/Users/annac/OneDrive/Desktop/projects/Foresight-for-Forests/EarthEngineExports'
df = pd.read_csv(os.path.join(base, 'merged_ndvi_firms_lst_weather_thrissur_2024.csv'))

# Convert date and units
df['date'] = pd.to_datetime(df['date'])
df['temp_C'] = df['temp_2m'] - 273.15
df['precip_mm'] = df['precip_m'] * 1000

# Rolling features
df['NDVI_drop_3d'] = df['NDVI'].rolling(3).apply(lambda x: x.iloc[0] - x.iloc[-1])
df['rain_7d'] = df['precip_mm'].rolling(7).sum()
df['fire_5d'] = df['fire_label'].rolling(5).sum()

# Lag features
df['temp_C_t-1'] = df['temp_C'].shift(1)
df['NDVI_t-1'] = df['NDVI'].shift(1)
df['fire_label_t-1'] = df['fire_label'].shift(1)

# Drop rows with NaNs
df = df.dropna().reset_index(drop=True)

# Select features and target
features = [
    'NDVI', 'NDVI_drop_3d', 'NDVI_t-1',
    'temp_C', 'temp_C_t-1',
    'rain_7d', 'wind_speed',
    'fire_5d', 'fire_label_t-1'
]
target = 'fire_label'

X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Final dataset
preprocessed = pd.DataFrame(X_scaled, columns=features)
preprocessed['fire_label'] = y.values
preprocessed['date'] = df['date']

# Preview
print("\n✅ Preprocessed daily wildfire dataset:")
print(preprocessed.head())

# Save output
preprocessed.to_csv(os.path.join(base, 'preprocessed_daily_fire_dataset_thrissur_2024.csv'), index=False)
print("\n📦 Saved to EarthEngineExports/preprocessed_daily_fire_dataset_thrissur_2024.csv")
