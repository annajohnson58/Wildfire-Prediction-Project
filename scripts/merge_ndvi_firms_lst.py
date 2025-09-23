import pandas as pd
import numpy as np
import os

# Base path
base = 'C:/Users/annac/OneDrive/Desktop/projects/Foresight-for-Forests/EarthEngineExports'

# Load NDVI + fire dataset
merged = pd.read_csv(os.path.join(base, 'merged_ndvi_firms_thrissur_2024.csv'))

# Load LST dataset
lst = pd.read_csv(os.path.join(base, 'lst_thrissur_2024.csv'))

# Merge on 'date'
merged = pd.merge(merged, lst[['date', 'LST_C']], on='date', how='left')

# Replace -9999 with NaN
merged['LST_C'] = merged['LST_C'].replace(-9999, np.nan)

# Preview
print("\n🔥 Final merged dataset:")
print(merged.head())

# Optional: Save output
merged.to_csv(os.path.join(base, 'merged_ndvi_firms_lst_thrissur_2024.csv'), index=False)
print("\n✅ Final dataset saved to EarthEngineExports/merged_ndvi_firms_lst_thrissur_2024.csv")
