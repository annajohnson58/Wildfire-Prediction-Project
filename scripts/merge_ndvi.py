import pandas as pd

# Load full grid with fire labels
df = pd.read_csv('data/kerala_wildfire_full.csv')

# Load NDVI data
ndvi = pd.read_csv('data/Sentinel2_NDVI_Kerala_2024.csv')

# Merge NDVI into full grid
merged = pd.merge(df, ndvi[['district', 'month', 'ndvi']], on=['district', 'month'], how='left')

# Fill missing NDVI with 0 or NaN-safe value
merged['ndvi'] = merged['ndvi'].fillna(0)

# Save enriched dataset
merged.to_csv('data/kerala_wildfire_with_ndvi.csv', index=False)
print("✅ NDVI merged into full grid. Rows:", len(merged))
