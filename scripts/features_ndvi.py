import pandas as pd

df = pd.read_csv('data/kerala_wildfire_with_ndvi.csv')

# Create lag and delta features
df['ndvi_lag1'] = df.groupby('district')['ndvi'].shift(1)
df['ndvi_delta'] = df['ndvi'] - df['ndvi_lag1']

# Save updated dataset
df.to_csv('data/kerala_wildfire_features.csv', index=False)
print("✅ NDVI features added.")
