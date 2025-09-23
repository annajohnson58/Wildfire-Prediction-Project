import pandas as pd

# Load climate features
climate = pd.read_csv('data/climate_features/kerala_climate_2024.csv')

# Load fire counts
fire = pd.read_csv('data/fire_counts_2024.csv')

# Merge on month and district
merged = pd.merge(climate, fire, on=['month', 'district'], how='left')

# Save final dataset
merged.to_csv('data/kerala_wildfire_dataset_2024.csv', index=False)
