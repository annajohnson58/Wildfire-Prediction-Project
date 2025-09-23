import pandas as pd
import os

# Set base path to your EarthEngineExports folder
base_path = 'C:/Users/annac/OneDrive/Desktop/projects/Foresight-for-Forests/EarthEngineExports'

# Load NDVI data
ndvi_path = os.path.join(base_path, 'ndvi_thrissur_2024.csv')
ndvi = pd.read_csv(ndvi_path)

# Load FIRMS fire data
firms_path = os.path.join(base_path, 'firms_firecounts_thrissur_2024.csv')  # or 'firms_firecounts_thrissur_2024.csv'
firms = pd.read_csv(firms_path)

# Inspect columns to confirm correct names
print("NDVI columns:", ndvi.columns.tolist())
print("FIRMS columns:", firms.columns.tolist())

# Use correct column names based on your actual CSV structure
# If FIRMS has 'confidence', use that; if it has 'fire_count', use that
firms_column = 'confidence' if 'confidence' in firms.columns else 'fire_count'

# Merge datasets on 'date'
merged = pd.merge(ndvi[['date', 'NDVI']], firms[['date', firms_column]], on='date', how='left')

# Fill missing fire values with 0
merged[firms_column] = merged[firms_column].fillna(0)

# Create binary fire label
if firms_column == 'confidence':
    # Use a threshold (e.g. 50) to label fire days
    merged['fire_label'] = merged['confidence'].apply(lambda x: 1 if x > 50 else 0)
else:
    # If using fire_count, label fire days if count > 0
    merged['fire_label'] = merged['fire_count'].apply(lambda x: 1 if x > 0 else 0)

# Preview merged dataset
print("\nMerged NDVI + FIRMS data:")
print(merged.head())

# Optional: Save merged output
output_path = os.path.join(base_path, 'merged_ndvi_firms_thrissur_2024.csv')
merged.to_csv(output_path, index=False)
print(f"\n✅ Merged dataset saved to: {output_path}")
