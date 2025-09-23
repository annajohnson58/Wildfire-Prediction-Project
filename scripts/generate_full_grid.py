import pandas as pd
from itertools import product

# ✅ Define districts and months
districts = [
    'Alappuzha', 'Ernakulam', 'Idukki', 'Kannur', 'Kasaragod', 'Kollam',
    'Kottayam', 'Kozhikode', 'Malappuram', 'Palakkad', 'Pathanamthitta',
    'Thiruvananthapuram', 'Thrissur', 'Wayanad'
]
months = [f'2024-{str(m).zfill(2)}' for m in range(1, 13)]

# ✅ Create full grid
grid = pd.DataFrame(list(product(districts, months)), columns=['district', 'month'])

# ✅ Load fire data
fire_df = pd.read_csv('data/kerala_wildfire_dataset_2024.csv')

# ✅ Merge and fill missing fire counts
merged = pd.merge(grid, fire_df, on=['district', 'month'], how='left')
merged['fire_count'] = merged['fire_count'].fillna(0)
merged['fire_label'] = (merged['fire_count'] > 0).astype(int)

# ✅ Save full dataset
merged.to_csv('data/kerala_wildfire_full.csv', index=False)
print("✅ Full grid with fire labels saved.")
