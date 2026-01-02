import pandas as pd
import os

# Load raw export
df = pd.read_csv("Data/Kerala_MODIS_NDVI_2022_2024.csv")

# Keep only relevant columns
df_clean = df[["grid_id", "date", "mean"]].rename(columns={"mean": "ndvi"})

# Drop rows with missing NDVI
df_clean = df_clean.dropna(subset=["ndvi"])

# Save cleaned file
os.makedirs("data/features", exist_ok=True)
df_clean.to_csv("data/features/ndvi_grid_clean.csv", index=False)
print("✅ Cleaned NDVI saved: ndvi_grid_clean.csv")
