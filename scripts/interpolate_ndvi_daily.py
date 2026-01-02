import pandas as pd
import os

# Load cleaned NDVI export
df = pd.read_csv("data/features/ndvi_grid_clean.csv")

# Parse ISO-format date (e.g. "2022-01-01")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

# Interpolate NDVI per grid_id
daily_frames = []
for grid_id, group in df.groupby("grid_id"):
    group = group.set_index("date").sort_index()
    daily = group.resample("D").interpolate("linear")
    daily["grid_id"] = grid_id
    daily_frames.append(daily)

# Combine all grid cells
daily_df = pd.concat(daily_frames).reset_index()

# Save to CSV
os.makedirs("data/features", exist_ok=True)
daily_df.to_csv("data/features/daily_ndvi_grid.csv", index=False)
print("✅ Daily NDVI saved: daily_ndvi_grid.csv")
