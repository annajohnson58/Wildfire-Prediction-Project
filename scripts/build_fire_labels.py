# import pandas as pd

# # Load both fire files
# fires1 = pd.read_csv('data/firms_fire_points_1.csv')
# fires2 = pd.read_csv('data/firms_fire_points_2.csv')

# # Combine them
# fires = pd.concat([fires1, fires2], ignore_index=True)

# # Filter for Kerala bounding box and confidence ≥ 50
# kerala_box = {'lat_min': 8.0, 'lat_max': 12.8, 'lon_min': 74.8, 'lon_max': 77.5}
# fires_kerala = fires[
#     (fires['latitude'] >= kerala_box['lat_min']) &
#     (fires['latitude'] <= kerala_box['lat_max']) &
#     (fires['longitude'] >= kerala_box['lon_min']) &
#     (fires['longitude'] <= kerala_box['lon_max']) &
#     (fires['confidence'] >= 50)
# ].copy()

# fires_kerala['date'] = pd.to_datetime(fires_kerala['acq_date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')

# district_boxes = {
#     'Thrissur':  [10.3, 76.0, 10.7, 76.4],
#     'Palakkad':  [10.5, 76.4, 10.9, 76.9],
#     'Idukki':    [9.5, 76.8, 10.1, 77.2],
#     # Add more districts as needed
# }

# def assign_district(row):
#     for district, box in district_boxes.items():
#         if box[0] <= row['latitude'] <= box[2] and box[1] <= row['longitude'] <= box[3]:
#             return district
#     return None

# fires_kerala['district'] = fires_kerala.apply(assign_district, axis=1)
# fires_kerala = fires_kerala.dropna(subset=['district'])


# fire_labels = fires_kerala.groupby(['date', 'district']).size().reset_index(name='fire_occurred')
# fire_labels['fire_occurred'] = 1  # binary label

# features = pd.read_csv('data/historical_ndvi_climate.csv')  # includes date + district + ndvi + t2m + tp + u10 + v10

# labeled = features.merge(fire_labels, on=['date', 'district'], how='left')
# labeled['fire_occurred'] = labeled['fire_occurred'].fillna(0).astype(int)

# labeled.to_csv('data/historical_fire_data.csv', index=False)
# print("✅ Labeled dataset saved as historical_fire_data.csv")

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# 📂 Paths
firms_path = "data/fire_archive_SV-C2_671063.csv"  # Your VIIRS CSV
districts_path = "data/shapefiles/kerala_districts.shp"
output_path = "data/daily_predictions.csv"

# 🛰️ Load VIIRS fire detections
firms = pd.read_csv(firms_path)
firms["date"] = pd.to_datetime(firms["acq_date"])
firms["geometry"] = [Point(xy) for xy in zip(firms["longitude"], firms["latitude"])]
gdf = gpd.GeoDataFrame(firms, geometry="geometry", crs="EPSG:4326")

# 🗺️ Load Kerala districts
districts = gpd.read_file(districts_path).to_crs("EPSG:4326")

# 🔗 Spatial join: assign each fire point to a district
joined = gpd.sjoin(gdf, districts, how="inner", predicate="intersects")

# 📊 Group by date and district
daily = joined.groupby(["date", "DISTRICT"]).size().reset_index(name="thermal_count")

# 🔥 Add fire_label (1 if fire detected, else 0)
daily["fire_label"] = (daily["thermal_count"] > 0).astype(int)

# 📤 Save to CSV
os.makedirs("data", exist_ok=True)
daily.to_csv(output_path, index=False)
print("✅ Saved daily_predictions.csv with", len(daily), "rows.")
