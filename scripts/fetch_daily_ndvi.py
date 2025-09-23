# import ee
# import pandas as pd
# from datetime import date

# ee.Initialize()

# today = date.today().isoformat()
# gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
# kerala_districts = gaul.filter(ee.Filter.eq('ADM1_NAME', 'Kerala'))
# target_districts = ['Thrissur', 'Palakkad', 'Idukki']
# results = []

# # 🛰️ Sentinel-2 surface reflectance
# sentinel = ee.ImageCollection('COPERNICUS/S2_SR') \
#     .filterDate('2025-09-01', today) \
#     .filterBounds(kerala_districts.geometry()) \
#     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
#     .select(['B4', 'B8'])  # Red and NIR

# # 🧠 Compute NDVI: (NIR - Red) / (NIR + Red)
# def compute_ndvi(image):
#     ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
#     return image.addBands(ndvi)

# sentinel_ndvi = sentinel.map(compute_ndvi)
# latest_ndvi = sentinel_ndvi.sort('system:time_start', False).first()

# image_count = sentinel_ndvi.size().getInfo()
# print(f"🛰️ Sentinel-2 NDVI images found: {image_count}")

# if image_count == 0:
#     print("⚠️ No Sentinel-2 NDVI images available in the selected date range.")
# else:
#     for district_name in target_districts:
#         district = kerala_districts.filter(ee.Filter.eq('ADM2_NAME', district_name))

#         try:
#             ndvi_mean = latest_ndvi.select('NDVI').reduceRegion(
#                 reducer=ee.Reducer.mean(),
#                 geometry=district.geometry(),
#                 scale=10,
#                 maxPixels=1e9
#             ).get('NDVI').getInfo()

#             results.append({
#                 'district': district_name,
#                 'date': today,
#                 'ndvi': ndvi_mean
#             })
#         except Exception as e:
#             print(f"⚠️ Error fetching NDVI for {district_name}: {e}")
#             results.append({
#                 'district': district_name,
#                 'date': today,
#                 'ndvi': None
#             })

#     df = pd.DataFrame(results)
#     df.to_csv('data/daily_ndvi.csv', index=False)
#     print("✅ Daily NDVI saved for:", ', '.join(target_districts))

import ee
import geemap
import pandas as pd
from datetime import datetime

# 🌍 Initialize Earth Engine
ee.Initialize()

# 🗺️ Load Kerala districts (replace with your asset path)
kerala_fc = ee.FeatureCollection("users/riarosemj27/kerala_districts")

# 📅 MODIS NDVI (every 16 days)
modis = ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI") \
    .filterDate("2022-09-17", "2025-09-17") \
    .map(lambda img: img.divide(10000).copyProperties(img, ["system:time_start"]))

# 📦 Output list
ndvi_list = []

# 🔁 Loop through images
timestamps = modis.aggregate_array("system:time_start").getInfo()
for ts in timestamps:
    date_str = datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d')
    image = modis.filter(ee.Filter.eq("system:time_start", ts)).first()

    try:
        stats = geemap.zonal_statistics(
            image,
            kerala_fc,
            statistics_type='MEAN',
            scale=250,
            return_fc=False
        )
        stats["date"] = date_str
        stats.rename(columns={"mean": "ndvi"}, inplace=True)
        stats["DISTRICT"] = stats["DISTRICT"].str.title()
        stats = stats[["date", "DISTRICT", "ndvi"]]
        stats.rename(columns={"DISTRICT": "district"}, inplace=True)
        ndvi_list.append(stats)
        print(f"✅ NDVI fetched for {date_str}")
    except Exception as e:
        print(f"❌ Failed for {date_str}: {e}")

# 💾 Save to CSV
if ndvi_list:
    df = pd.concat(ndvi_list, ignore_index=True)
    df.to_csv("data/modis_ndvi_kerala_2022_2025.csv", index=False)
    print("🎉 NDVI saved to data/modis_ndvi_kerala_2022_2025.csv")
else:
    print("⚠️ No NDVI data collected.")
