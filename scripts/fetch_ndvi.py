import ee
import datetime
import pandas as pd

ee.Initialize()

districts = ee.FeatureCollection("users/riarosemj27/kerala_districts")

# Load MODIS NDVI collection
modis_collection = ee.ImageCollection("MODIS/061/MOD13Q1")

# Print available image dates
dates = modis_collection.aggregate_array('system:time_start').getInfo()
print("🗓️ Available MODIS image timestamps:", dates[:5])

# Get latest image
latest_image = modis_collection.sort('system:time_start', False).first()
ndvi_image = latest_image.select('NDVI').multiply(0.0001)




# Reduce NDVI over districts
def extract_ndvi(feature):
    ndvi = ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=500,
        maxPixels=1e9
    )
    return feature.set({'NDVI': ndvi.get('NDVI')})

ndvi_per_district = districts.map(extract_ndvi)


# Convert to DataFrame
features = ndvi_per_district.getInfo()['features']
timestamp = latest_image.get('system:time_start').getInfo()
date_str = datetime.datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

data = []
for f in features:
    props = f['properties']
    district = props.get('DISTRICT', 'Unknown')
    ndvi = props.get('NDVI', None)
    data.append({
        'district': district,
        'ndvi': ndvi if ndvi is not None else 'NA',
        'date': date_str  # ✅ Add this line inside the loop
    })

df = pd.DataFrame(data)
df.to_csv("data/ndvi/modis_ndvi_debug.csv", index=False)
print("✅ MODIS NDVI debug data saved.")


# data.append({
#     'district': district,
#     'ndvi': ndvi if ndvi is not None else 'NA',
#     'date': date_str
# })
# df = pd.DataFrame(data)
# df.to_csv("data/ndvi/modis_ndvi.csv", index=False)
# print(f"✅ MODIS NDVI data saved for {date_str}")

