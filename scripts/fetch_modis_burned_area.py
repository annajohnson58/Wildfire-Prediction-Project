import ee
ee.Initialize(project='foresightai-469610')

# Thrissur region (10 km buffer)
region = ee.Geometry.Point([76.214, 10.527]).buffer(10000)

# Time range
start_date = '2024-01-01'
end_date = '2024-12-31'

# Load MODIS Burned Area
burned = (
    ee.ImageCollection("MODIS/061/MCD64A1")
    .filterDate(start_date, end_date)
    .filterBounds(region)
    .select('BurnDate')
)

# Extract burn day (if any) per month
def extract_burn(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=500,
        maxPixels=1e13
    )
    return ee.Feature(None, {
        'month': image.date().format('YYYY-MM'),
        'burn_histogram': stats.get('BurnDate')
    })

# Map and export
features = burned.map(extract_burn)
fc = ee.FeatureCollection(features)

task = ee.batch.Export.table.toDrive(
    collection=fc,
    description='MODIS_BurnedArea_Thrissur_2024',
    folder='EarthEngineExports',
    fileNamePrefix='modis_burned_thrissur_2024',
    fileFormat='CSV'
)
task.start()

print("✅ MODIS burned area export started. Check your Drive > EarthEngineExports folder.")
