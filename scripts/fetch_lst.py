import ee
ee.Initialize(project='foresightai-469610')

# Thrissur region
region = ee.Geometry.Point([76.214, 10.527]).buffer(10000)

# Time range
start_date = '2024-01-01'
end_date = '2024-12-31'

# Load MODIS LST with quality band
lst = (
    ee.ImageCollection("MODIS/061/MOD11A1")
    .filterDate(start_date, end_date)
    .filterBounds(region)
)

# Mask out bad-quality pixels
def mask_quality(image):
    qa = image.select('QC_Day')
    mask = qa.eq(0)  # 0 = good quality
    return image.updateMask(mask)

# Apply mask and extract LST
def extract_lst(image):
    image = mask_quality(image)
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000,
        maxPixels=1e13
    )
    kelvin = stats.get('LST_Day_1km')
    # Only convert if kelvin is not null
    return ee.Algorithms.If(
        kelvin,
        ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'LST_C': ee.Number(kelvin).multiply(0.02).subtract(273.15)
        }),
        ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'LST_C': -9999
        })
    )


# Map and export
features = lst.map(extract_lst)
fc = ee.FeatureCollection(features)

task = ee.batch.Export.table.toDrive(
    collection=fc,
    description='LST_Thrissur_2024',
    folder='EarthEngineExports',
    fileNamePrefix='lst_thrissur_2024',
    fileFormat='CSV'
)
task.start()

print("✅ MODIS LST export with quality mask started. Check your Drive > EarthEngineExports folder.")
