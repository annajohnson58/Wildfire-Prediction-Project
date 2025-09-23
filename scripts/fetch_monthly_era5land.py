import ee
ee.Initialize(project='foresightai-469610')

# Define district centers
districts = {
    'Thrissur': [76.214, 10.527],
    'Palakkad': [76.6548, 10.7867],
    'Idukki': [77.0972, 9.8490]
}

# Time range
start = ee.Date('2024-01-01')
end = ee.Date('2025-01-01')

# ERA5-Land monthly collection
era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").filterDate(start, end)

# Grid sampling function
def generate_grid(center, spacing_km=10, count=3):
    lon, lat = center
    points = []
    for dx in range(-count, count + 1):
        for dy in range(-count, count + 1):
            new_lon = lon + dx * spacing_km / 111.32
            new_lat = lat + dy * spacing_km / 110.57
            points.append(ee.Feature(ee.Geometry.Point([new_lon, new_lat])))
    return ee.FeatureCollection(points)

# Loop over districts
for name, center in districts.items():
    print(f"🚀 Sampling monthly ERA5-Land for {name}...")

    grid = generate_grid(center)

    def extract_features(img):
        date = img.date().format('YYYY-MM')
        sampled = img.sampleRegions(collection=grid, scale=1000, geometries=True)
        sampled = sampled.map(lambda f: f.set('month', date))
        return sampled

    monthly_samples = era5.map(extract_features).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=monthly_samples,
        description=f'Monthly_ERA5Land_Grid_{name}_2024',
        folder='EarthEngineExports',
        fileNamePrefix=f'monthly_weather_grid_{name.lower()}_2024',
        fileFormat='CSV'
    )
    task.start()
    print(f"✅ Export started for {name}. Check your Drive > EarthEngineExports folder.")
