import pandas as pd

ndvi_df = pd.read_csv("data/ndvi/modis_ndvi_debug.csv")
weather_df = pd.read_csv("data/weather/district_weather_2025-09-11.csv")

ndvi_df['district'] = ndvi_df['district'].str.strip().str.lower()
weather_df['DISTRICT'] = weather_df['DISTRICT'].str.strip().str.lower()

merged = pd.merge(
    ndvi_df,
    weather_df,
    left_on='district',
    right_on='DISTRICT',
    how='inner'
)
fused = merged[[
    'district',
    'ndvi',
    'temp',
    'rh',
    'wind',
    'precip'
]].copy()

fused['date_ndvi'] = merged['date_x']
fused['date_weather'] = merged['date_y']
fused.to_csv("data/fused/fused_ndvi_weather.csv", index=False)
print("✅ Fused NDVI + weather data saved.")
