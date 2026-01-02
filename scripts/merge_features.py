import pandas as pd
import os

try:
    print("📂 Loading input files...")

    ndvi_path = "data/features/daily_ndvi_grid.csv"
    thermal_path = "data/features/thermal_flag_grid_daily.csv"
    weather_path = "data/features/weather_grid_daily.csv"

    for path in [ndvi_path, thermal_path, weather_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Missing file: {path}")
        else:
            print(f"✅ Found: {path}")

    ndvi = pd.read_csv(ndvi_path, parse_dates=["date"])
    print(f"📈 NDVI rows: {len(ndvi)}")

    thermal = pd.read_csv(thermal_path, parse_dates=["date"])
    print(f"🔥 Thermal flag rows: {len(thermal)}")

    weather = pd.read_csv(weather_path, parse_dates=["date"])
    print(f"🌧️ Weather rows: {len(weather)}")

    print("\n🔗 Merging NDVI + Weather...")
    merged = pd.merge(ndvi, weather, on=["grid_id", "date"], how="inner")
    print(f"✅ After NDVI + Weather merge: {len(merged)} rows")

    print("🔗 Merging with Thermal Flags...")
    merged = pd.merge(merged, thermal, on=["grid_id", "date"], how="left")
    merged["thermal_flag"] = merged["thermal_flag"].fillna(0).astype(int)
    print(f"✅ Final merged rows: {len(merged)}")

    output_path = "data/features/final_feature_matrix.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n✅ Final feature matrix saved: {output_path}")

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
