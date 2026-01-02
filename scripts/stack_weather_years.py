import pandas as pd
import os

try:
    print("📂 Checking file paths...")

    base_path = "data/"
    files = [
        "Kerala_Grid_Weather_2022.csv",
        "Kerala_Grid_Weather_2023.csv",
        "Kerala_Grid_Weather_2024.csv"
    ]

    for f in files:
        full_path = os.path.join(base_path, f)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"❌ Missing file: {full_path}")
        else:
            print(f"✅ Found: {f}")

    print("\n📊 Reading CSVs...")
    weather_2022 = pd.read_csv(os.path.join(base_path, files[0]))
    print(f"✅ Loaded 2022: {len(weather_2022)} rows")

    weather_2023 = pd.read_csv(os.path.join(base_path, files[1]))
    print(f"✅ Loaded 2023: {len(weather_2023)} rows")

    weather_2024 = pd.read_csv(os.path.join(base_path, files[2]))
    print(f"✅ Loaded 2024: {len(weather_2024)} rows")

    print("\n🔗 Concatenating all years...")
    weather_all = pd.concat([weather_2022, weather_2023, weather_2024], ignore_index=True)
    print(f"📦 Total rows combined: {len(weather_all)}")

    output_path = os.path.join(base_path, "weather_grid_daily.csv")
    weather_all.to_csv(output_path, index=False)
    print(f"\n✅ Combined weather file saved: {output_path}")

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
