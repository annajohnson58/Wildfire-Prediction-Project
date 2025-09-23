import pandas as pd
import os

# 📂 Paths to your daily files
data_folder = 'data/'
ndvi_file = os.path.join(data_folder, 'daily_ndvi.csv')
climate_file = os.path.join(data_folder, 'daily_climate.csv')

# 📊 Store merged rows
merged_rows = []

try:
    # 🔄 Load NDVI
    ndvi_df = pd.read_csv(ndvi_file)
    ndvi_date = ndvi_df['date'].iloc[0]  # assumes all rows share the same date
    print(f"📅 NDVI date: {ndvi_date}")

    # 🔄 Load climate
    climate_df = pd.read_csv(climate_file)
    climate_date = climate_df['date'].iloc[0]
    print(f"📅 Climate date: {climate_date}")

    if ndvi_date != climate_date:
        print("⚠️ NDVI and climate dates do not match. Skipping merge.")
    else:
        # 🧠 Aggregate climate
        climate_mean = climate_df[['t2m', 'tp', 'u10', 'v10']].mean().to_dict()

        # 🧬 Merge climate into NDVI
        for key in climate_mean:
            ndvi_df[key] = climate_mean[key]

        ndvi_df['date'] = ndvi_date
        merged_rows.append(ndvi_df)

except Exception as e:
    print(f"❌ Error during merge: {e}")

# 🧾 Append to historical file
if merged_rows:
    merged_df = pd.concat(merged_rows, ignore_index=True)

    # 🔄 Load existing historical file if it exists
    historical_path = os.path.join(data_folder, 'historical_ndvi_climate.csv')
    if os.path.exists(historical_path):
        historical_df = pd.read_csv(historical_path)
        historical_df = pd.concat([historical_df, merged_df], ignore_index=True)
    else:
        historical_df = merged_df

    # 💾 Save updated historical file
    historical_df.to_csv(historical_path, index=False)
    print("✅ historical_ndvi_climate.csv updated.")
else:
    print("❌ No data merged. Check file contents and date alignment.")
