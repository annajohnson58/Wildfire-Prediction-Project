import requests
import pandas as pd
from datetime import datetime, timedelta

# === CONFIG ===
OUTPUT_PATH = "data/weather_data.csv"
START_DATE = "2025-08-01"
END_DATE = "2025-08-13"

REGIONS = {
    "West": (21.0, 72.0),
    "Central": (22.5, 78.5),
    "East": (23.0, 85.0),
    "Far East": (26.0, 93.0)
}

def fetch_weather(lat, lon, start, end):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        f"&timezone=Asia/Kolkata"
    )
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['daily'])
    return df

def main():
    all_weather = []

    for region, (lat, lon) in REGIONS.items():
        print(f"📡 Fetching weather for {region}...")
        try:
            df = fetch_weather(lat, lon, START_DATE, END_DATE)
            df['region'] = region
            all_weather.append(df)
        except Exception as e:
            print(f"⚠️ Failed for {region}: {e}")

    if all_weather:
        full_df = pd.concat(all_weather)
        full_df.rename(columns={"time": "acq_date"}, inplace=True)
        full_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Weather data saved to: {OUTPUT_PATH}")
        print(full_df.head())
    else:
        print("❌ No weather data fetched.")

if __name__ == "__main__":
    main()
