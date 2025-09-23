import pandas as pd
import folium
import os

# === CONFIG ===
INPUT_PATH = "data/india_firms_fire_data.csv"
OUTPUT_MAP = "data/india_fire_map.html"

def create_fire_map(df):
    m = folium.Map(location=[22.0, 78.0], zoom_start=5)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color='red',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    return m

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    try:
        df = pd.read_csv(INPUT_PATH)
        fire_map = create_fire_map(df)
        fire_map.save(OUTPUT_MAP)
        print(f"✅ Interactive map saved to: {OUTPUT_MAP}")
    except Exception as e:
        print(f"⚠️ Error during map creation: {e}")

if __name__ == "__main__":
    main()
