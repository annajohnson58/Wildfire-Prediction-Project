# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import st_folium
# from folium.plugins import MarkerCluster

# # === Load Data ===
# df = pd.read_csv("data/exports/grid_predictions_Dplus2_realtime.csv")

# # === Sidebar Filters ===
# st.sidebar.title("🔥 Wildfire Forecast Dashboard")
# st.sidebar.markdown(f"**Forecast Date:** {df['forecast_date'].iloc[0]}")

# # === Summary Stats ===
# st.title("🌲 Foresight for Forests: Fire Risk Overview")
# col1, col2, col3 = st.columns(3)
# col1.metric("Total Grids", len(df))
# col2.metric("Predicted Fires", df['fire_pred'].sum())
# col3.metric("Avg NDVI", round(df['ndvi'].mean(), 2))

# # === Map ===
# st.subheader("🗺️ Fire Risk Map")

# m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=7)
# marker_cluster = MarkerCluster().add_to(m)

# for _, row in df.iterrows():
#     color = "red" if row["fire_pred"] == 1 else "green"
#     folium.CircleMarker(
#         location=[row["lat"], row["lon"]],
#         radius=4,
#         color=color,
#         fill=True,
#         fill_opacity=0.7,
#         popup=f"Grid: {row['grid_id']}<br>Fire Prob: {row['fire_prob']:.3f}"
#     ).add_to(marker_cluster)

# st_data = st_folium(m, width=700, height=500)

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from datetime import datetime

# === Page Config ===
st.set_page_config(page_title="Kerala Wildfire D+2 Forecast", layout="wide")

# === Auto-refresh every 60 seconds ===
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="refresh")
except:
    pass  # fallback if package not available

# === Load Data ===
df = pd.read_csv("data/exports/grid_predictions_Dplus2_realtime.csv")

# === Sidebar Filters ===
st.sidebar.title(" Wildfire Forecast Dashboard")
st.sidebar.markdown(f"**Forecast Date:** {df['forecast_date'].iloc[0]}")

threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.3, 0.05)
severity_filter = st.sidebar.multiselect("Severity Tier", ["Low", "Medium", "High"], default=["Low","Medium","High"])

# === Severity Tier Mapping ===
def tier(x):
    if pd.isna(x): return "Unknown"
    if x < 0.33: return "Low"
    elif x < 0.66: return "Medium"
    else: return "High"

if "severity_pred" in df.columns:
    df["severity_tier"] = df["severity_pred"].apply(tier)
else:
    df["severity_tier"] = "Unknown"

# === Summary Stats ===
st.title(" Foresight for Forests: Fire Risk Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Grids", len(df))
col2.metric("Predicted Fires", int((df["fire_prob"] >= threshold).sum()))
col3.metric("Avg NDVI", round(df["ndvi"].mean(), 2))
col4.metric("High Severity Alerts", int(((df["fire_prob"] >= threshold) & (df["severity_tier"] == "High")).sum()))

# === Alerts Panel ===
st.subheader(" Alerts")
alerts = df[(df["fire_prob"] >= threshold) & (df["severity_tier"].isin(severity_filter)) & (df["fire_pred"] == 1)]
st.write(f"Triggered alerts: {len(alerts)}")
st.dataframe(
    alerts[["grid_id", "lat", "lon", "fire_prob", "severity_tier"]],
    width="stretch"
)

# === Map ===
st.subheader(" Fire Risk Map")
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=7, tiles="CartoDB positron")
marker_cluster = MarkerCluster().add_to(m)

def color_for_row(row):
    if row["fire_prob"] < threshold:
        return "#2c7bb6"  # blue for safe
    if row["severity_tier"] == "High": return "#d7191c"   # red
    if row["severity_tier"] == "Medium": return "#fdae61" # orange
    if row["severity_tier"] == "Low": return "#abdda4"    # green
    return "gray"

# Always iterate over full dataset (like old version)
for _, row in df.iterrows():
    color = color_for_row(row)
    popup_html = f"""
    <b>Grid:</b> {row['grid_id']}<br>
    <b>Fire Prob:</b> {row['fire_prob']:.3f}<br>
    <b>Severity:</b> {row['severity_tier']}<br>
    <b>NDVI:</b> {row['ndvi']:.2f}<br>
    <b>Coords:</b> {row['lat']:.4f}, {row['lon']:.4f}<br>
    <b>Forecast:</b> {df['forecast_date'].iloc[0]}
    """
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(marker_cluster)

st_data = st_folium(m, width=900, height=520)
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


