# import streamlit as st
# import pandas as pd
# import geopandas as gpd
# import folium
# from streamlit_folium import st_folium
# import matplotlib.pyplot as plt
# import os

# # Load shapefile safely
# shapefile_path = "data/shapefiles/kerala_districts.shp"
# if os.path.exists(shapefile_path):
#     districts = gpd.read_file(shapefile_path)
# else:
#     st.warning("⚠️ Kerala district shapefile not found. Map will not render.")
#     districts = pd.DataFrame({"DISTRICT": ["Thrissur", "Idukki", "Palakkad"]})  # fallback

# # Load predictions and alerts
# predictions = pd.read_csv("data/daily_predictions.csv")
# alerts = pd.read_csv("data/alerts.csv")

# # Sidebar
# st.sidebar.title("🔥 Kerala Wildfire Sentinel")
# selected_date = st.sidebar.date_input("Select Date", pd.to_datetime("today"))
# selected_district = st.sidebar.selectbox("Select District", districts["DISTRICT"].unique())

# # Filter data
# date_str = selected_date.strftime("%Y-%m-%d")
# daily_data = predictions[predictions["date"] == date_str]
# district_data = predictions[predictions["district"] == selected_district]

# # Choropleth Map
# st.subheader("🗺️ Fire Risk Map")
# if "geometry" in districts.columns:
#     m = folium.Map(location=[10.5, 76.5], zoom_start=7)
#     for _, row in districts.iterrows():
#         risk_row = daily_data[daily_data["district"] == row["DISTRICT"]]
#         if not risk_row.empty:
#             risk_level = risk_row["risk_level"].values[0]
#             color_map = {"Low": "green", "Medium": "orange", "High": "red"}
#             color = color_map.get(risk_level, "gray")
#             folium.GeoJson(row["geometry"], style_function=lambda x, color=color: {
#                 "fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.6
#             }, tooltip=f"{row['DISTRICT']}: Risk {risk_level}").add_to(m)
#     st_data = st_folium(m, width=700)
# else:
#     st.info("Map view disabled—geometry not available.")

# # Alerts Table
# st.subheader("🔔 High-Risk Alerts")
# alert_today = alerts[alerts["date"] == date_str]
# if alert_today.empty:
#     st.info("No high-risk alerts for this date.")
# else:
#     st.dataframe(alert_today)

# # Risk Trend Plot
# st.subheader(f"📈 Risk Trend for {selected_district}")
# fig, ax = plt.subplots()
# if all(col in district_data.columns for col in ["ndvi", "rh", "dryness_index"]):
#     ax.plot(district_data["date"], district_data["ndvi"], label="NDVI", color="green")
#     ax.plot(district_data["date"], district_data["rh"], label="RH (%)", color="blue")
#     ax.plot(district_data["date"], district_data["dryness_index"], label="Dryness Index", color="red")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Value")
#     ax.legend()
#     st.pyplot(fig)
# else:
#     st.info("Trend data not available for selected district.")

# # Download Button
# st.subheader("📤 Export Daily Predictions")
# csv = daily_data.to_csv(index=False).encode("utf-8")
# st.download_button("Download CSV", csv, f"fire_risk_{date_str}.csv", "text/csv")

# # Footer

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import streamlit_authenticator as stauth
import os

# ---------------------- AUTHENTICATION ----------------------
credentials = {
    "usernames": {
        "anna": {
            "name": "Anna",
            "password": stauth.Hasher(["sentinel123"]).generate()[0]
        },
        "ranger": {
            "name": "Ranger",
            "password": stauth.Hasher(["fieldwatch"]).generate()[0]
        },
        "teacher": {
            "name": "Teacher",
            "password": stauth.Hasher(["reviewpanel"]).generate()[0]
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "wildfire_dashboard",
    "abcdef",
    30
)


# ---------------------- LOGIN UI ----------------------
st.markdown("""
<div style='text-align:center'>
    <h2 style='color:#d9534f;'>Login</h2>
    
</div>
""", unsafe_allow_html=True)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("❌ Incorrect username or password")
elif authentication_status is None:
    st.warning("🔐 Please enter your credentials")
elif authentication_status:

    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Welcome, {name} 👋")

    # ---------------------- HEADER ----------------------
    st.markdown("""
    <div style='text-align:center'>
        <h1 style='color:#d9534f;'>AI-Powered Wildfire Prediction for Kerala</h1>
       
        
    </div>
    """, unsafe_allow_html=True)

    # ---------------------- LOAD DATA ----------------------
    shapefile_path = "data/shapefiles/kerala_districts.shp"
    if os.path.exists(shapefile_path):
        districts = gpd.read_file(shapefile_path)
    else:
        st.warning("⚠️ Kerala district shapefile not found. Map will not render.")
        districts = pd.DataFrame({"DISTRICT": ["Thrissur", "Idukki", "Palakkad"]})  # fallback

    with st.spinner("🔄 Loading prediction and alert data..."):
        predictions = pd.read_csv("data/daily_predictions.csv")
        alerts = pd.read_csv("data/alerts.csv")

    # ---------------------- SIDEBAR ----------------------
    with st.sidebar:
        st.markdown("## 🔧 Controls")
        selected_date = st.date_input("📅 Select Date", pd.to_datetime("today"))
        selected_district = st.selectbox("📍 District", districts["DISTRICT"].unique())
        st.markdown("---")
        st.markdown("### 📤 Export")
        date_str = selected_date.strftime("%Y-%m-%d")
        daily_data = predictions[predictions["date"] == date_str]
        csv = daily_data.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, f"fire_risk_{date_str}.csv", "text/csv")

    # ---------------------- TABS ----------------------
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Risk Map", "🔔 Alerts", "📈 Trends", "🧠 Explainability"])

    # ---------------------- TAB 1: MAP ----------------------
    with tab1:
        st.subheader("🗺️ Fire Risk Map")
        if "geometry" in districts.columns:
            m = folium.Map(location=[10.5, 76.5], zoom_start=7)
            for _, row in districts.iterrows():
                risk_row = daily_data[daily_data["district"] == row["DISTRICT"]]
                if not risk_row.empty:
                    risk_level = risk_row["risk_level"].values[0]
                    color_map = {"Low": "green", "Medium": "orange", "High": "red"}
                    color = color_map.get(risk_level, "gray")
                    folium.GeoJson(row["geometry"], style_function=lambda x, color=color: {
                        "fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.6
                    }, tooltip=f"{row['DISTRICT']}: Risk {risk_level}").add_to(m)
            st_data = st_folium(m, width=700)
        else:
            st.info("Map view disabled—geometry not available.")

    # ---------------------- TAB 2: ALERTS ----------------------
    with tab2:
        st.subheader("🔔 High-Risk Alerts")
        alert_today = alerts[alerts["date"] == date_str]
        if alert_today.empty:
            st.info("No high-risk alerts for this date.")
        else:
            for _, alert in alert_today.iterrows():
                st.markdown(f"""
                <div style='background-color:#fff3cd;padding:10px;border-left:5px solid #ffc107;margin-bottom:10px'>
                    <strong>{alert['district']}</strong>: {alert['alert_type']}<br>
                    <em>{alert['message']}</em>
                </div>
                """, unsafe_allow_html=True)

    # ---------------------- TAB 3: TRENDS ----------------------
    with tab3:
        st.subheader(f"📈 Risk Indicators for {selected_district}")
        district_data = predictions[predictions["district"] == selected_district]
        if all(col in district_data.columns for col in ["ndvi", "rh", "dryness_index"]):
            fig = px.line(
                district_data,
                x="date",
                y=["ndvi", "rh", "dryness_index"],
                labels={"value": "Score", "variable": "Indicator"},
                title=f"NDVI, RH, and Dryness Index Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trend data not available for selected district.")

    # ---------------------- TAB 4: EXPLAINABILITY ----------------------
    with tab4:
        st.subheader("🧠 Why This Risk?")
        st.markdown("""
        <ul>
            <li>🌿 <strong>Low NDVI</strong>: Indicates vegetation stress</li>
            <li>🔥 <strong>High Temperature</strong>: Increases ignition risk</li>
            <li>💨 <strong>Low Relative Humidity</strong>: Dry air accelerates fire spread</li>
            <li>📡 <strong>Recent VIIRS Detections</strong>: Confirms thermal anomalies</li>
        </ul>
        """, unsafe_allow_html=True)

    # ---------------------- FOOTER ----------------------
   