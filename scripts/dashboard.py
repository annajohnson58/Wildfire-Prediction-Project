
# import streamlit as st
# import pandas as pd
# import geopandas as gpd
# import folium
# from streamlit_folium import st_folium
# import plotly.express as px
# import streamlit_authenticator as stauth
# import os

# # ---------------------- AUTHENTICATION ----------------------
# credentials = {
#     "usernames": {
#         "anna": {
#             "name": "Anna",
#             "password": stauth.Hasher(["sentinel123"]).generate()[0]
#         },
#         "ranger": {
#             "name": "Ranger",
#             "password": stauth.Hasher(["fieldwatch"]).generate()[0]
#         },
#         "teacher": {
#             "name": "Teacher",
#             "password": stauth.Hasher(["reviewpanel"]).generate()[0]
#         }
#     }
# }

# authenticator = stauth.Authenticate(
#     credentials,
#     "wildfire_dashboard",
#     "abcdef",
#     30
# )

# # ---------------------- LOGIN UI ----------------------
# st.markdown("<h2 style='text-align:center;color:#d9534f;'>Login</h2>", unsafe_allow_html=True)
# name, authentication_status, username = authenticator.login("Login", "main")

# if authentication_status is None:
#     st.warning("Please enter your credentials")
#     st.stop()

# elif authentication_status is False:
#     st.error("Incorrect username or password")
#     st.stop()

# # ---------------------- DASHBOARD ----------------------
# authenticator.logout("Logout", "sidebar")
# st.sidebar.success(f"Welcome, {name}")

# # ---------------------- HEADER ----------------------
# st.markdown("<h1 style='text-align:center;color:#d9534f;'>Kerala Wildfire Risk Dashboard</h1>", unsafe_allow_html=True)

# # ---------------------- LOAD DATA ----------------------
# shapefile_path = "data/shapefiles/kerala_districts.shp"
# if os.path.exists(shapefile_path):
#     districts = gpd.read_file(shapefile_path)
#     districts["DISTRICT"] = districts["DISTRICT"].str.title()
# else:
#     st.warning("Kerala district shapefile not found. Map will not render.")
#     districts = pd.DataFrame({"DISTRICT": ["Thrissur", "Idukki", "Palakkad"]})

# with st.spinner("Loading prediction data..."):
#     predictions = pd.read_csv("data/predictions_ensemble.csv", parse_dates=["date"], dayfirst=True)
#     predictions["date"] = pd.to_datetime(predictions["date"]).dt.date
#     predictions["district"] = predictions["district"].str.title()

# # ---------------------- SIDEBAR ----------------------
# with st.sidebar:
#     st.markdown("## Controls")
#     selected_date = st.date_input("Select Date", pd.to_datetime("2023-03-03").date())
#     selected_district = st.selectbox("District", sorted(predictions["district"].unique()))
#     st.markdown("---")
#     st.markdown("### Export")
#     daily_data = predictions[predictions["date"] == selected_date]
#     csv = daily_data.to_csv(index=False).encode("utf-8")
#     st.download_button("Download CSV", csv, f"fire_risk_{selected_date}.csv", "text/csv")

# # ---------------------- TABS ----------------------
# tab1, tab2, tab3, tab4 = st.tabs(["Map", "Alerts", "Trends", "Drivers"])

# # ---------------------- TAB 1: MAP ----------------------
# with tab1:
#     st.subheader("District-Level Fire Risk")
#     st.markdown("Each district is color-coded based on predicted fire risk. Hover for details.")
#     if "geometry" in districts.columns:
#         m = folium.Map(location=[10.5, 76.5], zoom_start=7)
#         for _, row in districts.iterrows():
#             risk_row = daily_data[daily_data["district"] == row["DISTRICT"]]
#             if not risk_row.empty:
#                 prob = risk_row["probability"].values[0]
#                 risk_level = "High" if prob > 0.8 else "Medium" if prob > 0.5 else "Low"
#                 color_map = {"Low": "green", "Medium": "orange", "High": "red"}
#                 color = color_map[risk_level]
#                 folium.GeoJson(row["geometry"], style_function=lambda x, color=color: {
#                     "fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.6
#                 }, tooltip=f"{row['DISTRICT']}: Risk {risk_level} ({prob:.2f})").add_to(m)
#         st_data = st_folium(m, width=700)
#     else:
#         st.info("Map view disabled—geometry not available.")

# # ---------------------- TAB 2: ALERTS ----------------------
# with tab2:
#     st.subheader("High-Risk Alerts")
#     high_risk_today = daily_data[daily_data["probability"] > 0.8]
#     if high_risk_today.empty:
#         st.info("No high-risk alerts for this date.")
#     else:
#         for _, alert in high_risk_today.iterrows():
#             st.markdown(f"""
#             <div style='background-color:#fff3cd;padding:10px;border-left:5px solid #ffc107;margin-bottom:10px'>
#                 <strong>{alert['district']}</strong><br>
#                 Surge Probability: <code>{alert['probability']:.2f}</code><br>
#                 Predicted Surge: <code>{alert['predicted']}</code>
#             </div>
#             """, unsafe_allow_html=True)

# # ---------------------- TAB 3: TRENDS ----------------------
# with tab3:
#     st.subheader(f"Surge Probability Over Time: {selected_district}")
#     district_data = predictions[predictions["district"] == selected_district]
#     if all(col in district_data.columns for col in ["date", "probability"]):
#         fig = px.line(
#             district_data,
#             x="date",
#             y="probability",
#             labels={"probability": "Surge Probability"},
#             title=f"Surge Probability Trend"
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Trend data not available for selected district.")

# # ---------------------- TAB 4: DRIVERS ----------------------
# with tab4:
#     st.subheader("Key Risk Drivers")
#     st.markdown("""
#     <ul>
#         <li><strong>Low NDVI</strong>: Indicates vegetation stress</li>
#         <li><strong>High Temperature</strong>: Increases ignition risk</li>
#         <li><strong>Low Relative Humidity</strong>: Dry air accelerates fire spread</li>
#         <li><strong>Recent Thermal Anomalies</strong>: Confirmed by satellite detections</li>
#     </ul>
#     """, unsafe_allow_html=True)

# # ---------------------- FOOTER ----------------------
# st.markdown("""
# <hr>
# <div style='text-align:center;font-size:14px;color:gray'>
# Kerala Wildfire Risk System | Powered by AI | Phase 1
# </div>
# """, unsafe_allow_html=True)


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
st.markdown("<h2 style='text-align:center;color:#d9534f;'>Login</h2>", unsafe_allow_html=True)
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is None:
    st.warning("Please enter your credentials")
    st.stop()

elif authentication_status is False:
    st.error("Incorrect username or password")
    st.stop()

# ---------------------- DASHBOARD ----------------------
authenticator.logout("Logout", "sidebar")
st.sidebar.success(f"Welcome, {name}")

# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align:center;color:#d9534f;'>Kerala Wildfire Risk & Severity Dashboard</h1>", unsafe_allow_html=True)

# ---------------------- LOAD DATA ----------------------
shapefile_path = "data/shapefiles/kerala_districts.shp"
if os.path.exists(shapefile_path):
    districts = gpd.read_file(shapefile_path)
    districts["DISTRICT"] = districts["DISTRICT"].str.title()
else:
    st.warning("Kerala district shapefile not found. Map will not render.")
    districts = pd.DataFrame({"DISTRICT": ["Thrissur", "Idukki", "Palakkad"]})

with st.spinner("Loading prediction data..."):
    # IMPORTANT: use the new file with severity outputs
    predictions = pd.read_csv("data/predictions_with_severity.csv", parse_dates=["date"])
    predictions["date"] = pd.to_datetime(predictions["date"]).dt.date
    predictions["district"] = predictions["district"].str.title()

    # Safety: fill missing severity columns if any
    if "severity_pred" not in predictions.columns:
        predictions["severity_pred"] = "Low"
    if "prob_high" not in predictions.columns:
        predictions["prob_high"] = 0.0
    if "prob_moderate" not in predictions.columns:
        predictions["prob_moderate"] = 0.0
    if "prob_low" not in predictions.columns:
        predictions["prob_low"] = 0.0

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.markdown("## Controls")
    if not predictions.empty:
        default_date = sorted(predictions["date"].unique())[0]
    else:
        default_date = pd.to_datetime("2023-03-03").date()

    selected_date = st.date_input("Select Date", default_date)
    selected_district = st.selectbox("District", sorted(predictions["district"].unique()))

    st.markdown("---")
    st.markdown("### Export (Selected Date)")
    daily_data = predictions[predictions["date"] == selected_date]

    export_cols = [
        "district", "date",
        "probability", "predicted",
        "severity_pred", "prob_high", "prob_moderate", "prob_low"
    ]
    export_cols = [c for c in export_cols if c in daily_data.columns]
    daily_export = daily_data[export_cols]

    csv = daily_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"fire_risk_{selected_date}.csv", "text/csv")

# ---------------------- TABS ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Map", "Surge Alerts", "Severity Alerts", "Trends", "Drivers"])

# ---------------------- TAB 1: MAP ----------------------
with tab1:
    st.subheader("District-Level Fire Severity Map")
    st.markdown("Each district is color-coded based on predicted fire severity for the selected date.")

    if "geometry" in districts.columns:
        m = folium.Map(location=[10.5, 76.5], zoom_start=7)

        # Color mapping for severity
        color_map = {"Low": "green", "Moderate": "orange", "High": "red"}

        for _, row in districts.iterrows():
            risk_row = daily_data[daily_data["district"] == row["DISTRICT"]]
            if not risk_row.empty:
                severity = risk_row["severity_pred"].values[0]
                prob_high = risk_row["prob_high"].values[0] if "prob_high" in risk_row.columns else None

                color = color_map.get(severity, "gray")

                tooltip_text = f"{row['DISTRICT']}: Severity {severity}"
                if prob_high is not None:
                    tooltip_text += f" (High Sev Prob: {prob_high:.2f})"

                folium.GeoJson(
                    row["geometry"],
                    style_function=lambda x, color=color: {
                        "fillColor": color,
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.6
                    },
                    tooltip=tooltip_text
                ).add_to(m)

        st_data = st_folium(m, width=700, height=500)
        st.markdown("""
        **Severity Legend**  
        🟢 Low  
        🟠 Moderate  
        🔴 High  
        """)
    else:
        st.info("Map view disabled—geometry not available.")

# ---------------------- TAB 2: SURGE ALERTS ----------------------
with tab2:
    st.subheader("High-Risk Surge Alerts")
    if "probability" in daily_data.columns:
        high_risk_today = daily_data[daily_data["probability"] > 0.8]
        if high_risk_today.empty:
            st.info("No high-risk surge alerts for this date.")
        else:
            for _, alert in high_risk_today.iterrows():
                st.markdown(f"""
                <div style='background-color:#fff3cd;padding:10px;border-left:5px solid #ffc107;margin-bottom:10px'>
                    <strong>{alert['district']}</strong><br>
                    Surge Probability: <code>{alert['probability']:.2f}</code><br>
                    Predicted Surge: <code>{alert['predicted']}</code>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Surge probability data not available.")

# ---------------------- TAB 3: SEVERITY ALERTS ----------------------
with tab3:
    st.subheader("High Severity Alerts")
    severe_today = daily_data[daily_data["severity_pred"] == "High"]

    action_map = {
        "Low": "Routine monitoring and patrolling.",
        "Moderate": "Deploy local forest watchers and prepare fire lines.",
        "High": "Mobilize rapid response teams, water units, and coordinate with district control room."
    }

    if severe_today.empty:
        st.info("No high-severity alerts for this date.")
    else:
        for _, alert in severe_today.iterrows():
            prob_high = alert["prob_high"] if "prob_high" in alert else 0.0
            sev = alert["severity_pred"]

            st.markdown(f"""
            <div style='background-color:#f8d7da;padding:10px;border-left:5px solid #dc3545;margin-bottom:10px'>
                <strong>{alert['district']}</strong><br>
                Severity: <code>{sev}</code><br>
                High Severity Probability: <code>{prob_high:.2f}</code><br>
                Recommended Action: <strong>{action_map.get(sev, "Monitor conditions closely.")}</strong>
            </div>
            """, unsafe_allow_html=True)

# ---------------------- TAB 4: TRENDS ----------------------
with tab4:
    st.subheader(f"Surge & Severity Trends: {selected_district}")

    district_data = predictions[predictions["district"] == selected_district]

    # Surge Probability Trend
    if all(col in district_data.columns for col in ["date", "probability"]):
        fig1 = px.line(
            district_data,
            x="date",
            y="probability",
            labels={"probability": "Surge Probability"},
            title="Surge Probability Over Time"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Surge trend data not available for selected district.")

    # Severity (High Severity Probability) Trend
    if all(col in district_data.columns for col in ["date", "prob_high"]):
        fig2 = px.line(
            district_data,
            x="date",
            y="prob_high",
            labels={"prob_high": "High Severity Probability"},
            title="High Severity Probability Over Time"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Severity trend data not available for selected district.")

# ---------------------- TAB 5: DRIVERS ----------------------
with tab5:
    st.subheader("Key Risk Drivers")
    st.markdown("""
    <ul>
        <li><strong>Low NDVI</strong>: Indicates vegetation stress and dry fuel availability.</li>
        <li><strong>High Temperature</strong>: Increases ignition likelihood and fire intensity.</li>
        <li><strong>Low Rainfall</strong>: Leads to drier conditions and higher severity potential.</li>
        <li><strong>High Wind Speed</strong>: Accelerates fire spread and makes suppression harder.</li>
    </ul>
    """, unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("""
<hr>
<div style='text-align:center;font-size:14px;color:gray'>
Kerala Wildfire Risk & Severity System | Powered by AI | Phase 2
</div>
""", unsafe_allow_html=True)
