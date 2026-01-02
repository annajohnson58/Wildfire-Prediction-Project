# import pandas as pd
# import requests
# import joblib
# import os
# import time
# from datetime import datetime, timedelta

# # === CONFIG ===
# API_KEY = "3a4ade61299245585231113be187c820"  # Replace with your actual key
# GRID_CSV = "data/grid_features_realtime.csv"  # Use enriched features
# FIRE_MODEL_PATH = "models/fire_model_v3.pkl"
# SEVERITY_MODEL_PATH = "models/severity_model_v3.pkl"
# EXPORT_PATH = "data/exports/grid_predictions_Dplus2_realtime.csv"

# # === Load Grid Locations ===
# grids = pd.read_csv(GRID_CSV)
# print(f"Loaded {len(grids)} grid locations.")

# # === Fetch Weather Forecast for Each Grid ===
# def fetch_weather(lat, lon, retries=3, delay=2):
#     url = (
#         f"https://api.openweathermap.org/data/2.5/forecast?"
#         f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
#     )
#     for attempt in range(retries):
#         try:
#             r = requests.get(url, timeout=10)
#             data = r.json()
#             if "list" not in data or len(data["list"]) < 17:
#                 raise ValueError("Incomplete forecast data")
#             forecast = data["list"][16]
#             return {
#                 "temperature": forecast["main"]["temp"],
#                 "rainfall": forecast.get("rain", {}).get("3h", 0),
#                 "wind": forecast["wind"]["speed"],
#                 "humidity": forecast["main"]["humidity"]
#             }
#         except Exception as e:
#             print(f"⚠️  Attempt {attempt+1} failed for lat={lat}, lon={lon}: {e}")
#             time.sleep(delay)
#     print(f"❌ Skipping lat={lat}, lon={lon} after {retries} retries.")
#     return None

# # === Build Feature Table ===
# features = []
# for _, row in grids.iterrows():
#     weather = fetch_weather(row["lat"], row["lon"])
#     if weather is None:
#         continue
#     features.append({
#         "grid_id": row["grid_id"],
#         "ndvi": row.get("ndvi", 0.4),
#         "temperature": weather["temperature"],
#         "rainfall": weather["rainfall"],
#         "wind": weather["wind"],
#         "dryness_index": 100 - weather["humidity"],
#         "firs_hotspots": row.get("firs_hotspots", 0),
#         "dry_lag1": row.get("dry_lag1", 0)
#     })

# df = pd.DataFrame(features)
# print(f"Prepared features for {len(df)} grids.")

# # === Load Models ===
# fire_model = joblib.load(FIRE_MODEL_PATH)
# severity_model = joblib.load(SEVERITY_MODEL_PATH)

# # === Predict Fire Occurrence ===
# X = df[["ndvi", "temperature", "rainfall", "wind", "dryness_index", "firs_hotspots", "dry_lag1"]]
# fire_proba = fire_model.predict_proba(X)[:, 1]
# df["fire_prob"] = fire_proba
# df["fire_pred"] = (fire_proba >= 0.3).astype(int)

# # === Predict Severity for Fire-Positive Grids ===
# fire_positive = df[df["fire_pred"] == 1]
# if not fire_positive.empty:
#     X_sev = fire_positive[["ndvi", "temperature", "rainfall", "wind", "dryness_index", "firs_hotspots", "dry_lag1"]]
#     sev_pred = severity_model.predict(X_sev)

#     # Handle 2D outputs
#     if len(sev_pred.shape) > 1:
#         if sev_pred.shape[1] == 1:
#             sev_pred = sev_pred.ravel()
#         elif sev_pred.shape[1] == 2:
#             sev_pred = sev_pred[:, 1]  # Use probability of class 1
#         else:
#             raise ValueError(f"Unexpected severity prediction shape: {sev_pred.shape}")

#     fire_positive["severity_pred"] = sev_pred
#     df = df.merge(fire_positive[["grid_id", "severity_pred"]], on="grid_id", how="left")
# else:
#     df["severity_pred"] = None

# # === Save Predictions ===
# df["forecast_date"] = (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d")
# os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
# df.to_csv(EXPORT_PATH, index=False)
# print(f"\n✅ Predictions saved to {EXPORT_PATH}")

# import pandas as pd
# import requests
# import joblib
# import os
# import time
# from datetime import datetime, timedelta

# # === CONFIG ===
# API_KEY = "3a4ade61299245585231113be187c820"
# GRID_CSV = "data/grid_features_realtime.csv"
# FIRE_MODEL_PATH = "models/fire_model_v3.pkl"
# SEVERITY_MODEL_PATH = "models/severity_model_v3.pkl"
# EXPORT_PATH = "data/exports/grid_predictions_Dplus2_realtime.csv"

# # === Load Grid Locations ===
# grids = pd.read_csv(GRID_CSV)
# print(f"Loaded {len(grids)} grid locations.")

# # === Round lat/lon to group nearby grids ===
# grids["lat_group"] = grids["lat"].round(1)
# grids["lon_group"] = grids["lon"].round(1)
# group_keys = grids[["lat_group", "lon_group"]].drop_duplicates()

# # === Fetch Weather for Each Group ===
# def fetch_weather(lat, lon, retries=3, delay=2):
#     url = (
#         f"https://api.openweathermap.org/data/2.5/forecast?"
#         f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
#     )
#     for attempt in range(retries):
#         try:
#             r = requests.get(url, timeout=10)
#             data = r.json()
#             if "list" not in data or len(data["list"]) < 17:
#                 raise ValueError("Incomplete forecast data")
#             forecast = data["list"][16]
#             return {
#                 "temperature": forecast["main"]["temp"],
#                 "rainfall": forecast.get("rain", {}).get("3h", 0),
#                 "wind": forecast["wind"]["speed"],
#                 "humidity": forecast["main"]["humidity"]
#             }
#         except Exception as e:
#             print(f"⚠️  Attempt {attempt+1} failed for group ({lat},{lon}): {e}")
#             time.sleep(delay)
#     print(f"❌ Skipping group ({lat},{lon}) after {retries} retries.")
#     return None

# # === Build Weather Lookup Table ===
# weather_lookup = {}
# for _, row in group_keys.iterrows():
#     lat, lon = row["lat_group"], row["lon_group"]
#     weather = fetch_weather(lat, lon)
#     if weather:
#         weather_lookup[(lat, lon)] = weather

# # === Assign Weather to Each Grid ===
# features = []
# for _, row in grids.iterrows():
#     key = (row["lat_group"], row["lon_group"])
#     weather = weather_lookup.get(key)
#     if not weather:
#         continue
#     features.append({
#         "grid_id": row["grid_id"],
#         "ndvi": row.get("ndvi", 0.4),
#         "temperature": weather["temperature"],
#         "rainfall": weather["rainfall"],
#         "wind": weather["wind"],
#         "dryness_index": 100 - weather["humidity"],
#         "firs_hotspots": row.get("firs_hotspots", 0),
#         "dry_lag1": row.get("dry_lag1", 0)
#     })

# df = pd.DataFrame(features)
# print(f"Prepared features for {len(df)} grids.")

# # === Load Models ===
# fire_model = joblib.load(FIRE_MODEL_PATH)
# severity_model = joblib.load(SEVERITY_MODEL_PATH)

# # === Predict Fire Occurrence ===
# X = df[["ndvi", "temperature", "rainfall", "wind", "dryness_index", "firs_hotspots", "dry_lag1"]]
# fire_proba = fire_model.predict_proba(X)[:, 1]
# df["fire_prob"] = fire_proba
# df["fire_pred"] = (fire_proba >= 0.3).astype(int)

# # === Predict Severity for Fire-Positive Grids ===
# fire_positive = df[df["fire_pred"] == 1]
# if not fire_positive.empty:
#     X_sev = fire_positive[["ndvi", "temperature", "rainfall", "wind", "dryness_index", "firs_hotspots", "dry_lag1"]]
#     sev_pred = severity_model.predict(X_sev)

#     if len(sev_pred.shape) > 1:
#         if sev_pred.shape[1] == 1:
#             sev_pred = sev_pred.ravel()
#         elif sev_pred.shape[1] == 2:
#             sev_pred = sev_pred[:, 1]
#         else:
#             raise ValueError(f"Unexpected severity prediction shape: {sev_pred.shape}")

#     fire_positive["severity_pred"] = sev_pred
#     df = df.merge(fire_positive[["grid_id", "severity_pred"]], on="grid_id", how="left")
# else:
#     df["severity_pred"] = None

# # === Save Predictions ===
# df["forecast_date"] = (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d")
# os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
# # Add lat/lon back from original grid data
# df = df.merge(grids[["grid_id", "lat", "lon"]], on="grid_id", how="left")

# df.to_csv(EXPORT_PATH, index=False)
# print(f"\n✅ Predictions saved to {EXPORT_PATH}")

import pandas as pd
import requests
import joblib
import os
import time
from datetime import datetime, timedelta

# === CONFIG ===
API_KEY = "3a4ade61299245585231113be187c820"
GRID_CSV = "data/grid_features_realtime.csv"
FIRE_MODEL_PATH = "models/fire_model_v3.pkl"
SEVERITY_MODEL_PATH = "models/severity_model_v3.pkl"
EXPORT_PATH = "data/exports/grid_predictions_Dplus2_realtime.csv"

# === Load Grid Features ===
grids = pd.read_csv(GRID_CSV)
print(f"Loaded {len(grids)} grid locations.")

# === Safety check: NDVI normalization ===
if "ndvi" in grids.columns:
    ndvi_max = pd.to_numeric(grids["ndvi"], errors="coerce").dropna().max()
    if pd.notna(ndvi_max) and ndvi_max > 1.5:
        print("⚠️ NDVI appears unnormalized. Scaling NDVI by 1/10000.")
        grids["ndvi"] = pd.to_numeric(grids["ndvi"], errors="coerce") / 10000.0

# === Round lat/lon to group nearby grids ===
grids["lat_group"] = grids["lat"].round(1)
grids["lon_group"] = grids["lon"].round(1)
group_keys = grids[["lat_group", "lon_group"]].drop_duplicates()

# === Fetch Weather for Each Group ===
def fetch_weather(lat, lon, retries=3, delay=2):
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?"
        f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    )
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if "list" not in data or len(data["list"]) < 17:
                raise ValueError("Incomplete forecast data")
            forecast = data["list"][16]  # ~48h ahead (D+2)
            return {
                "temperature": forecast["main"]["temp"],
                "rainfall": forecast.get("rain", {}).get("3h", 0) or 0,
                "wind": forecast["wind"]["speed"],
                "humidity": forecast["main"]["humidity"]
            }
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for group ({lat},{lon}): {e}")
            time.sleep(delay)
    print(f"❌ Skipping group ({lat},{lon}) after {retries} retries.")
    return None

# === Build Weather Lookup Table ===
weather_lookup = {}
for _, row in group_keys.iterrows():
    lat, lon = row["lat_group"], row["lon_group"]
    weather = fetch_weather(lat, lon)
    if weather:
        weather_lookup[(lat, lon)] = weather

# === Assign Features to Each Grid ===
features = []
for _, row in grids.iterrows():
    key = (row["lat_group"], row["lon_group"])
    weather = weather_lookup.get(key)
    if not weather:
        continue
    features.append({
        "grid_id": row["grid_id"],
        "ndvi": row.get("ndvi", 0.4),
        "temperature": weather["temperature"],
        "rainfall": weather["rainfall"],
        "wind": weather["wind"],
        "dryness_index": 100 - weather["humidity"],
        "firs_hotspots": row.get("firs_hotspots", 0),
        "dry_lag1": row.get("dry_lag1", 0),
        # keep terrain in dataframe but not used in model
        "elevation": row.get("elevation", None),
        "slope": row.get("slope", None),
        "aspect": row.get("aspect", None)
    })

df = pd.DataFrame(features)
print(f"Prepared features for {len(df)} grids.")

# === Ensure numeric types ===
num_cols = ["ndvi", "temperature", "rainfall", "wind", "dryness_index", "firs_hotspots", "dry_lag1",
            "elevation", "slope", "aspect"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# === Load Models ===
fire_model = joblib.load(FIRE_MODEL_PATH)
severity_model = joblib.load(SEVERITY_MODEL_PATH)

# === Predict Fire Occurrence (only trained features) ===
trained_feats = ["ndvi", "temperature", "rainfall", "wind", "dryness_index", "firs_hotspots", "dry_lag1"]
X = df[trained_feats]
fire_proba = fire_model.predict_proba(X)[:, 1]
df["fire_prob"] = fire_proba
df["fire_pred"] = (fire_proba >= 0.3).astype(int)

# === Predict Severity for Fire-Positive Grids ===
fire_positive = df[df["fire_pred"] == 1].copy()
if not fire_positive.empty:
    X_sev = fire_positive[trained_feats]
    sev_pred = severity_model.predict(X_sev)

    if len(sev_pred.shape) > 1:
        if sev_pred.shape[1] == 1:
            sev_pred = sev_pred.ravel()
        elif sev_pred.shape[1] == 2:
            sev_pred = sev_pred[:, 1]
        else:
            raise ValueError(f"Unexpected severity prediction shape: {sev_pred.shape}")

    fire_positive["severity_pred"] = sev_pred
    df = df.merge(fire_positive[["grid_id", "severity_pred"]], on="grid_id", how="left")
else:
    df["severity_pred"] = None

# === Save Predictions ===
df["forecast_date"] = (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d")
os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)

# Add lat/lon back from original grid data
df = df.merge(grids[["grid_id", "lat", "lon"]], on="grid_id", how="left")

df.to_csv(EXPORT_PATH, index=False)
print(f"\n✅ Predictions saved to {EXPORT_PATH}")
