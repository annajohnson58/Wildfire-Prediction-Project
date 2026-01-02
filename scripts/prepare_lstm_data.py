# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import os

# # 📂 Paths
# era5_path = "data/daily_climate_district.csv"
# ndvi_path = "data/ndvi/Kerala_District_Daily_NDVI_MODIS.csv"
# thermal_path = "data/daily_predictions.csv"
# os.makedirs("data", exist_ok=True)

# # 🛰️ Load ERA5 climate data from CSV
# print("🔄 Loading ERA5 CSV...")
# try:
#     climate = pd.read_csv(era5_path)
#     print("📄 ERA5 columns:", climate.columns)
#     climate["date"] = pd.to_datetime(climate["date"])
#     climate = climate[(climate["date"] >= "2022-01-01") & (climate["date"] <= "2022-12-31")]
#     print("✅ Climate rows:", len(climate))
# except Exception as e:
#     print(f"❌ Failed to load ERA5 CSV: {e}")
#     exit()

# # 🔧 Rename and compute wind
# climate = climate.rename(columns={
#     "t2m": "temperature",
#     "tp": "rainfall",
#     "u10": "wind_u",
#     "v10": "wind_v"
# })
# climate["wind"] = (climate["wind_u"]**2 + climate["wind_v"]**2)**0.5

# # 🌿 Load NDVI
# print("🔄 Loading NDVI CSV...")
# try:
#     ndvi = pd.read_csv(ndvi_path)
#     print("📄 NDVI columns:", ndvi.columns)
#     if "date" not in ndvi.columns:
#         ndvi.columns = ndvi.columns.str.strip().str.lower()
#     ndvi["date"] = pd.to_datetime(ndvi["date"])
#     print("✅ NDVI rows:", len(ndvi))
# except Exception as e:
#     print(f"❌ Failed to load NDVI file: {e}")
#     exit()

# # 🔥 Load thermal surge data
# print("🔄 Loading thermal surge CSV...")
# try:
#     thermal = pd.read_csv(thermal_path)
#     print("📄 Thermal columns:", thermal.columns)
#     if "date" not in thermal.columns:
#         thermal.columns = thermal.columns.str.strip().str.lower()
#     thermal["date"] = pd.to_datetime(thermal["date"])
#     print("✅ Thermal rows:", len(thermal))
# except Exception as e:
#     print(f"❌ Failed to load thermal file: {e}")
#     exit()

# # 🔗 Merge all sources
# df = climate.merge(ndvi[["date", "ndvi"]], on="date", how="inner")
# df = df.merge(thermal[["date", "risk_level"]], on="date", how="inner")
# df = df.rename(columns={"risk_level": "thermal_count"})
# print("✅ Merged rows:", len(df))
# if df.empty:
#     print("⚠️ Merged DataFrame is empty. Check date alignment across sources.")
#     exit()
# print(df.head())

# # 📊 Select and scale features
# features = ["ndvi", "thermal_count", "temperature", "rainfall", "wind"]
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(df[features])

# # 🧠 Create LSTM sequences
# def create_sequences(data, seq_len=7):
#     X, y = [], []
#     for i in range(len(data) - seq_len):
#         X.append(data[i:i+seq_len])
#         y.append(data[i+seq_len][1])  # thermal_count as target
#     return np.array(X), np.array(y)

# X_seq, y_seq = create_sequences(scaled)

# # 💾 Save sequences
# np.save("data/X_lstm.npy", X_seq)
# np.save("data/y_lstm.npy", y_seq)
# print("✅ LSTM-ready sequences saved.")
# print("📦 X shape:", X_seq.shape, "| y shape:", y_seq.shape)
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import os

# # 📂 Paths
# era5_path = "data/daily_climate_district.csv"
# ndvi_path = "data/ndvi/Kerala_District_Daily_NDVI_MODIS.csv"
# thermal_path = "data/daily_predictions.csv"
# os.makedirs("data", exist_ok=True)

# # 🛰️ Load ERA5 climate data
# print("🔄 Loading ERA5 CSV...")
# try:
#     climate = pd.read_csv(era5_path)
#     climate["date"] = pd.to_datetime(climate["date"])
#     climate = climate[(climate["date"] >= "2022-01-01") & (climate["date"] <= "2022-12-31")]
#     print("✅ Climate rows:", len(climate))
# except Exception as e:
#     print(f"❌ Failed to load ERA5 CSV: {e}")
#     exit()

# # 🔧 Compute wind speed
# climate["wind"] = (climate["wind_u"]**2 + climate["wind_v"]**2)**0.5

# # 🌿 Load NDVI
# print("🔄 Loading NDVI CSV...")
# try:
#     ndvi = pd.read_csv(ndvi_path)
#     ndvi.columns = ndvi.columns.str.strip().str.lower()
#     ndvi = ndvi.rename(columns={"mean": "ndvi"})
#     ndvi["date"] = pd.to_datetime(ndvi["date"])
#     ndvi = ndvi[(ndvi["date"] >= "2022-01-01") & (ndvi["date"] <= "2022-12-31")]
#     print("✅ NDVI rows:", len(ndvi))
# except Exception as e:
#     print(f"❌ Failed to load NDVI file: {e}")
#     exit()

# # 🔥 Load thermal surge data
# print("🔄 Loading thermal surge CSV...")
# try:
#     thermal = pd.read_csv(thermal_path)
#     thermal.columns = thermal.columns.str.strip().str.lower()
#     thermal = thermal.rename(columns={"risk_level": "thermal_count"})
#     thermal["date"] = pd.to_datetime(thermal["date"])
#     thermal = thermal[(thermal["date"] >= "2022-01-01") & (thermal["date"] <= "2022-12-31")]
#     print("✅ Thermal rows:", len(thermal))
# except Exception as e:
#     print(f"❌ Failed to load thermal file: {e}")
#     exit()

# # 🔗 Merge all sources on date and district
# print("🔗 Merging datasets...")
# try:
#     df = climate.merge(ndvi[["date", "district", "ndvi"]], on=["date", "district"], how="inner")
#     df = df.merge(thermal[["date", "district", "thermal_count"]], on=["date", "district"], how="inner")
#     print("✅ Merged rows:", len(df))
# except Exception as e:
#     print(f"❌ Merge failed: {e}")
#     exit()

# if df.empty:
#     print("⚠️ Merged DataFrame is empty. Check date and district alignment.")
#     exit()

# print(df.head())

# # 📊 Select and scale features
# features = ["ndvi", "thermal_count", "temperature", "rainfall", "wind"]
# scaler = MinMaxScaler()
# scaled = scaler.fit_transform(df[features])

# # 🧠 Create LSTM sequences (per district)
# def create_sequences(data, seq_len=7):
#     X, y = [], []
#     for i in range(len(data) - seq_len):
#         X.append(data[i:i+seq_len])
#         y.append(data[i+seq_len][1])  # thermal_count as target
#     return np.array(X), np.array(y)

# # 🔁 Group by district and generate sequences
# X_all, y_all = [], []
# for district in df["district"].unique():
#     df_d = df[df["district"] == district].sort_values("date")
#     scaled_d = scaler.transform(df_d[features])
#     X_d, y_d = create_sequences(scaled_d)
#     X_all.append(X_d)
#     y_all.append(y_d)

# # 📦 Stack all districts
# X_seq = np.vstack(X_all)
# y_seq = np.hstack(y_all)

# # 💾 Save sequences
# np.save("data/X_lstm.npy", X_seq)
# np.save("data/y_lstm.npy", y_seq)
# print("✅ LSTM-ready sequences saved.")
# print("📦 X shape:", X_seq.shape, "| y shape:", y_seq.shape)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# 📂 Paths
era5_path = "data/daily_climate_district.csv"
ndvi_path = "data/ndvi/Kerala_District_Daily_NDVI_MODIS.csv"
thermal_path = "data/daily_predictions_full.csv"
os.makedirs("data", exist_ok=True)

# 🛰️ Load ERA5 climate data
print("🔄 Loading ERA5 CSV...")
climate = pd.read_csv(era5_path)
climate["date"] = pd.to_datetime(climate["date"])
climate["wind"] = (climate["wind_u"]**2 + climate["wind_v"]**2)**0.5
print("✅ Climate rows:", len(climate))

# 🌿 Load NDVI
print("🔄 Loading NDVI CSV...")
ndvi = pd.read_csv(ndvi_path)
ndvi.columns = ndvi.columns.str.strip().str.lower()
ndvi = ndvi.rename(columns={"mean": "ndvi"})
ndvi["date"] = pd.to_datetime(ndvi["date"])
print("✅ NDVI rows:", len(ndvi))

# 🔥 Load thermal surge data
print("🔄 Loading thermal surge CSV...")
thermal = pd.read_csv(thermal_path)
thermal.columns = thermal.columns.str.strip().str.lower()
thermal["date"] = pd.to_datetime(thermal["date"])
print("✅ Thermal rows:", len(thermal))

# 🔗 Merge all sources
print("🔗 Merging datasets...")
df = climate.merge(ndvi[["date", "district", "ndvi"]], on=["date", "district"], how="inner")
df = df.merge(thermal[["date", "district", "thermal_count"]], on=["date", "district"], how="inner")
print("✅ Merged rows:", len(df))

# 🔁 Sort by district and date
df = df.sort_values(["district", "date"])

# 🧠 Engineer surge-sensitive features
df["ndvi_drop"] = df.groupby("district")["ndvi"].diff().fillna(0)
df["wind_surge"] = df.groupby("district")["wind"].diff().fillna(0)
df["rainfall_7d"] = df.groupby("district")["rainfall"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["rainfall_deficit"] = df["rainfall_7d"] - df["rainfall"]
df["thermal_lag"] = df.groupby("district")["thermal_count"].shift(1).fillna(0)

# 🔁 Add rolling 3-day features
df["rainfall_3d"] = df.groupby("district")["rainfall"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["wind_3d"] = df.groupby("district")["wind"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["ndvi_3d"] = df.groupby("district")["ndvi"].transform(lambda x: x.rolling(3, min_periods=1).mean())

# 🔁 Normalize thermal count per district
df["thermal_scaled"] = df.groupby("district")["thermal_count"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

# 🔥 Create binary surge label
df["surge_label"] = (df["thermal_count"] >= df.groupby("district")["thermal_count"].transform("mean") + 2).astype(int)

# 📊 Select features
features = [
    "ndvi", "thermal_count", "temperature", "rainfall", "wind",
    "ndvi_drop", "wind_surge", "rainfall_deficit", "thermal_lag", "thermal_scaled",
    "rainfall_3d", "wind_3d", "ndvi_3d"
]
scaler = MinMaxScaler()

# 🧠 Create LSTM sequences
def create_sequences(data, surge_labels, seq_len=14):
    X, y_reg, y_cls = [], [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y_reg.append(data[i+seq_len][features.index("thermal_scaled")])
        y_cls.append(surge_labels[i+seq_len])
    return np.array(X), np.array(y_reg), np.array(y_cls)

# 🔁 Group by district and generate sequences
X_all, y_reg_all, y_cls_all = [], [], []
for district in df["district"].unique():
    df_d = df[df["district"] == district].sort_values("date")
    if len(df_d) < 15:
        print(f"⚠️ Skipping {district}: only {len(df_d)} rows")
        continue
    scaled_d = scaler.fit_transform(df_d[features])
    surge_labels = df_d["surge_label"].values
    X_d, y_reg_d, y_cls_d = create_sequences(scaled_d, surge_labels)
    if X_d.ndim == 3:
        X_all.append(X_d)
        y_reg_all.append(y_reg_d)
        y_cls_all.append(y_cls_d)
    else:
        print(f"⚠️ Skipping {district}: invalid sequence shape")

# 📦 Stack sequences
X_seq = np.vstack(X_all)
y_reg_seq = np.hstack(y_reg_all)
y_cls_seq = np.hstack(y_cls_all)
# 🧭 Reconstruct alignment keys for each sequence
keys = []
for district in df["district"].unique():
    df_d = df[df["district"] == district].sort_values("date")
    if len(df_d) < 15:
        continue
    dates = df_d["date"].values
    for i in range(len(dates) - 14):  # 14 = sequence length
        keys.append({
            "district": district,
            "date": pd.to_datetime(dates[i + 14])  # target date for each sequence
        })

# 💾 Save alignment keys
keys_df = pd.DataFrame(keys)
keys_df.to_csv("data/X_lstm_keys.csv", index=False)
print("✅ Alignment keys saved to data/X_lstm_keys.csv")
print("📦 Keys shape:", keys_df.shape)

# 💾 Save
np.save("data/X_lstm.npy", X_seq)
np.save("data/y_lstm.npy", y_reg_seq)
np.save("data/y_surge.npy", y_cls_seq)
print("✅ LSTM-ready sequences saved.")
print("📦 X shape:", X_seq.shape, "| y_reg shape:", y_reg_seq.shape, "| y_cls shape:", y_cls_seq.shape)
