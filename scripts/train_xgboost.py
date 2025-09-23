# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt

# # ---------------------- Load Full Dataset ----------------------
# df = pd.read_csv("data/training_matrix.csv")
# df["date"] = pd.to_datetime(df["date"])

# # ---------------------- Add Lag Features ----------------------
# df["ndvi_lag1"] = df["ndvi"].shift(1)
# df["ndvi_lag2"] = df["ndvi"].shift(2)
# df["thermal_lag1"] = df["thermal_count"].shift(1)
# df["thermal_lag3"] = df["thermal_count"].shift(3)
# df["rh_lag1"] = df["rh"].shift(1)
# df["rh_lag5"] = df["rh"].shift(5)

# # ---------------------- Add Rolling and EMA Features ----------------------
# df["ndvi_roll3"] = df["ndvi"].rolling(3).mean()
# df["ndvi_roll5"] = df["ndvi"].rolling(5).mean()
# df["thermal_roll3"] = df["thermal_count"].rolling(3).mean()
# df["thermal_ema3"] = df["thermal_count"].ewm(span=3).mean()

# # ---------------------- Add Compound Indicators ----------------------
# df["dry_heat_index"] = df["rh_anomaly"] * df["temperature"]
# df["veg_stress_index"] = df["ndvi_drop"] * df["wind"]

# df.dropna(inplace=True)

# # ---------------------- Create Binary Target ----------------------
# df["fire_flag"] = (df["thermal_spike"] > 0).astype(int)
# target = "fire_flag"

# # ---------------------- Define Features ----------------------
# features = [
#     "ndvi", "ndvi_drop", "temperature", "rh", "rh_anomaly", "wind", "rainfall",
#     "ndvi_lag1", "ndvi_lag2", "thermal_lag1", "thermal_lag3", "rh_lag1", "rh_lag5",
#     "ndvi_roll3", "ndvi_roll5", "thermal_roll3", "thermal_ema3",
#     "dry_heat_index", "veg_stress_index"
# ]

# X = df[features]
# y = df[target]

# # ---------------------- Train/Test Split ----------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # ---------------------- Scale Features ----------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # ---------------------- Compute Class Imbalance ----------------------
# neg, pos = np.bincount(y_train)
# ratio = neg / pos
# print(f" Class imbalance ratio: {ratio:.2f}")

# # ---------------------- Train XGBoost Classifier ----------------------
# model = XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.03,
#     max_depth=7,
#     subsample=0.9,
#     colsample_bytree=0.9,
#     scale_pos_weight=ratio,
#     random_state=42
# )

# model.fit(X_train_scaled, y_train)

# # ---------------------- Predict with Custom Threshold ----------------------
# y_proba = model.predict_proba(X_test_scaled)[:, 1]
# y_pred = (y_proba > 0.3).astype(int)

# # ---------------------- Evaluate ----------------------
# acc = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred)
# rec = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f" XGBoost classifier trained successfully.")
# print(f" Accuracy: {acc:.2f}")
# print(f" Precision: {prec:.2f}")
# print(f" Recall: {rec:.2f}")
# print(f" F1 Score: {f1:.2f}")

# # ---------------------- Visualize Predictions ----------------------
# plt.figure(figsize=(10, 4))
# plt.plot(y_test.values, label="Actual", color="firebrick", linewidth=2)
# plt.plot(y_pred, label="Predicted", color="dodgerblue", linestyle="--")
# plt.title("Fire Surge Classification: Actual vs Predicted (Full Dataset)")
# plt.xlabel("Time Index")
# plt.ylabel("Fire Surge Flag")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # ---------------------- Feature Importance ----------------------
# importances = model.feature_importances_
# feature_names = X.columns

# importance_df = pd.DataFrame({
#     "Feature": feature_names,
#     "Importance": importances
# }).sort_values(by="Importance", ascending=False)
# importance_df.to_csv("data/feature_importance.csv", index=False)

# print("\n Top Features Driving Fire Surge Predictions:")
# print(importance_df.head(10))

# # ---------------------- Visualize Feature Importance ----------------------
# plt.figure(figsize=(8, 5))
# plt.barh(importance_df["Feature"], importance_df["Importance"], color="forestgreen")
# plt.gca().invert_yaxis()
# plt.title("🔥 Feature Importance: Fire Surge Classifier")
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.show()

# # ---------------------- Export Predictions ----------------------
# df_test = X_test.copy()
# df_test["actual"] = y_test.values
# df_test["predicted"] = y_pred
# df_test["probability"] = y_proba

# df_test["district"] = df.loc[X_test.index, "district"].values


# # Optional: Add date column if available
# df_test["date"] = df.loc[X_test.index, "date"].values

# # Save to CSV
# df_test.to_csv("data/predictions.csv", index=False)
# print("✅ Predictions exported to data/predictions.csv")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ---------------------- Load Full Dataset ----------------------
df = pd.read_csv("data/training_matrix.csv")
df["date"] = pd.to_datetime(df["date"])

# ---------------------- Add Lag Features ----------------------
df["ndvi_lag1"] = df["ndvi"].shift(1)
df["ndvi_lag2"] = df["ndvi"].shift(2)
df["thermal_lag1"] = df["thermal_count"].shift(1)
df["thermal_lag3"] = df["thermal_count"].shift(3)
df["rh_lag1"] = df["rh"].shift(1)
df["rh_lag5"] = df["rh"].shift(5)

# ---------------------- Add Rolling and EMA Features ----------------------
df["ndvi_roll3"] = df["ndvi"].rolling(3).mean()
df["ndvi_roll5"] = df["ndvi"].rolling(5).mean()
df["thermal_roll3"] = df["thermal_count"].rolling(3).mean()
df["thermal_ema3"] = df["thermal_count"].ewm(span=3).mean()

# ---------------------- Add Compound Indicators ----------------------
df["dry_heat_index"] = df["rh_anomaly"] * df["temperature"]
df["veg_stress_index"] = df["ndvi_drop"] * df["wind"]

df.dropna(inplace=True)

# ---------------------- Create Binary Target ----------------------
df["fire_flag"] = (df["thermal_spike"] > 0).astype(int)
target = "fire_flag"

# ---------------------- Define Features ----------------------
features = [
    "ndvi", "ndvi_drop", "temperature", "rh", "rh_anomaly", "wind", "rainfall",
    "ndvi_lag1", "ndvi_lag2", "thermal_lag1", "thermal_lag3", "rh_lag1", "rh_lag5",
    "ndvi_roll3", "ndvi_roll5", "thermal_roll3", "thermal_ema3",
    "dry_heat_index", "veg_stress_index"
]

X = df[features]
y = df[target]

# ---------------------- Scale Features ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- Compute Class Imbalance ----------------------
neg, pos = np.bincount(y)
ratio = neg / pos
print(f" Class imbalance ratio: {ratio:.2f}")

# ---------------------- Train XGBoost Classifier ----------------------
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=ratio,
    random_state=42
)

model.fit(X_scaled, y)

# ---------------------- Predict with Custom Threshold ----------------------
y_proba = model.predict_proba(X_scaled)[:, 1]
y_pred = (y_proba > 0.3).astype(int)

# ---------------------- Evaluate ----------------------
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f" XGBoost classifier trained successfully.")
print(f" Accuracy: {acc:.2f}")
print(f" Precision: {prec:.2f}")
print(f" Recall: {rec:.2f}")
print(f" F1 Score: {f1:.2f}")

# ---------------------- Visualize Predictions ----------------------
plt.figure(figsize=(10, 4))
plt.plot(y.values, label="Actual", color="firebrick", linewidth=2)
plt.plot(y_pred, label="Predicted", color="dodgerblue", linestyle="--")
plt.title("Fire Surge Classification: Actual vs Predicted (Full Dataset)")
plt.xlabel("Time Index")
plt.ylabel("Fire Surge Flag")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------- Feature Importance ----------------------
importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
importance_df.to_csv("data/feature_importance.csv", index=False)

print("\n Top Features Driving Fire Surge Predictions:")
print(importance_df.head(10))

# ---------------------- Visualize Feature Importance ----------------------
plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="forestgreen")
plt.gca().invert_yaxis()
plt.title("🔥 Feature Importance: Fire Surge Classifier")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ---------------------- Export Predictions ----------------------
df["actual"] = y
df["predicted"] = y_pred
df["probability"] = y_proba

# Save to CSV with district and date
df.to_csv("data/predictions.csv", index=False)
print("✅ Predictions exported to data/predictions.csv")
