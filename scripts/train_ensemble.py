import pandas as pd, numpy as np, tensorflow as tf, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import shap

# ---------------------- Load Surge Matrix ----------------------
df = pd.read_csv("data/preprocessed/training_matrix_xgb_preprocessed.csv")
df["date"] = pd.to_datetime(df["date"])
df["district"] = df["district"].str.title()
df = df.sort_values(["district", "date"]).reset_index(drop=True)

# ---------------------- Load LSTM Forecast ----------------------
X_lstm = np.load("data/X_lstm.npy")
lstm_model = tf.keras.models.load_model("models/lstm_hotspot_forecaster.keras")
thermal_forecast = lstm_model.predict(X_lstm).flatten()

# ---------------------- Load Forecast Keys ----------------------
keys_df = pd.read_csv("data/X_lstm_keys.csv")
keys_df["date"] = pd.to_datetime(keys_df["date"])
keys_df["district"] = keys_df["district"].str.title()
keys_df["key"] = keys_df["district"] + "_" + keys_df["date"].astype(str)
keys_df["thermal_forecast_raw"] = thermal_forecast

# ---------------------- Merge Forecast into Surge Matrix ----------------------
df["key"] = df["district"] + "_" + df["date"].astype(str)
df = df.merge(keys_df[["key", "thermal_forecast_raw"]], on="key", how="left")

# ---------------------- Smooth Forecast ----------------------
df["thermal_forecast_smooth"] = df.groupby("district")["thermal_forecast_raw"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# ---------------------- Define Target and Features ----------------------
target = "surge_label"
drop_cols = [
    "date", "district", "surge_label", "thermal_forecast_raw", "key",
    "thermal_ema3", "thermal_roll3"  # prevent leakage
]
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y = df[target]

# ---------------------- Train/Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------- Scale Features ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- Handle Class Imbalance ----------------------
neg, pos = np.bincount(y_train)
ratio = neg / pos
print(f" Class imbalance ratio: {ratio:.2f}")

# ---------------------- Train XGBoost Classifier ----------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---------------------- Predict and Tune Threshold ----------------------
y_proba = model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"\n Optimal threshold (F1-max): {best_thresh:.2f}")

# ---------------------- Final Predictions ----------------------
y_pred = (y_proba > best_thresh).astype(int)

# ---------------------- Evaluation ----------------------
print("\n Final Evaluation (XGBoost with LSTM feature):")
print(f" Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f" Precision: {precision_score(y_test, y_pred):.2f}")
print(f" Recall: {recall_score(y_test, y_pred):.2f}")
print(f" F1 Score: {f1_score(y_test, y_pred):.2f}")

# ---------------------- Visualize Predictions ----------------------
plt.figure(figsize=(10, 4))
plt.plot(y_test.values, label="Actual", color="firebrick", linewidth=2)
plt.plot(y_pred, label="Predicted", color="dodgerblue", linestyle="--")
plt.title("Surge Classification: Actual vs Predicted")
plt.xlabel("Time Index")
plt.ylabel("Surge Flag")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------- Feature Importance ----------------------
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)
importance_df.to_csv("data/feature_importance_ensemble.csv", index=False)

print("\n Top Features Driving Predictions:")
print(importance_df.head(10))

plt.figure(figsize=(8, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="forestgreen")
plt.gca().invert_yaxis()
plt.title("Feature Importance: Surge Classifier with LSTM Feature")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ---------------------- SHAP Explanations ----------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# ---------------------- Export Predictions ----------------------
df_test = X_test.copy()
df_test["actual"] = y_test.values
df_test["probability"] = y_proba
df_test["predicted"] = y_pred
df_test["district"] = df.loc[X_test.index, "district"].values
df_test["date"] = df.loc[X_test.index, "date"].values
df_test.to_csv("data/predictions_ensemble.csv", index=False)
print("Predictions exported to data/predictions_ensemble.csv")

# ---------------------- Generate Alerts ----------------------
alerts = df_test[df_test["probability"] > 0.8].copy()
alerts["alert_type"] = "Surge Alert"
alerts["message"] = "Predicted surge risk exceeds threshold"
alerts.to_csv("data/alerts.csv", index=False)


# ---------------------- Save Model ----------------------
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_ensemble_classifier.json")
print("Model saved to models/xgb_ensemble_classifier.json")
