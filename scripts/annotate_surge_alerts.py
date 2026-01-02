import numpy as np
import pandas as pd
import joblib
import shap

# 📂 Load fused feature matrix
X = np.load("data/X_lstm.npy").reshape(-1, 182)  # 14 time steps × 13 features

# 📦 Load surge classifier
model = joblib.load("models/surge_classifier.pkl")
y_pred = model.predict(X)

# 🧠 SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 🔍 Decode feature names
feature_names = [
    "ndvi", "thermal_count", "temperature", "rainfall", "wind",
    "ndvi_drop", "wind_surge", "rainfall_deficit", "thermal_lag", "thermal_scaled",
    "rainfall_3d", "wind_3d", "ndvi_3d"
]
decoded = [f"{feature_names[i % 13]}[t-{14 - i // 13}]" for i in range(182)]

# 📂 Load district and date info
district_list = np.load("data/district_list.npy", allow_pickle=True)
date_list = np.load("data/date_list.npy", allow_pickle=True)

# 🔍 Annotate surge alerts
alert_data = []
for i in range(len(X)):
    if y_pred[i] == 1:
        top_indices = np.argsort(np.abs(shap_values[i].values))[-3:][::-1]
        top_features = [decoded[j] for j in top_indices]
        alert_data.append({
            "sample": i,
            "district": district_list[i],
            "date": date_list[i],
            "drivers": ", ".join(top_features)
        })

# 💾 Save annotated alerts
df_alerts = pd.DataFrame(alert_data)
df_alerts.to_csv("data/annotated_surge_alerts.csv", index=False)
print("✅ Annotated alerts saved to data/annotated_surge_alerts.csv")
