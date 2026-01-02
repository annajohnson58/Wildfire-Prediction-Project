# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import joblib

# # ✅ Optional: SMOTE oversampling
# USE_SMOTE = False
# if USE_SMOTE:
#     from imblearn.over_sampling import SMOTE

# # 📂 Load data
# X = np.load("data/X_lstm.npy")
# y = np.load("data/y_surge.npy")

# # 🔁 Flatten sequences
# X_flat = X.reshape(X.shape[0], -1)

# # ✅ Apply SMOTE if enabled
# if USE_SMOTE:
#     print("🔁 Applying SMOTE oversampling...")
#     X_flat, y = SMOTE().fit_resample(X_flat, y)
#     print("✅ Resampled shape:", X_flat.shape)

# # 🔁 Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# # 🚀 Train XGBoost with surge weighting
# model = xgb.XGBClassifier(
#     n_estimators=100,
#     max_depth=5,
#     learning_rate=0.1,
#     scale_pos_weight=40,  # ~756 calm / 18 surge
#     use_label_encoder=False,
#     eval_metric="logloss"
# )
# model.fit(X_train, y_train)

# # 📊 Evaluate
# y_pred = model.predict(X_test)
# print("\n📈 Surge Classifier Performance:\n")
# print(classification_report(y_test, y_pred))

# # 📊 Feature importance
# print("\n🔍 Top Features Driving Surge Prediction:")
# xgb.plot_importance(model, max_num_features=10)
# plt.tight_layout()
# plt.show()

# # 💾 Save model
# joblib.dump(model, "models/surge_classifier.pkl")
# print("✅ Surge classifier saved to models/surge_classifier.pkl")
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap
import joblib
import os

# Load data
df = pd.read_csv("data/features/final_feature_matrix.csv")
import pandas as pd

df = pd.read_csv("data/features/final_feature_matrix.csv", low_memory=False)
print("🧾 Columns in your CSV:")
print(df.columns.tolist())

# Features and target
features = ["ndvi", "rainfall", "rh", "wind_speed"]
X = df[features]
y = df["thermal_flag"]

# Balance classes with SMOTE
print("⚖️ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
print(f"✅ Balanced samples: {len(X_bal)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

# Train XGBoost
print("🚀 Training XGBoost...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("\n📊 Evaluation:")
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP importance
print("\n🔍 SHAP Feature Importance:")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)
shap.summary_plot(shap_values, X_train, show=False)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/surge_classifier.pkl")
print("✅ Model saved: models/surge_classifier.pkl")
