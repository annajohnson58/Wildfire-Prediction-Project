

# import pandas as pd, numpy as np, os, sqlite3, shap, matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, confusion_matrix
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split

# # ---------------------- Load Data ----------------------
# df = pd.read_csv("data/preprocessed/training_matrix_xgb_preprocessed.csv")
# df["date"] = pd.to_datetime(df["date"])
# df["district"] = df["district"].str.title()
# df = df.sort_values(["district", "date"])

# # ---------------------- Feature Engineering ----------------------
# for col in ["ndvi", "temperature", "rainfall", "wind"]:
#     if col in df.columns:
#         df[f"{col}_delta"] = df.groupby("district")[col].diff()
#         df[f"{col}_lag1"] = df.groupby("district")[col].shift(1)
# df = df.dropna().reset_index(drop=True)

# # ---------------------- Define Features ----------------------
# target = "surge_label"
# exclude = ["date", "district", "surge_label"]
# features = [col for col in df.columns if col not in exclude]

# X = df[features]
# y = df[target]

# # ---------------------- SHAP-Based Feature Pruning ----------------------
# shap_path = "data/feature_importance.csv"
# if os.path.exists(shap_path):
#     top_features = pd.read_csv(shap_path).sort_values("Importance", ascending=False).head(15)["Feature"].tolist()
#     X = X[top_features]
#     print(f" Using top {len(top_features)} SHAP features.")
# else:
#     print(" SHAP importance file not found. Using all features.")

# # ---------------------- Time-Aware Split ----------------------
# split_idx = int(len(df) * 0.8)
# X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
# y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# # ---------------------- SMOTE Resampling ----------------------
# print(" Applying SMOTE to balance surge class...")
# X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# # ---------------------- Train Model ----------------------
# model = XGBClassifier(
#     n_estimators=300,
#     learning_rate=0.03,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     eval_metric="logloss",
#     random_state=42
# )
# model.fit(X_train_resampled, y_train_resampled)

# # ---------------------- Predict & Tune Threshold ----------------------
# y_proba = model.predict_proba(X_test)[:, 1]
# precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
# best_thresh = thresholds[np.argmax(f1_scores)]
# print(f" Optimal threshold (F1-max): {best_thresh:.2f}")

# # ---------------------- Log Recall Across Thresholds ----------------------
# print("\n Recall at different thresholds:")
# for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
#     preds = (y_proba > t).astype(int)
#     r = recall_score(y_test, preds)
#     p = precision_score(y_test, preds)
#     print(f" Threshold {t:.2f} → Precision: {p:.2f}, Recall: {r:.2f}")

# # ---------------------- Ensemble Smoothing ----------------------
# df_test = X_test.copy()
# df_test["probability"] = y_proba
# df_test["smoothed_prob"] = df_test["probability"].rolling(window=3, min_periods=1).mean()
# threshold = 0.6
# df_test["predicted"] = (df_test["smoothed_prob"] > threshold).astype(int)

# # ---------------------- Evaluation ----------------------
# y_pred = df_test["predicted"]
# acc = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred)
# rec = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# cm = confusion_matrix(y_test, y_pred)

# print("\n Final Evaluation (Threshold = 0.6, Smoothed):")
# print(f" Accuracy: {acc:.2f}")
# print(f" Precision: {prec:.2f}")
# print(f" Recall: {rec:.2f}")
# print(f" F1 Score: {f1:.2f}")
# print("Confusion Matrix:\n", cm)

# # ---------------------- Severity Classification ----------------------
# def classify_severity(p):
#     return "High" if p > 0.8 else "Moderate" if p > 0.6 else "Low"
# df_test["severity"] = df_test["smoothed_prob"].apply(classify_severity)

# # ---------------------- SHAP ----------------------
# explainer = shap.Explainer(model)
# shap_values = explainer(X_test)
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# # ---------------------- Export Predictions ----------------------
# df_test["actual"] = y_test.values
# df_test["district"] = df.loc[X_test.index, "district"].values
# df_test["date"] = df.loc[X_test.index, "date"].values
# df_test.to_csv("data/predictions.csv", index=False)
# print(" Predictions exported to data/predictions.csv")

# # ---------------------- Save Model ----------------------
# os.makedirs("models", exist_ok=True)
# model.save_model("models/xgb_surge_classifier.json")
# print(" Model saved to models/xgb_surge_classifier.json")

# # ---------------------- Save to SQLite ----------------------
# def save_predictions_to_db(df, model_version="xgb_v5.0"):
#     conn = sqlite3.connect("wildfire.db")
#     cursor = conn.cursor()
#     for _, row in df.iterrows():
#         cursor.execute("""
#         INSERT INTO daily_features (
#             district, date, surge_prob, alert_triggered, model_version
#         ) VALUES (?, ?, ?, ?, ?)
#         """, (
#             row["district"],
#             row["date"].strftime("%Y-%m-%d"),
#             float(row["smoothed_prob"]),
#             bool(row["predicted"]),
#             model_version
#         ))
#     conn.commit()
#     conn.close()
#     print(" Predictions saved to wildfire.db")

# save_predictions_to_db(df_test)

# # ---------------------- Optional: Hybrid Ensemble with LSTM ----------------------
# # Stub for integration
# # lstm_probs = load_lstm_predictions()  # shape must match y_proba
# # df_test["ensemble_prob"] = 0.6 * df_test["smoothed_prob"] + 0.4 * lstm_probs
# # df_test["ensemble_pred"] = (df_test["ensemble_prob"] > 0.6).astype(int)


import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, confusion_matrix, classification_report
)

# ============================================================
# 1. Load Data
# ============================================================
df = pd.read_csv("data/preprocessed/training_matrix_xgb_preprocessed.csv")
df["date"] = pd.to_datetime(df["date"])
df["district"] = df["district"].str.title()
df = df.sort_values(["district", "date"])

# ============================================================
# 2. Feature Engineering (shared for both models)
# ============================================================
for col in ["ndvi", "temperature", "rainfall", "wind"]:
    if col in df.columns:
        df[f"{col}_delta"] = df.groupby("district")[col].diff()
        df[f"{col}_lag1"] = df.groupby("district")[col].shift(1)

df = df.dropna().reset_index(drop=True)

# ============================================================
# 3. ---- SURGE MODEL (BINARY) ----
# ============================================================

target = "surge_label"
exclude = ["date", "district", "surge_label"]
features = [col for col in df.columns if col not in exclude]

X = df[features]
y = df[target]

# ---------------------- SHAP-Based Feature Pruning ----------------------
shap_path = "data/feature_importance.csv"
if os.path.exists(shap_path):
    top_features = (
        pd.read_csv(shap_path)
        .sort_values("Importance", ascending=False)
        .head(15)["Feature"]
        .tolist()
    )
    X = X[top_features]
    print(f"Using top {len(top_features)} SHAP features.")
else:
    print("SHAP importance file not found. Using all features.")

# ---------------------- Time-Aware Split ----------------------
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ---------------------- SMOTE Resampling ----------------------
print("Applying SMOTE to balance surge class...")
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# ---------------------- Train Surge Model ----------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# ---------------------- Predict & Tune Threshold ----------------------
y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

best_thresh = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold (F1-max): {best_thresh:.2f}")

print("\nRecall at different thresholds:")
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    preds = (y_proba > t).astype(int)
    r = recall_score(y_test, preds)
    p = precision_score(y_test, preds)
    print(f"Threshold {t:.2f} → Precision: {p:.2f}, Recall: {r:.2f}")

# ---------------------- Ensemble Smoothing ----------------------
df_test = X_test.copy()
df_test["probability"] = y_proba
df_test["smoothed_prob"] = df_test["probability"].rolling(window=3, min_periods=1).mean()
threshold = 0.6
df_test["predicted"] = (df_test["smoothed_prob"] > threshold).astype(int)

# ---------------------- Evaluation ----------------------
y_pred = df_test["predicted"]
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nFinal Surge Evaluation (Threshold = 0.6, Smoothed):")
print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:\n", cm)

# ============================================================
# 4. ---- TRUE SEVERITY MODEL (MULTI-CLASS) ----
# ============================================================

# If severity_label doesn't exist, create it
if "severity_label" not in df.columns:
    print("\nCreating severity labels (rule-based)...")

    def compute_severity(row):
        score = 0
        if row["temperature"] > 32:
            score += 1
        if row["wind"] > 12:
            score += 1
        if row["rainfall"] < 2:
            score += 1
        if row["ndvi"] < 0.3:
            score += 1

        if score >= 3:
            return 2   # High
        if score == 2:
            return 1   # Moderate
        return 0       # Low

    df["severity_label"] = df.apply(compute_severity, axis=1)

# ---------------------- FIX: Ensure all severity classes exist ----------------------
unique_classes = df["severity_label"].unique()
print("Severity classes found:", unique_classes)

for cls in [0, 1, 2]:
    if cls not in unique_classes:
        print(f"Class {cls} missing — adding one sample artificially.")
        df.loc[df.sample(1).index, "severity_label"] = cls

# ---------------------- Prepare Severity Model Data ----------------------
target_sev = "severity_label"
exclude_sev = ["date", "district", "surge_label", "severity_label"]
features_sev = [col for col in df.columns if col not in exclude_sev]

X_sev = df[features_sev]
y_sev = df[target_sev]

# Time-aware split (aligned with surge model)
X_train_sev, X_test_sev = X_sev.iloc[:split_idx], X_sev.iloc[split_idx:]
y_train_sev, y_test_sev = y_sev.iloc[:split_idx], y_sev.iloc[split_idx:]

print("\nTraining multi-class severity model...")

severity_model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

severity_model.fit(X_train_sev, y_train_sev)

# ---------------------- Severity Predictions ----------------------
sev_proba = severity_model.predict_proba(X_test_sev)
sev_pred = severity_model.predict(X_test_sev)

inv_map = {0: "Low", 1: "Moderate", 2: "High"}

df_sev = pd.DataFrame({
    "severity_pred": [inv_map[p] for p in sev_pred],
    "prob_low": sev_proba[:, 0],
    "prob_moderate": sev_proba[:, 1],
    "prob_high": sev_proba[:, 2]
}, index=X_test_sev.index)

print("\nSeverity Model Evaluation:")
print(classification_report(
    y_test_sev,
    sev_pred,
    labels=[0, 1, 2],
    target_names=["Low", "Moderate", "High"],
    zero_division=0
))
print("Severity Confusion Matrix:\n", confusion_matrix(y_test_sev, sev_pred, labels=[0, 1, 2]))

# ============================================================
# 5. SHAP (surge model)
# ============================================================
print("\nComputing SHAP values for surge model...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# ============================================================
# 6. Export Combined Predictions (Surge + Severity)
# ============================================================
df_test["actual_surge"] = y_test.values
df_test["district"] = df.loc[X_test.index, "district"].values
df_test["date"] = df.loc[X_test.index, "date"].values

df_merged = df_test.join(df_sev, how="left")

# Ensure directory exists
os.makedirs("data", exist_ok=True)
os.makedirs("data/preprocessed", exist_ok=True)

df_merged.to_csv("data/predictions_with_severity.csv", index=False)
print("\n✅ Predictions with severity exported to data/predictions_with_severity.csv")

# ============================================================
# 7. Save Models
# ============================================================
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_surge_classifier.json")
severity_model.save_model("models/xgb_severity_classifier.json")

print("✅ Surge and Severity models saved in /models/")
