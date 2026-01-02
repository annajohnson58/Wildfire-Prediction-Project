# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# df = pd.read_csv('data/kerala_wildfire_features.csv')

# # Create binary label from fire_count
# df['fire_label'] = (df['fire_count'] > 0).astype(int)

# # Define features and label
# features = ['t2m', 'tp', 'u10', 'v10', 'ndvi', 'ndvi_lag1', 'ndvi_delta']
# X = df[features]
# y = df['fire_label']

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# print("✅ Model trained. Accuracy:", model.score(X_test, y_test))
# print(df['fire_label'].value_counts())

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(model, X, y, cv=5)
# print("Cross-validation scores:", scores)
# print("Mean accuracy:", scores.mean())


# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Predict on test set
# y_pred = model.predict(X_test)

# # 📊 Confusion Matrix
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # 📋 Classification Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # 📈 Feature Importance
# importances = model.feature_importances_
# sns.barplot(x=importances, y=features)
# plt.title("Feature Importance")
# plt.tight_layout()
# plt.show()

# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report

# # 📥 Load fused dataset
# fused = pd.read_csv("data/fused/fused_ndvi_weather_thermal.csv")

# # 🧠 Define features and target
# features = ['ndvi', 'temp', 'rh', 'wind', 'precip', 'dryness_index']
# X = fused[features]
# y = fused['thermal_flag']

# # 🚀 Train model on full data
# model = XGBClassifier()
# model.fit(X, y)

# # 📊 Predict on same data
# y_pred = model.predict(X)
# print(classification_report(y, y_pred, zero_division=0))

# fused['predicted_risk'] = model.predict_proba(X)[:, 1]
# fused[['district', 'predicted_risk']].to_csv("data/fused/predicted_risk.csv", index=False)



import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#  Load fused dataset
fused = pd.read_csv("data/fused/fused_ndvi_weather_thermal.csv")

#  Define features and target
features = ['ndvi', 'temp', 'rh', 'wind', 'precip', 'dryness_index']
X = fused[features]
y = fused['thermal_flag']

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

#  Safe SMOTE resampling
if sum(y_train == 1) >= 2:
    print(f" Applying SMOTE (minority samples: {sum(y_train == 1)})")
    smote = SMOTE(k_neighbors=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    print(f" Too few minority samples ({sum(y_train == 1)}) for SMOTE. Skipping resampling.")
    X_train_resampled, y_train_resampled = X_train, y_train

#  Train model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate on test set
y_pred = model.predict(X_test)
print("\n Evaluation on test set:")
print(classification_report(y_test, y_pred, zero_division=0))

#  Export test predictions
fused_test = fused.iloc[y_test.index].copy()
fused_test['predicted_risk'] = model.predict_proba(X_test)[:, 1]
fused_test[['district', 'predicted_risk']].to_csv("data/fused/predicted_risk.csv", index=False)
print("Test predictions exported to data/fused/predicted_risk.csv")
