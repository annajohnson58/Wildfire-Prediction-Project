import joblib
import matplotlib.pyplot as plt

model = joblib.load("models/fire_predictor.pkl")
features = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'windspeed_10m_max']
importances = model.feature_importances_

plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
