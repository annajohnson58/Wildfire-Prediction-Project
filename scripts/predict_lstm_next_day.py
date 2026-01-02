import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load latest data
df = pd.read_csv("data/daily_climate.csv")
features = ["ndvi", "thermal_count", "temperature", "rh", "wind", "rainfall"]
df = df[features].dropna()

# Scale and extract last 7 days
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)
last_seq = scaled[-7:].reshape(1, 7, len(features))

# Load model and predict
model = load_model("models/lstm_surge_predictor.h5")
proba = model.predict(last_seq)[0][0]
pred = int(proba > 0.3)

print(f"🧠 LSTM Surge Prediction for Tomorrow:")
print(f"Probability: {proba:.2f} → Predicted Surge: {bool(pred)}")
