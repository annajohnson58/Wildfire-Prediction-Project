import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 📂 Load data and model
X = np.load("data/X_lstm.npy")
y_true = np.load("data/y_lstm.npy")
model = load_model("models/lstm_hotspot_forecaster.keras")

# 🔮 Predict
y_pred = model.predict(X).flatten()

# 📊 Plot
plt.figure(figsize=(12, 6))
plt.plot(y_true, label="Actual", color="firebrick")
plt.plot(y_pred, label="Predicted", color="dodgerblue")
plt.title("🔥 Thermal Count: Actual vs. Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Thermal Count (scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/lstm_prediction_plot.png")
plt.show()
