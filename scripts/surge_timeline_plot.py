import numpy as np
import matplotlib.pyplot as plt
import joblib

# 📂 Load data
X = np.load("data/X_lstm.npy")
y_true = np.load("data/y_surge.npy")  # Binary surge labels
y_reg = np.load("data/y_lstm.npy")    # Scaled thermal count
X_flat = X.reshape(X.shape[0], -1)

# 🔍 Load classifier
model = joblib.load("models/surge_classifier.pkl")
y_cls = model.predict(X_flat)

# 🧠 Ensemble alert logic
ensemble_alerts = (y_reg > 0.6) & (y_cls == 1)

# 📊 Plot surge timeline
plt.figure(figsize=(12, 4))
plt.plot(y_reg, color="dodgerblue", label="LSTM Prediction", linewidth=1)
plt.scatter(np.where(y_true == 1), [1]*sum(y_true), color="firebrick", label="Actual Surge", marker="|", s=100)
plt.scatter(np.where(y_cls == 1), [0.8]*sum(y_cls), color="limegreen", label="XGBoost Surge Flag", marker="x")
plt.scatter(np.where(ensemble_alerts), [0.95]*sum(ensemble_alerts), color="purple", label="Ensemble Alert", marker="o")

plt.xlabel("Sample Index")
plt.ylabel("Surge Signal")
plt.title("🔥 Surge Timeline: Actual vs Predicted")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("data/surge_timeline_plot.png")
plt.show()
