import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------- Load Training Matrix ----------------------
df = pd.read_csv("data/training_matrix.csv")
df["date"] = pd.to_datetime(df["date"])

# ---------------------- Define Features and Target ----------------------
features = ["ndvi", "ndvi_drop", "temperature", "rh", "rh_anomaly", "wind", "rainfall"]
target = "thermal_count"  # or "thermal_spike" if you're forecasting sudden fire surges

X = df[features]
y = df[target]

# ---------------------- Train/Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------- Optional: Scale Features ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
