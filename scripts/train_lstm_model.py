
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

#  Load data
X = np.load("data/X_lstm.npy")
y = np.load("data/y_lstm.npy")

#  Define LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Predict thermal_count
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#  Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X, y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

#  Save model
os.makedirs("models", exist_ok=True)
model.save("models/lstm_hotspot_forecaster.keras")

print(" Model saved: models/lstm_hotspot_forecaster.keras")
