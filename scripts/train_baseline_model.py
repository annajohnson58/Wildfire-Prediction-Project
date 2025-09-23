import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Load your dataset
df = pd.read_csv('data/kerala_wildfire_dataset_2024.csv')

# Preview the data
print(df.head())
# Sort by district and month
df = df.sort_values(by=['district', 'month'])

# Create lag-1 features for each climate variable and fire count
for col in ['t2m', 'tp', 'u10', 'v10', 'fire_count']:
    df[f'{col}_lag1'] = df.groupby('district')[col].shift(1)
df = df.dropna().reset_index(drop=True)
df = pd.get_dummies(df, columns=['district'])
X = df.drop(columns=['month', 'fire_count'])  # Features
y = df['fire_count']                          # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # No shuffle to preserve time order
)


# Create and train the model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Predict fire counts
preds = model.predict(X_test)

# Evaluate performance
import numpy as np

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)


xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Fire Count', marker='o')
plt.plot(preds, label='Predicted Fire Count', marker='x')
plt.title('Wildfire Prediction: Actual vs. Predicted')
plt.xlabel('Time (Test Months)')
plt.ylabel('Fire Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

xgb.plot_importance(model, max_num_features=10)
plt.title("Top Features Influencing Fire Count")
plt.tight_layout()
plt.show()
