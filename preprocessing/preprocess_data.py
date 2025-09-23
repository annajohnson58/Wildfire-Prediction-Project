import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import importlib.util
import os

# Dynamically load collect_data.py
def load_collect_data():
    path = os.path.abspath("../Foresight-for-Forests/data/collect_data.py")
    spec = importlib.util.spec_from_file_location("collect_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect_data

def preprocess_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month

    features_to_scale = ['temperature', 'humidity', 'wind_speed', 'vegetation_index']
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=[f"{col}_scaled" for col in features_to_scale])

    df_processed = pd.concat([df[['location', 'hour', 'day', 'month']], df_scaled], axis=1)

    print("\nProcessed Features:")
    print(df_processed.head())

    return df_processed

# Run if executed directly
if __name__ == "__main__":
    collect_data = load_collect_data()
    raw_df = collect_data("data/mock_inputs.csv")
    processed_df = preprocess_data(raw_df)
