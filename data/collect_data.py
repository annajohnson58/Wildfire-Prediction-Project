import pandas as pd

def collect_data(file_path):
    """
    Reads wildfire sensor data from a CSV file and performs basic validation.
    Returns a cleaned DataFrame.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Show basic info
        print("Original Data:")
        print(df.head())

        # Drop rows with missing values
        df_cleaned = df.dropna()

        # Optional: filter out unrealistic values
        df_cleaned = df_cleaned[
            (df_cleaned['temperature'] > 0) &
            (df_cleaned['humidity'] >= 0) &
            (df_cleaned['wind_speed'] >= 0) &
            (df_cleaned['vegetation_index'] >= 0)
        ]

        print("\nCleaned Data:")
        print(df_cleaned.head())

        return df_cleaned

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Run the function if this file is executed directly
if __name__ == "__main__":
    data = collect_data("data/mock_inputs.csv")
