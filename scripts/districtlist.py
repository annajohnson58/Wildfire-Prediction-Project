import pandas as pd
import numpy as np

df = pd.read_csv("data/daily_climate_district.csv")  # or your source

district_list = df["district"].astype(str).values
date_list = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").values

np.save("data/district_list.npy", district_list)
np.save("data/date_list.npy", date_list)


print("✅ district_list.npy and date_list.npy saved.")
