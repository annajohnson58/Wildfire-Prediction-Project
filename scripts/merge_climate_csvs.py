import pandas as pd
import glob

files = sorted(glob.glob('data/climate_features/climate_2024-*.csv'))
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_all.to_csv('data/climate_features/kerala_climate_2024.csv', index=False)
