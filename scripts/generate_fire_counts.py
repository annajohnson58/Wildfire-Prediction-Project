import pandas as pd

def aggregate_fire(file_path, district_name):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.strftime('%Y-%m')
    monthly = df.groupby('month')['fire_label'].sum().reset_index()
    monthly['district'] = district_name
    return monthly

# Aggregate for each district
thrissur = aggregate_fire('data/fire_labels_thrissur.csv', 'Thrissur')
palakkad = aggregate_fire('data/fire_labels_palakkad.csv', 'Palakkad')
idukki = aggregate_fire('data/fire_labels_idukki.csv', 'Idukki')

# Combine and save
fire_counts = pd.concat([thrissur, palakkad, idukki], ignore_index=True)
fire_counts.rename(columns={'fire_label': 'fire_count'}, inplace=True)
fire_counts.to_csv('data/fire_counts_2024.csv', index=False)
