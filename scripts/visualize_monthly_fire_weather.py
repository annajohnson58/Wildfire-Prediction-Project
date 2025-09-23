import pandas as pd
import matplotlib.pyplot as plt
import os

# Load merged dataset
base = 'C:/Users/annac/OneDrive/Desktop/projects/Foresight-for-Forests/EarthEngineExports'
df = pd.read_csv(os.path.join(base, 'merged_monthly_fire_weather_thrissur_2024.csv'))

# Convert month to datetime
df['month'] = pd.to_datetime(df['month'])

# Plot fire counts vs temperature
plt.figure(figsize=(10, 5))
plt.plot(df['month'], df['temp_C'], label='Temperature (°C)', color='orange')
plt.bar(df['month'], df['fire_label'], label='Fire Count', alpha=0.4, color='red')
plt.title('🔥 Monthly Fire Activity vs Temperature in Thrissur (2024)')
plt.xlabel('Month')
plt.ylabel('Fire Count / Temperature')
plt.legend()
plt.tight_layout()
plt.show()

# Plot fire counts vs precipitation
plt.figure(figsize=(10, 5))
plt.plot(df['month'], df['precip_m'] * 1000, label='Precipitation (mm)', color='blue')  # Convert m to mm
plt.bar(df['month'], df['fire_label'], label='Fire Count', alpha=0.4, color='red')
plt.title('🔥 Monthly Fire Activity vs Rainfall in Thrissur (2024)')
plt.xlabel('Month')
plt.ylabel('Fire Count / Rainfall (mm)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot fire counts vs wind speed
plt.figure(figsize=(10, 5))
plt.plot(df['month'], df['wind_speed'], label='Wind Speed (m/s)', color='green')
plt.bar(df['month'], df['fire_label'], label='Fire Count', alpha=0.4, color='red')
plt.title('🔥 Monthly Fire Activity vs Wind Speed in Thrissur (2024)')
plt.xlabel('Month')
plt.ylabel('Fire Count / Wind Speed')
plt.legend()
plt.tight_layout()
plt.show()
