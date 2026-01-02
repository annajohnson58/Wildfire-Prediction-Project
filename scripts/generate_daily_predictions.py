import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 📅 Define date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)
dates = pd.date_range(start=start_date, end=end_date)

# 🗺️ Define districts (use your actual list if needed)
districts = [
    "Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod",
    "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad",
    "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"
]

# 🔥 Simulate surge levels
np.random.seed(42)
rows = []
for date in dates:
    for district in districts:
        # Simulate risk level: 0 (no fire), 1 (low), 2 (moderate), 3 (high)
        risk = np.random.choice([0, 1, 2, 3], p=[0.85, 0.1, 0.04, 0.01])
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "district": district,
            "risk_level": risk
        })

# 📤 Save to CSV
df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/daily_predictions.csv", index=False)
print("✅ Regenerated daily_predictions.csv with", len(df), "rows.")
