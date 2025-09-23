import pandas as pd
from datetime import datetime
import os

# Create output folders
os.makedirs("data", exist_ok=True)

# Today's date
today = datetime.today().strftime("%Y-%m-%d")

# Kerala districts (sample subset)
districts = ["Thrissur", "Idukki", "Palakkad", "Wayanad", "Kozhikode", "Pathanamthitta"]

# 🔥 Dummy predictions
predictions = pd.DataFrame({
    "date": [today] * len(districts),
    "district": districts,
    "risk_level": ["High", "Medium", "Low", "Medium", "High", "Low"]
})
predictions.to_csv("data/daily_predictions.csv", index=False)
print("✅ Created: data/daily_predictions.csv")

# 🚨 Dummy alerts
alerts = pd.DataFrame({
    "date": [today] * 3,
    "district": ["Thrissur", "Wayanad", "Kozhikode"],
    "alert_type": ["Fire Risk", "Dryness Warning", "Heat Alert"],
    "message": [
        "High fire risk due to low NDVI and high temperature",
        "Dryness index elevated—monitor vegetation stress",
        "High temperature spike detected—stay alert"
    ]
})
alerts.to_csv("data/alerts.csv", index=False)
print("✅ Created: data/alerts.csv")
