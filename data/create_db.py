import sqlite3

# Connect to (or create) a local database file
conn = sqlite3.connect("wildfire.db")
cursor = conn.cursor()

# Create a table to store daily predictions
cursor.execute("""
CREATE TABLE IF NOT EXISTS daily_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    district TEXT,
    date TEXT,
    ndvi_mean REAL,
    thermal_mean REAL,
    rainfall_mean REAL,
    wind_mean REAL,
    surge_prob REAL,
    alert_triggered BOOLEAN,
    model_version TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS district_coords (
    district TEXT PRIMARY KEY,
    latitude REAL,
    longitude REAL
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")

# Optional: Insert sample users
cursor.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", ("anna", "foresight123"))
cursor.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", ("admin", "secure456"))


conn.commit()
conn.close()
print("✅ Database and table created!")
