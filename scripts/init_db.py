import sqlite3
import os

DB_PATH = "data/wildfire_grid.db"
os.makedirs("data", exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1) Grid cells: exact locations
cur.execute("""
CREATE TABLE IF NOT EXISTS grid_cells (
    grid_id TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    district TEXT,
    forest TEXT
);
""")

# 2) Daily features for each grid cell
cur.execute("""
CREATE TABLE IF NOT EXISTS grid_daily_features (
    grid_id TEXT NOT NULL,
    date TEXT NOT NULL,
    ndvi REAL,
    temperature REAL,
    rainfall REAL,
    wind REAL,
    dryness_index REAL,
    firs_hotspots INTEGER,
    PRIMARY KEY (grid_id, date),
    FOREIGN KEY (grid_id) REFERENCES grid_cells(grid_id)
);
""")

# 3) Labels: D+2 fire + severity
cur.execute("""
CREATE TABLE IF NOT EXISTS grid_labels (
    grid_id TEXT NOT NULL,
    date TEXT NOT NULL,            -- base date D
    fire_Dplus2 INTEGER,           -- 0/1
    severity_Dplus2 INTEGER,       -- 0=Low,1=Moderate,2=High
    PRIMARY KEY (grid_id, date),
    FOREIGN KEY (grid_id) REFERENCES grid_cells(grid_id)
);
""")

# 4) Predictions
cur.execute("""
CREATE TABLE IF NOT EXISTS grid_predictions (
    grid_id TEXT NOT NULL,
    date TEXT NOT NULL,            -- base date D
    forecast_date TEXT NOT NULL,   -- D+2
    fire_prob REAL,
    severity_pred TEXT,
    prob_low REAL,
    prob_moderate REAL,
    prob_high REAL,
    model_version TEXT,
    PRIMARY KEY (grid_id, date),
    FOREIGN KEY (grid_id) REFERENCES grid_cells(grid_id)
);
""")

conn.commit()
conn.close()
print("✅ Database initialized at data/wildfire_grid.db")
