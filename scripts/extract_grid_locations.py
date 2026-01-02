import sqlite3
import pandas as pd

# Connect to your database
conn = sqlite3.connect("data/wildfire_grid.db")

# Extract grid_id, lat, lon from grid_cells
query = """
SELECT DISTINCT grid_id, lat, lon
FROM grid_cells
"""

df = pd.read_sql_query(query, conn)
conn.close()

# Save to CSV
df.to_csv("data/grid_locations.csv", index=False)
print(f"✅ grid_locations.csv created with {len(df)} grid cells.")
