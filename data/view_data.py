import sqlite3

conn = sqlite3.connect("wildfire.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM daily_features")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
