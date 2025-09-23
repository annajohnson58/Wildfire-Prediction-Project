# init_db.py
import psycopg2

DB_NAME = "foresight_db"
DB_USER = "foresight_user"
DB_PASSWORD = "Daivam@45An" # <<< IMPORTANT: REPLACE THIS with your project DB password!
DB_HOST = "localhost"
DB_PORT = "5432"

def create_tables():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        # Table for IoT Sensor Data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iot_sensor_data (
                id SERIAL PRIMARY KEY,
                sensor_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                temperature_c REAL,
                humidity_rh REAL,
                smoke_co_ppm INTEGER,
                rain_detected BOOLEAN,
                battery_level INTEGER,
                latitude REAL,  -- Added for sensor location
                longitude REAL  -- Added for sensor location
            );
        """)
        print("Table 'iot_sensor_data' created or already exists.")

        # Table for Satellite Data (e.g., NDVI, FIRMS hotspots)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS satellite_data (
                id SERIAL PRIMARY KEY,
                forest_grid_id VARCHAR(50) NOT NULL,
                image_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                avg_ndvi REAL,
                avg_ndwi REAL,
                avg_nbr REAL,
                firms_hotspot_count INTEGER,
                firms_power_sum REAL,
                UNIQUE (forest_grid_id, image_timestamp) -- Ensure unique entries per grid per timestamp
            );
        """)
        print("Table 'satellite_data' created or already exists.")

        # Table for Meteorological Data (e.g., wind, precipitation)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meteorological_data (
                id SERIAL PRIMARY KEY,
                forest_grid_id VARCHAR(50) NOT NULL,
                met_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                avg_wind_speed_kmh REAL,
                wind_direction_deg REAL,
                precipitation_24h_mm REAL,
                days_since_last_rain INTEGER,
                drought_index_spi REAL,
                UNIQUE (forest_grid_id, met_timestamp) -- Ensure unique entries per grid per timestamp
            );
        """)
        print("Table 'meteorological_data' created or already exists.")

        # Table for AI Predictions (Wildfire Risk)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_predictions (
                id SERIAL PRIMARY KEY,
                forest_grid_id VARCHAR(50) NOT NULL,
                prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                risk_score REAL NOT NULL, -- Probability of fire (0-1)
                risk_level VARCHAR(20),   -- e.g., 'Low', 'Moderate', 'High', 'Critical'
                model_version VARCHAR(50),
                UNIQUE (forest_grid_id, prediction_timestamp)
            );
        """)
        print("Table 'ai_predictions' created or already exists.")

        # Table for Wildfire Alerts (triggered by AI or direct sensor detection)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wildfire_alerts (
                id SERIAL PRIMARY KEY,
                alert_id VARCHAR(100) UNIQUE NOT NULL, -- Unique ID for the alert
                sensor_id VARCHAR(50), -- If triggered by a specific sensor
                forest_grid_id VARCHAR(50) NOT NULL,
                alert_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                alert_type VARCHAR(50) NOT NULL, -- e.g., 'High Risk Prediction', 'Sensor Detected Fire'
                risk_level VARCHAR(20) NOT NULL, -- e.g., 'High', 'Critical'
                status VARCHAR(20) NOT NULL, -- e.g., 'Active', 'Acknowledged', 'Resolved', 'False Alarm'
                latitude REAL,  -- For map display
                longitude REAL, -- For map display
                description TEXT,
                resolution_notes TEXT,
                resolved_timestamp TIMESTAMP WITH TIME ZONE
            );
        """)
        print("Table 'wildfire_alerts' created or already exists.")

        conn.commit()
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    create_tables()