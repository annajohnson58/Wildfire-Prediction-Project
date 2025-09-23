import psycopg2

# Database connection details
DB_NAME = "foresight_db"
DB_USER = "foresight_user"
DB_PASSWORD = "Foresight@2025"
DB_HOST = "localhost"
DB_PORT = "5432"

def create_tables():
    conn = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        # Create iot_sensor_data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS iot_sensor_data (
                id SERIAL PRIMARY KEY,
                sensor_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                temperature_c NUMERIC,
                humidity_rh NUMERIC,
                smoke_co_ppm NUMERIC,
                rain_detected BOOLEAN,
                battery_level NUMERIC
            );
        """)
        print("Table 'iot_sensor_data' created or already exists.")

        # Create satellite_data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS satellite_data (
                id SERIAL PRIMARY KEY,
                forest_grid_id VARCHAR(100) NOT NULL,
                image_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                avg_ndvi NUMERIC,
                avg_ndwi NUMERIC,
                avg_nbr NUMERIC,
                firms_hotspot_count INTEGER,
                firms_power_sum NUMERIC,
                UNIQUE (forest_grid_id, image_timestamp)
            );
        """)
        print("Table 'satellite_data' created or already exists.")

        # Create meteorological_data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meteorological_data (
                id SERIAL PRIMARY KEY,
                forest_grid_id VARCHAR(100) NOT NULL,
                met_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                avg_wind_speed_kmh NUMERIC,
                wind_direction_deg NUMERIC,
                precipitation_24h_mm NUMERIC,
                days_since_last_rain INTEGER,
                drought_index_spi NUMERIC,
                UNIQUE (forest_grid_id, met_timestamp)
            );
        """)
        print("Table 'meteorological_data' created or already exists.")

        # Create wildfire_alerts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wildfire_alerts (
                id SERIAL PRIMARY KEY,
                alert_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                alert_type VARCHAR(50) NOT NULL,
                location_lat NUMERIC,
                location_lon NUMERIC,
                risk_score NUMERIC,
                status VARCHAR(50) DEFAULT 'Active'
            );
        """)
        print("Table 'wildfire_alerts' created or already exists.")

        # Commit the changes
        conn.commit()

    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    create_tables()
