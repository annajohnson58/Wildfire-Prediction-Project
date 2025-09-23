from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import json
from datetime import datetime
import threading
import paho.mqtt.client as mqtt # Will install this soon

    # --- Database Configuration ---
DB_NAME = "foresight_db"
DB_USER = "foresight_user"
DB_PASSWORD = "Foresight@2025" # <<< IMPORTANT: REPLACE THIS with your actual DB password!
DB_HOST = "localhost"
DB_PORT = "5432"

    # --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_SENSOR_DATA = "foresight/sensors/+/data" # Wildcard to subscribe to all sensor data topics

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

    # --- Database Connection Helper ---
def get_db_connection():
        """Establishes and returns a new PostgreSQL database connection."""
        try:
            conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT
            )
            return conn
        except Exception as e:
            print(f"Database connection failed: {e}")
            return None

    # --- Data Insertion Functions ---
def insert_iot_sensor_data(sensor_data):
        """Inserts IoT sensor data into the iot_sensor_data table."""
        conn = get_db_connection()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO iot_sensor_data (sensor_id, timestamp, temperature_c, humidity_rh, smoke_co_ppm, rain_detected, battery_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                sensor_data.get('sensor_id'),
                sensor_data.get('timestamp'), # Expecting ISO format string or datetime object
                sensor_data.get('temperature_c'),
                sensor_data.get('humidity_rh'),
                sensor_data.get('smoke_co_ppm'),
                sensor_data.get('rain_detected'),
                sensor_data.get('battery_level')
            ))
            conn.commit()
            print(f"Inserted IoT sensor data for {sensor_data.get('sensor_id')} at {sensor_data.get('timestamp')}")
            return True
        except Exception as e:
            print(f"Error inserting IoT sensor data: {e}")
            conn.rollback() # Rollback in case of error
            return False
        finally:
            if conn:
                cur.close()
                conn.close()

def insert_satellite_data(data_list):
        """Inserts a list of satellite data entries into the satellite_data table."""
        conn = get_db_connection()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            for data_entry in data_list:
                cur.execute("""
                    INSERT INTO satellite_data (forest_grid_id, image_timestamp, avg_ndvi, avg_ndwi, avg_nbr, firms_hotspot_count, firms_power_sum)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (forest_grid_id, image_timestamp) DO UPDATE SET
                        avg_ndvi = EXCLUDED.avg_ndvi,
                        avg_ndwi = EXCLUDED.avg_ndwi,
                        avg_nbr = EXCLUDED.avg_nbr,
                        firms_hotspot_count = EXCLUDED.firms_hotspot_count,
                        firms_power_sum = EXCLUDED.firms_power_sum;
                """, (
                    data_entry.get('forest_grid_id'),
                    data_entry.get('image_timestamp'),
                    data_entry.get('avg_ndvi'),
                    data_entry.get('avg_ndwi'),
                    data_entry.get('avg_nbr'),
                    data_entry.get('firms_hotspot_count'),
                    data_entry.get('firms_power_sum')
                ))
            conn.commit()
            print(f"Inserted/Updated {len(data_list)} satellite data entries.")
            return True
        except Exception as e:
            print(f"Error inserting satellite data: {e}")
            conn.rollback()
            return False
        finally:
            if conn:
                cur.close()
                conn.close()
def insert_meteorological_data(data_list):
        """Inserts a list of meteorological data entries into the meteorological_data table."""
        conn = get_db_connection()
        if conn is None:
            return False
        try:
            cur = conn.cursor()
            for data_entry in data_list:
                cur.execute("""
                    INSERT INTO meteorological_data (forest_grid_id, met_timestamp, avg_wind_speed_kmh, wind_direction_deg, precipitation_24h_mm, days_since_last_rain, drought_index_spi)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (forest_grid_id, met_timestamp) DO UPDATE SET
                        avg_wind_speed_kmh = EXCLUDED.avg_wind_speed_kmh,
                        wind_direction_deg = EXCLUDED.wind_direction_deg,
                        precipitation_24h_mm = EXCLUDED.precipitation_24h_mm,
                        days_since_last_rain = EXCLUDED.days_since_last_rain,
                        drought_index_spi = EXCLUDED.drought_index_spi;
                """, (
                    data_entry.get('forest_grid_id'),
                    data_entry.get('met_timestamp'),
                    data_entry.get('avg_wind_speed_kmh'),
                    data_entry.get('wind_direction_deg'),
                    data_entry.get('precipitation_24h_mm'),
                    data_entry.get('days_since_last_rain'),
                    data_entry.get('drought_index_spi')
                ))
            conn.commit()
            print(f"Inserted/Updated {len(data_list)} meteorological data entries.")
            return True
        except Exception as e:
            print(f"Error inserting meteorological data: {e}")
            conn.rollback()
            return False
        finally:
            if conn:
                cur.close()
                conn.close()

    # --- RESTful API Endpoints ---
@app.route('/')
def home():
        """A simple home endpoint to check if the server is running."""
        return jsonify({"message": "Foresight AI Backend is running!"})

@app.route('/api/v1/data/iot', methods=['POST'])
def receive_iot_data():
        """
        Receives processed IoT sensor data.
        Expected JSON format (example):
        {
            "sensor_id": "sensor001",
            "timestamp": "2024-07-29T10:30:00Z",
            "temperature_c": 28.5,
            "humidity_rh": 70.2,
            "smoke_co_ppm": 150,
            "rain_detected": false,
            "battery_level": 95
        }
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Basic validation (can be expanded)
        if not all(k in data for k in ['sensor_id', 'timestamp', 'temperature_c', 'humidity_rh', 'smoke_co_ppm']):
            return jsonify({"error": "Missing required IoT data fields"}), 400

        if insert_iot_sensor_data(data):
            return jsonify({"message": "IoT data received and stored"}), 200
        else:
            return jsonify({"error": "Failed to store IoT data"}), 500

@app.route('/api/v1/data/satellite', methods=['POST'])
def receive_satellite_data():
        """
        Receives processed satellite data (e.g., NDVI, NBR, FIRMS hotspots aggregated).
        Expected JSON format: A list of data entries.
        [
            {
                "forest_grid_id": "grid_A1",
                "image_timestamp": "2024-07-29T08:00:00Z",
                "avg_ndvi": 0.65,
                "avg_ndwi": 0.30,
                "avg_nbr": -0.15,
                "firms_hotspot_count": 0,
                "firms_power_sum": 0.0
            },
            {
                "forest_grid_id": "grid_B2",
                "image_timestamp": "2024-07-29T08:00:00Z",
                "avg_ndvi": 0.70,
                "avg_ndwi": 0.35,
                "avg_nbr": -0.20,
                "firms_hotspot_count": 1,
                "firms_power_sum": 15.2
            }
        ]
        """
        data_list = request.get_json()
        if not isinstance(data_list, list) or not data_list:
            return jsonify({"error": "Expected a list of satellite data entries"}), 400
        
        # Basic validation for first item (can be expanded for all)
        if not all(k in data_list[0] for k in ['forest_grid_id', 'image_timestamp', 'avg_ndvi']):
            return jsonify({"error": "Missing required satellite data fields"}), 400

        if insert_satellite_data(data_list):
            return jsonify({"message": f"{len(data_list)} satellite data entries received and stored"}), 200
        else:
            return jsonify({"error": "Failed to store satellite data"}), 500

@app.route('/api/v1/data/meteorological', methods=['POST'])
def receive_meteorological_data():
        """
        Receives processed meteorological data.
        Expected JSON format: A list of data entries.
        [
            {
                "forest_grid_id": "grid_A1",
                "met_timestamp": "2024-07-29T12:00:00Z",
                "avg_wind_speed_kmh": 10.5,
                "wind_direction_deg": 270,
                "precipitation_24h_mm": 5.2,
                "days_since_last_rain": 2,
                "drought_index_spi": -0.5
            }
        ]
        """
        data_list = request.get_json()
        if not isinstance(data_list, list) or not data_list:
            return jsonify({"error": "Expected a list of meteorological data entries"}), 400
        
        # Basic validation for first item (can be expanded for all)
        if not all(k in data_list[0] for k in ['forest_grid_id', 'met_timestamp', 'avg_wind_speed_kmh']):
            return jsonify({"error": "Missing required meteorological data fields"}), 400

        if insert_meteorological_data(data_list):
            return jsonify({"message": f"{len(data_list)} meteorological data entries received and stored"}), 200
        else:
            return jsonify({"error": "Failed to store meteorological data"}), 500

    # --- MQTT Client Setup ---
def on_connect(client, userdata, flags, rc):
        """Callback function when the MQTT client connects to the broker."""
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(MQTT_TOPIC_SENSOR_DATA) # Subscribe to sensor data topic
            print(f"Subscribed to MQTT topic: {MQTT_TOPIC_SENSOR_DATA}")
        else:
            print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
        """Callback function when a message is received on a subscribed topic."""
        print(f"MQTT Message Received on topic {msg.topic}: {msg.payload.decode()}")
        try:
            # Assuming sensor data comes as JSON
            sensor_data = json.loads(msg.payload.decode())
            
            # Ensure timestamp is in correct format (e.g., ISO 8601)
            if 'timestamp' in sensor_data:
                # Attempt to parse if it's a string, then convert to PostgreSQL's preferred format
                sensor_data['timestamp'] = datetime.fromisoformat(sensor_data['timestamp'].replace('Z', '+00:00'))
            else:
                sensor_data['timestamp'] = datetime.now() # Fallback to current time if no timestamp provided

            # Insert into database
            insert_iot_sensor_data(sensor_data)

        except json.JSONDecodeError:
            print("Error: Received non-JSON MQTT message.")
        except Exception as e:
            print(f"Error processing MQTT message: {e}")

    # Create MQTT client instance
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

def start_mqtt_client():
        """Starts the MQTT client loop in a separate thread."""
        try:
            mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            mqtt_client.loop_forever() # Blocks, so run in a separate thread
        except Exception as e:
            print(f"MQTT client connection error: {e}")

    # --- Main Application Run ---
if __name__ == '__main__':
        # Start MQTT client in a separate thread so Flask app can also run
        mqtt_thread = threading.Thread(target=start_mqtt_client)
        mqtt_thread.daemon = True # Allow main program to exit even if thread is running
        mqtt_thread.start()
        
        print("Starting Flask application...")
        app.run(debug=True, port=5000) # debug=True restarts server on code changes, port 5000 is common
   