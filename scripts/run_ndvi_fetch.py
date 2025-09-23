import subprocess
import datetime

def run_ndvi_fetch():
    try:
        print(f"🔄 Running NDVI fetch for {datetime.date.today()}")
        subprocess.run([
    r"C:\Users\annac\OneDrive\Desktop\projects\Foresight-for-Forests\foresight_ai_env_py310\Scripts\python.exe",
    "scripts/fetch_ndvi.py"
], check=True)

        print("✅ NDVI fetch completed.")
    except subprocess.CalledProcessError:
        print("⚠️ NDVI fetch failed. Will retry tomorrow.")

if __name__ == "__main__":
    run_ndvi_fetch()
with open("logs/ndvi_log.txt", "a") as log:
    log.write(f"{datetime.datetime.now()} - NDVI fetch completed\n")

try:
    import pandas as pd
except ImportError:
    print("❌ pandas not installed. Run: pip install pandas")
    exit(1)
