# fetch_aqi_data.py
import os
import requests
import pandas as pd
from datetime import datetime

# === Configuration ===
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")
if not WAQI_TOKEN:
    raise RuntimeError("❌ WAQI_TOKEN not found in environment variables")

URL = f"https://api.waqi.info/feed/A545332/?token={WAQI_TOKEN}"
FILE_NAME = "csv/hourly_aqi_data.csv"


def safe_get(iaqi_dict, key):
    """Safely extract a value from IAQI data."""
    try:
        val = iaqi_dict.get(key, {}).get("v")
        if val is None or (isinstance(val, (int, float)) and val < 0):
            return None
        return val
    except Exception:
        return None


def fetch_and_save():
    print(f"[{datetime.now()}] Fetching data from WAQI API...")
    try:
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "ok":
            print("❌ API returned error:", data.get("data"))
            return

        d = data["data"]

         # === Extract top-level info ===
        city = "NED University, Karachi"  # Hardcoded to avoid 'city' key errors
        aqi = d.get("aqi", None)
        dominentpol = d.get("dominentpol", None)
        time_stamp = d.get("time", {}).get("s", datetime.now().isoformat())

        iaqi = d.get("iaqi", {})

         # === Extract IAQI values ===
        iaqi = d.get("iaqi", {})                # Insert into the iaqi values

        temp = safe_get(iaqi, "t")
        humidity = safe_get(iaqi, "h")
        pm1 = safe_get(iaqi, "pm1")
        pm10 = safe_get(iaqi, "pm10")
        pm25 = safe_get(iaqi, "pm25")


        # === Create record ===
        record = {
            "timestamp": time_stamp,
            "city": city,
            "dominentpol": dominentpol,
            "temperature": temp,
            "humidity": humidity,
            "pm1": pm1,
            "pm10": pm10,
            "pm25": pm25,
            "aqi": aqi,
        }

        df = pd.DataFrame([record])

        # === Handle file safety ===
        if os.path.exists(FILE_NAME) and os.path.getsize(FILE_NAME) > 0:
            try:
                existing = pd.read_csv(FILE_NAME)
                if (
                    (existing["timestamp"] == record["timestamp"])
                    & (existing["aqi"] == record["aqi"])
                ).any():
                    print(f"⚠️ Duplicate record detected for {record['timestamp']}. Skipping save.")
                    return
            except Exception as e:
                print(f"⚠️ CSV read issue ({e}), recreating new file.")
                os.remove(FILE_NAME)

        # === Save record ===
        df.to_csv(
            FILE_NAME,
            mode="a",
            header=not os.path.exists(FILE_NAME) or os.path.getsize(FILE_NAME) == 0,
            index=False,
        )
        print(f"✅ Saved record for {record['city']} at {record['timestamp']}")

    except Exception as e:
        print(f"💥 Error while fetching or saving data: {e}")


if __name__ == "__main__":
    fetch_and_save()
