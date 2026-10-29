from pathlib import Path
import time

import pandas as pd
import requests


BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
COUNT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/count"

START = "2000-01-01"
END = "2026-01-01"   # exclusive boundary for monthly splitting
MIN_MAG = 4.0

OUT_DIR = Path("usgs_raw_m4_2000_2025")
OUT_DIR.mkdir(exist_ok=True)


months = pd.date_range(START, END, freq="MS")

for start, end in zip(months[:-1], months[1:]):
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    filename = OUT_DIR / f"usgs_m4_{start:%Y_%m}.csv"

    if filename.exists() and filename.stat().st_size > 0:
        print(f"Already exists: {filename}")
        continue

    params = {
        "format": "csv",
        "starttime": start_str,
        "endtime": end_str,
        "minmagnitude": MIN_MAG,
        "eventtype": "earthquake",
        "orderby": "time-asc",
    }

    # Check number of matching events first.
    count_response = requests.get(
        COUNT_URL,
        params={k: v for k, v in params.items() if k != "format"},
        timeout=60,
    )
    count_response.raise_for_status()
    count = int(count_response.text.strip())

    print(
        f"{start:%Y-%m}: {count:,} earthquakes",
        flush=True,
    )

    if count == 0:
        continue

    if count > 20000:
        raise RuntimeError(
            f"{start:%Y-%m} has {count:,} events; "
            "split this month into smaller intervals."
        )

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=180,
    )
    response.raise_for_status()

    filename.write_bytes(response.content)

    # Be polite to the service.
    time.sleep(0.3)

print("Finished.")