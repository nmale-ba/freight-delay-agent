"""
fetch_eia_diesel.py
Standalone EIA diesel fetch with diagnostics (self-contained; requires requests, pandas).

Use this when fetch_external_data.py reports "diesel (EIA): 0 rows".
It does NOT touch route_conditions or fx_rates.

    Windows PowerShell:  $env:EIA_API_KEY="your_key";  python fetch_eia_diesel.py
    macOS / Linux:       export EIA_API_KEY=your_key;  python fetch_eia_diesel.py
    Or pass it directly: python fetch_eia_diesel.py --key your_key

Outputs:
    external/diesel_prices_eia.csv
    external/diesel_load.sql     <- run this in VS Code
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

EIA_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
SERIES_ID = "EMD_EPD2D_PTE_NUS_DPG"     # U.S. No 2 diesel retail price, $/gal, weekly
START = "2024-01-01"
OUT_DIR = "external"


def call_eia(key, facet_params, label):
    params = {
        "api_key": key,
        "frequency": "weekly",
        "data[0]": "value",
        "start": START,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    params.update(facet_params)
    print(f"\nAttempt: {label}")
    try:
        resp = requests.get(EIA_URL, params=params, timeout=30)
    except requests.RequestException as exc:
        print(f"  Network error: {exc}")
        return []
    print(f"  HTTP status: {resp.status_code}")
    try:
        body = resp.json()
    except ValueError:
        print(f"  Non-JSON reply (first 300 chars): {resp.text[:300]}")
        return []
    if "error" in body:
        print(f"  EIA error: {body['error']}")
    response = body.get("response", {})
    for w in response.get("warnings", []) or []:
        print(f"  EIA warning: {w}")
    data = response.get("data", []) or []
    print(f"  Rows returned: {len(data)} (EIA reports total = {response.get('total')})")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=None, help="EIA API key (otherwise read from EIA_API_KEY)")
    args = parser.parse_args()

    key = (args.key or os.environ.get("EIA_API_KEY", "")).strip()
    if not key:
        sys.exit("No key found. Set EIA_API_KEY in THIS terminal, or run: python fetch_eia_diesel.py --key YOUR_KEY")
    print(f"Using key ending in ...{key[-4:]} (length {len(key)})")

    # Attempt 1: filter by series ID
    data = call_eia(key, {"facets[series][]": SERIES_ID}, f"series = {SERIES_ID}")

    # Attempt 2: filter by product + area + process, then keep the diesel retail series
    if not data:
        data = call_eia(key, {"facets[product][]": "EPD2D", "facets[duoarea][]": "NUS",
                              "facets[process][]": "PTE"},
                        "product = EPD2D, area = NUS, process = PTE")
        data = [r for r in data if r.get("series", SERIES_ID) == SERIES_ID]

    rows = [{"week_start": r["period"], "diesel_price_usd_gal": round(float(r["value"]), 3),
             "series_id": SERIES_ID, "source": "EIA"}
            for r in data if r.get("value") not in (None, "")]
    if not rows:
        sys.exit("\nStill 0 rows. Copy everything above and send it over.")

    df = pd.DataFrame(rows).drop_duplicates("week_start").sort_values("week_start")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "diesel_prices_eia.csv"), index=False)

    values = ",\n".join(
        f"(N'{r.week_start}', {r.diesel_price_usd_gal}, N'{r.series_id}', N'{r.source}')"
        for r in df.itertuples(index=False))
    sql = f"""/* EIA diesel load - generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC */
SET NOCOUNT ON;
GO
IF OBJECT_ID('freight.diesel_prices_eia') IS NULL
CREATE TABLE freight.diesel_prices_eia (
    week_start DATE NOT NULL PRIMARY KEY, diesel_price_usd_gal DECIMAL(6,3) NOT NULL,
    series_id VARCHAR(40) NOT NULL, source VARCHAR(40) NOT NULL);
GO
DELETE FROM freight.diesel_prices_eia;
GO
INSERT INTO freight.diesel_prices_eia (week_start, diesel_price_usd_gal, series_id, source) VALUES
{values};
GO
SELECT COUNT(*) AS weeks_loaded, MIN(week_start) AS first_week, MAX(week_start) AS latest_week
FROM freight.diesel_prices_eia;
SELECT TOP 4 week_start, diesel_price_usd_gal FROM freight.diesel_prices_eia ORDER BY week_start DESC;
GO
"""
    path = os.path.join(OUT_DIR, "diesel_load.sql")
    with open(path, "w", encoding="utf-8") as f:
        f.write(sql)

    print(f"\nSuccess: {len(df)} weeks, {df['week_start'].iloc[0]} to {df['week_start'].iloc[-1]}")
    print(f"Latest diesel price: ${df['diesel_price_usd_gal'].iloc[-1]:.3f}/gal")
    print(f"Now run {path} in VS Code.")


if __name__ == "__main__":
    main()
