"""
fetch_external_data.py
Predictive Freight Cost & Delay Agent - Phase 3: live external data (all free)

Self-contained (no imports from other project files). Requires: requests, pandas.
Run it on your own machine (VS Code terminal), in the same folder as ./data/lanes.csv.

APIs used - all free:
    1. OSRM public demo server  route distance, drive time, route geometry   no key   (fair use: ~1 request/sec)
    2. Open-Meteo forecast      weather along each route, past 7 + next 16 days   no key   (free for non-commercial use)
    3. Frankfurter (ECB rates)  USD -> CAD daily exchange rates since 2024    no key
    4. EIA Open Data v2         weekly US retail diesel price                 FREE key (optional)
       Get a key in ~1 minute at https://www.eia.gov/opendata/register.php,
       then set it:  Windows PowerShell:  $env:EIA_API_KEY="your_key"
                     macOS / Linux:       export EIA_API_KEY=your_key
       Without a key, the EIA step is skipped and everything else still runs.

Outputs (folder ./external/):
    lane_routes.csv         one row per lane: OSRM distance/duration and sampled route points
    route_conditions.csv    one row per lane x date: route-wide weather severity (0-3)
    fx_rates.csv            USD->CAD daily rates
    diesel_prices_eia.csv   weekly diesel prices (only if EIA key set)
    external_data_load.sql  run this in VS Code to load everything above into Azure SQL

Weather severity rule (worst point along the route on that day):
    3 severe   : snowfall >= 10 cm, or wind gusts >= 90 km/h, or heavy snow / freezing rain / thunderstorm codes
    2 moderate : snowfall >= 3 cm, or precipitation >= 20 mm, or gusts >= 65 km/h, or moderate snow/rain codes
    1 minor    : any snowfall, or precipitation >= 5 mm, or gusts >= 45 km/h
    0 none     : otherwise

Usage:
    python fetch_external_data.py
"""

import pandas, numpy, requests, sklearn, xgboost; print('All libraries OK')


import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

LANES_CSV = os.path.join("data", "lanes.csv")
OUT_DIR = "external"
ROUTE_SAMPLE_POINTS = 5
HTTP_TIMEOUT = 30
MAX_RETRIES = 3
USER_AGENT = "freight-delay-agent-portfolio/1.0"

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FRANKFURTER_URL = "https://api.frankfurter.dev/v1/{start}.."
EIA_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
EIA_DIESEL_SERIES = "EMD_EPD2D_PTE_NUS_DPG"   # U.S. No 2 diesel retail price, $/gal, weekly
HISTORY_START = "2024-01-01"

# WMO weather codes used by Open-Meteo
SEVERE_CODES = {65, 67, 75, 82, 86, 95, 96, 99}     # heavy rain/snow, freezing rain, thunderstorms
MODERATE_CODES = {63, 66, 73, 81, 85, 57}           # moderate rain/snow, freezing drizzle, showers


# ---------------------------------------------------------------------------
# HTTP helper with retries
# ---------------------------------------------------------------------------
def get_json(url, params=None, label=""):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT,
                                headers={"User-Agent": USER_AGENT})
            if resp.status_code == 429:
                time.sleep(5 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 * attempt)
    print(f"  ! {label} failed after {MAX_RETRIES} attempts: {last_error}")
    return None


# ---------------------------------------------------------------------------
# 1. OSRM routing
# ---------------------------------------------------------------------------
def sample_points(coords, n):
    """Pick n points evenly along a list of [lon, lat] coordinates (always includes both ends)."""
    if len(coords) <= n:
        return coords
    step = (len(coords) - 1) / (n - 1)
    return [coords[round(i * step)] for i in range(n)]


def fetch_routes(lanes):
    print("1/4  OSRM routing ...")
    rows = []
    for lane in lanes.itertuples(index=False):
        url = OSRM_URL.format(o_lon=lane.origin_lon, o_lat=lane.origin_lat,
                              d_lon=lane.dest_lon, d_lat=lane.dest_lat)
        data = get_json(url, params={"overview": "simplified", "geometries": "geojson"},
                        label=f"OSRM {lane.lane_id}")
        if data and data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            points = sample_points(route["geometry"]["coordinates"], ROUTE_SAMPLE_POINTS)
            rows.append({
                "lane_id": lane.lane_id,
                "route_distance_miles": round(route["distance"] / 1609.344, 1),
                "route_duration_hours": round(route["duration"] / 3600.0, 1),
                "sample_points_json": json.dumps([[round(p[1], 4), round(p[0], 4)] for p in points]),
                "source": "OSRM",
            })
        else:
            # Fallback: straight line between origin and destination, distance from lanes.csv
            pts = [[lane.origin_lat + (lane.dest_lat - lane.origin_lat) * i / (ROUTE_SAMPLE_POINTS - 1),
                    lane.origin_lon + (lane.dest_lon - lane.origin_lon) * i / (ROUTE_SAMPLE_POINTS - 1)]
                   for i in range(ROUTE_SAMPLE_POINTS)]
            rows.append({
                "lane_id": lane.lane_id,
                "route_distance_miles": float(lane.distance_miles),
                "route_duration_hours": None,
                "sample_points_json": json.dumps([[round(a, 4), round(b, 4)] for a, b in pts]),
                "source": "FALLBACK_STRAIGHT_LINE",
            })
        print(f"     {lane.lane_id}  {rows[-1]['route_distance_miles']} mi  ({rows[-1]['source']})")
        time.sleep(1.1)   # respect the public demo server's fair-use limit
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Open-Meteo weather along each route
# ---------------------------------------------------------------------------
def severity(code, precip_mm, snow_cm, gust_kmh):
    code = int(code) if pd.notna(code) else -1
    precip_mm = precip_mm if pd.notna(precip_mm) else 0.0
    snow_cm = snow_cm if pd.notna(snow_cm) else 0.0
    gust_kmh = gust_kmh if pd.notna(gust_kmh) else 0.0
    if snow_cm >= 10 or gust_kmh >= 90 or code in SEVERE_CODES:
        return 3
    if snow_cm >= 3 or precip_mm >= 20 or gust_kmh >= 65 or code in MODERATE_CODES:
        return 2
    if snow_cm > 0 or precip_mm >= 5 or gust_kmh >= 45:
        return 1
    return 0


def fetch_route_weather(routes):
    print("2/4  Open-Meteo weather along routes ...")
    as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    all_rows = []
    for r in routes.itertuples(index=False):
        points = json.loads(r.sample_points_json)
        params = {
            "latitude": ",".join(str(p[0]) for p in points),
            "longitude": ",".join(str(p[1]) for p in points),
            "daily": "weather_code,precipitation_sum,snowfall_sum,wind_gusts_10m_max",
            "past_days": 7,
            "forecast_days": 16,
            "timezone": "UTC",
        }
        data = get_json(OPEN_METEO_URL, params=params, label=f"Open-Meteo {r.lane_id}")
        if data is None:
            continue
        locations = data if isinstance(data, list) else [data]
        point_frames = []
        for loc in locations:
            daily = loc.get("daily", {})
            df = pd.DataFrame({
                "forecast_date": daily.get("time", []),
                "code": daily.get("weather_code", []),
                "precip": daily.get("precipitation_sum", []),
                "snow": daily.get("snowfall_sum", []),
                "gust": daily.get("wind_gusts_10m_max", []),
            })
            df["sev"] = [severity(*v) for v in df[["code", "precip", "snow", "gust"]].itertuples(index=False)]
            point_frames.append(df)
        if not point_frames:
            continue
        combined = pd.concat(point_frames)
        agg = combined.groupby("forecast_date").agg(
            weather_severity_forecast=("sev", "max"),
            precipitation_mm=("precip", "max"),
            snowfall_cm=("snow", "max"),
            wind_gust_kmh=("gust", "max"),
        ).reset_index()
        agg.insert(0, "lane_id", r.lane_id)
        agg["as_of_utc"] = as_of
        agg["route_distance_miles"] = r.route_distance_miles
        agg["route_duration_hours"] = r.route_duration_hours
        agg["source"] = "Open-Meteo + " + r.source
        all_rows.append(agg)
        worst = int(agg["weather_severity_forecast"].max())
        print(f"     {r.lane_id}  {len(agg)} days, worst severity {worst}")
        time.sleep(0.3)
    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    return out[["lane_id", "as_of_utc", "forecast_date", "weather_severity_forecast",
                "precipitation_mm", "snowfall_cm", "wind_gust_kmh",
                "route_distance_miles", "route_duration_hours", "source"]]


# ---------------------------------------------------------------------------
# 3. Frankfurter exchange rates
# ---------------------------------------------------------------------------
def fetch_fx():
    print("3/4  Frankfurter USD->CAD ...")
    data = get_json(FRANKFURTER_URL.format(start=HISTORY_START),
                    params={"base": "USD", "symbols": "CAD"}, label="Frankfurter")
    if not data or "rates" not in data:
        return pd.DataFrame()
    rows = [{"rate_date": d, "base_currency": "USD", "quote_currency": "CAD",
             "rate": v["CAD"], "source": "Frankfurter/ECB"}
            for d, v in sorted(data["rates"].items()) if "CAD" in v]
    print(f"     {len(rows)} daily rates, latest {rows[-1]['rate_date']} = {rows[-1]['rate']}" if rows else "     none")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. EIA diesel prices (optional, free key)
# ---------------------------------------------------------------------------
def fetch_eia_diesel():
    key = os.environ.get("BkMfzBMTJmgThT79kD77FPfzbgnkkD6fMzmMdBhz", "").strip()
    if not key:
        print("4/4  EIA diesel ... skipped (set EIA_API_KEY to enable; free registration)")
        return pd.DataFrame()
    print("4/4  EIA weekly diesel ...")
    params = {
        "api_key": key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": EIA_DIESEL_SERIES,
        "start": HISTORY_START,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }
    data = get_json(EIA_URL, params=params, label="EIA")
    records = (data or {}).get("response", {}).get("data", [])
    rows = [{"week_start": r["period"], "diesel_price_usd_gal": float(r["value"]),
             "series_id": EIA_DIESEL_SERIES, "source": "EIA"}
            for r in records if r.get("value") not in (None, "")]
    if rows:
        print(f"     {len(rows)} weeks, latest {rows[-1]['week_start']} = ${rows[-1]['diesel_price_usd_gal']:.3f}")
    else:
        print("     no rows returned - check the key")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SQL script writer
# ---------------------------------------------------------------------------
def sql_literal(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "NULL"
    return "N'" + str(v).replace("'", "''") + "'"


def insert_blocks(table, df, batch=250):
    if df.empty:
        return [f"/* {table}: no rows fetched */"]
    cols = ", ".join(df.columns)
    rows = ["(" + ", ".join(sql_literal(v) for v in rec) + ")"
            for rec in df.astype(object).itertuples(index=False, name=None)]
    blocks = []
    for i in range(0, len(rows), batch):
        blocks += [f"INSERT INTO {table} ({cols}) VALUES\n" + ",\n".join(rows[i:i + batch]) + ";", "GO"]
    return blocks


def write_sql(conditions, fx, diesel, path):
    lines = [
        "/* External data load - generated by fetch_external_data.py",
        f"   Generated (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}",
        "   Replaces previous external rows; does not touch shipments. */",
        "SET NOCOUNT ON;", "GO",
        """IF OBJECT_ID('freight.fx_rates') IS NULL
CREATE TABLE freight.fx_rates (
    rate_date DATE NOT NULL, base_currency CHAR(3) NOT NULL, quote_currency CHAR(3) NOT NULL,
    rate DECIMAL(12,6) NOT NULL, source VARCHAR(40) NOT NULL,
    CONSTRAINT pk_fx_rates PRIMARY KEY (rate_date, base_currency, quote_currency));""", "GO",
        """IF OBJECT_ID('freight.diesel_prices_eia') IS NULL
CREATE TABLE freight.diesel_prices_eia (
    week_start DATE NOT NULL PRIMARY KEY, diesel_price_usd_gal DECIMAL(6,3) NOT NULL,
    series_id VARCHAR(40) NOT NULL, source VARCHAR(40) NOT NULL);""", "GO",
    ]
    if not conditions.empty:
        lines += ["DELETE FROM freight.route_conditions;", "GO"]
        lines += insert_blocks("freight.route_conditions", conditions)
    if not fx.empty:
        lines += ["DELETE FROM freight.fx_rates;", "GO"]
        lines += insert_blocks("freight.fx_rates", fx)
    if not diesel.empty:
        lines += ["DELETE FROM freight.diesel_prices_eia;", "GO"]
        lines += insert_blocks("freight.diesel_prices_eia", diesel)
    lines += [
        "/* Validation */",
        "SELECT 'route_conditions' AS table_name, COUNT(*) AS row_count FROM freight.route_conditions "
        "UNION ALL SELECT 'fx_rates', COUNT(*) FROM freight.fx_rates "
        "UNION ALL SELECT 'diesel_prices_eia', COUNT(*) FROM freight.diesel_prices_eia;",
        "SELECT TOP 10 lane_id, forecast_date, weather_severity_forecast, snowfall_cm, wind_gust_kmh "
        "FROM freight.route_conditions WHERE weather_severity_forecast >= 2 "
        "ORDER BY weather_severity_forecast DESC, forecast_date;",
        "GO",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(LANES_CSV):
        sys.exit(f"Cannot find {LANES_CSV}. Run this from the project folder that contains ./data/.")
    lanes = pd.read_csv(LANES_CSV)
    os.makedirs(OUT_DIR, exist_ok=True)

    routes = fetch_routes(lanes)
    routes.to_csv(os.path.join(OUT_DIR, "lane_routes.csv"), index=False)

    conditions = fetch_route_weather(routes)
    fx = fetch_fx()
    diesel = fetch_eia_diesel()

    conditions.to_csv(os.path.join(OUT_DIR, "route_conditions.csv"), index=False)
    fx.to_csv(os.path.join(OUT_DIR, "fx_rates.csv"), index=False)
    if not diesel.empty:
        diesel.to_csv(os.path.join(OUT_DIR, "diesel_prices_eia.csv"), index=False)

    sql_path = os.path.join(OUT_DIR, "external_data_load.sql")
    write_sql(conditions, fx, diesel, sql_path)

    print("\nDone.")
    print(f"  route_conditions: {len(conditions)} rows")
    print(f"  fx_rates:         {len(fx)} rows")
    print(f"  diesel (EIA):     {len(diesel)} rows")
    print(f"  SQL to run in VS Code: {sql_path}")


if __name__ == "__main__":
    main()
