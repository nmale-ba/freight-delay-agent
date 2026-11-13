"""
load_to_azure_sql.py
Predictive Freight Cost & Delay Agent - Phase 1: create schema and load data

Self-contained (no imports from other project files). Requires: pandas, pymssql.
pymssql is used instead of pyodbc so NO ODBC driver install is needed - it works
in Azure Cloud Shell after a plain `pip install --user pymssql`.

What it does:
    1. Creates schema [freight] in your existing Azure SQL database (no new Azure resource).
    2. Drops and recreates the tables (safe to re-run).
    3. Loads ./data/*.csv produced by generate_freight_data.py.
    4. Creates two views the agent tools and Power BI will use.
    5. Prints validation counts.

Connection settings come from environment variables:
    SQL_SERVER    e.g. sql-digitalbanking-nmale        (".database.windows.net" optional)
    SQL_DATABASE  your database name
    SQL_USER      SQL admin login
    SQL_PASSWORD  SQL admin password

Usage:
    python load_to_azure_sql.py --dry-run     # validate CSVs + print DDL, no connection
    python load_to_azure_sql.py               # create and load
"""

import argparse
import math
import os
import sys

import pandas as pd

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# DDL - each list item is executed as its own batch
# ---------------------------------------------------------------------------
DROP_STATEMENTS = [
    "DROP VIEW IF EXISTS freight.v_active_shipments",
    "DROP VIEW IF EXISTS freight.v_carrier_lane_performance",
    "DROP TABLE IF EXISTS freight.agent_tool_calls",
    "DROP TABLE IF EXISTS freight.agent_runs",
    "DROP TABLE IF EXISTS freight.shipment_predictions",
    "DROP TABLE IF EXISTS freight.route_conditions",
    "DROP TABLE IF EXISTS freight.fx_rates",
    "DROP TABLE IF EXISTS freight.diesel_prices_eia",
    "DROP TABLE IF EXISTS freight.shipments",
    "DROP TABLE IF EXISTS freight.fuel_prices",
    "DROP TABLE IF EXISTS freight.lanes",
    "DROP TABLE IF EXISTS freight.carriers",
]

CREATE_STATEMENTS = [
    "IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'freight') EXEC('CREATE SCHEMA freight')",

    """CREATE TABLE freight.carriers (
        carrier_id            VARCHAR(10)   NOT NULL PRIMARY KEY,
        carrier_name          NVARCHAR(100) NOT NULL,
        mode                  VARCHAR(20)   NOT NULL,
        country               CHAR(2)       NOT NULL,
        quote_currency        CHAR(3)       NOT NULL,
        base_rate_per_mile    DECIMAL(6,2)  NOT NULL,
        reliability_baseline  DECIMAL(4,3)  NOT NULL,
        data_source           VARCHAR(20)   NOT NULL
    )""",

    """CREATE TABLE freight.lanes (
        lane_id            VARCHAR(10)   NOT NULL PRIMARY KEY,
        lane_name          NVARCHAR(120) NOT NULL,
        origin_city        NVARCHAR(60)  NOT NULL,
        origin_state       VARCHAR(5)    NOT NULL,
        origin_lat         DECIMAL(8,4)  NOT NULL,
        origin_lon         DECIMAL(8,4)  NOT NULL,
        dest_city          NVARCHAR(60)  NOT NULL,
        dest_state         VARCHAR(5)    NOT NULL,
        dest_country       CHAR(2)       NOT NULL,
        dest_lat           DECIMAL(8,4)  NOT NULL,
        dest_lon           DECIMAL(8,4)  NOT NULL,
        distance_miles     INT           NOT NULL,
        is_port_origin     BIT           NOT NULL,
        is_winter_exposed  BIT           NOT NULL,
        is_cross_border    BIT           NOT NULL
    )""",

    """CREATE TABLE freight.fuel_prices (
        week_start            DATE         NOT NULL PRIMARY KEY,
        diesel_price_usd_gal  DECIMAL(6,3) NOT NULL,
        data_source           VARCHAR(20)  NOT NULL
    )""",

    """CREATE TABLE freight.shipments (
        shipment_id            VARCHAR(12)   NOT NULL PRIMARY KEY,
        lane_id                VARCHAR(10)   NOT NULL REFERENCES freight.lanes(lane_id),
        carrier_id             VARCHAR(10)   NOT NULL REFERENCES freight.carriers(carrier_id),
        equipment_type         VARCHAR(30)   NOT NULL,
        weight_lbs             INT           NOT NULL,
        status                 VARCHAR(20)   NOT NULL,
        ship_date              DATE          NOT NULL,
        planned_transit_days   INT           NOT NULL,
        planned_delivery_date  DATE          NOT NULL,
        actual_delivery_date   DATE          NULL,
        actual_transit_days    INT           NULL,
        delay_days             INT           NULL,
        is_delayed             BIT           NULL,
        diesel_price_usd_gal   DECIMAL(6,3)  NOT NULL,
        weather_severity       TINYINT       NULL,
        port_congestion_index  DECIMAL(4,3)  NOT NULL,
        quote_currency         CHAR(3)       NOT NULL,
        fx_rate_to_usd         DECIMAL(10,5) NOT NULL,
        quoted_cost_native     DECIMAL(12,2) NOT NULL,
        quoted_cost_usd        DECIMAL(12,2) NOT NULL,
        accessorial_cost_usd   DECIMAL(12,2) NULL,
        actual_cost_usd        DECIMAL(12,2) NULL,
        data_source            VARCHAR(20)   NOT NULL,
        CONSTRAINT ck_status CHECK (status IN ('Booked', 'In Transit', 'Delivered')),
        CONSTRAINT ck_weather CHECK (weather_severity BETWEEN 0 AND 3)
    )""",
    "CREATE INDEX ix_shipments_status ON freight.shipments(status, planned_delivery_date)",
    "CREATE INDEX ix_shipments_carrier_lane ON freight.shipments(carrier_id, lane_id)",

    # --- Populated in Phase 3 (live weather / routing per lane) ---
    """CREATE TABLE freight.route_conditions (
        condition_id               INT IDENTITY(1,1) PRIMARY KEY,
        lane_id                    VARCHAR(10)  NOT NULL REFERENCES freight.lanes(lane_id),
        as_of_utc                  DATETIME2    NOT NULL,
        forecast_date              DATE         NOT NULL,
        weather_severity_forecast  TINYINT      NULL,
        precipitation_mm           DECIMAL(6,2) NULL,
        snowfall_cm                DECIMAL(6,2) NULL,
        wind_gust_kmh              DECIMAL(6,2) NULL,
        route_distance_miles       DECIMAL(8,1) NULL,
        route_duration_hours       DECIMAL(6,1) NULL,
        source                     VARCHAR(40)  NOT NULL
    )""",

    # --- Populated in Phase 3 (Frankfurter API: USD -> CAD daily rates) ---
    """CREATE TABLE freight.fx_rates (
        rate_date       DATE          NOT NULL,
        base_currency   CHAR(3)       NOT NULL,
        quote_currency  CHAR(3)       NOT NULL,
        rate            DECIMAL(12,6) NOT NULL,
        source          VARCHAR(40)   NOT NULL,
        CONSTRAINT pk_fx_rates PRIMARY KEY (rate_date, base_currency, quote_currency)
    )""",

    # --- Populated in Phase 3 (EIA API: weekly US retail diesel, live market reference) ---
    """CREATE TABLE freight.diesel_prices_eia (
        week_start            DATE         NOT NULL PRIMARY KEY,
        diesel_price_usd_gal  DECIMAL(6,3) NOT NULL,
        series_id             VARCHAR(40)  NOT NULL,
        source                VARCHAR(40)  NOT NULL
    )""",

    # --- Populated in Phase 2 (prediction engine output) ---
    """CREATE TABLE freight.shipment_predictions (
        prediction_id              INT IDENTITY(1,1) PRIMARY KEY,
        shipment_id                VARCHAR(12)   NOT NULL REFERENCES freight.shipments(shipment_id),
        predicted_at_utc           DATETIME2     NOT NULL,
        model_version              VARCHAR(30)   NOT NULL,
        delay_probability          DECIMAL(5,4)  NOT NULL,
        expected_delay_days        DECIMAL(5,2)  NULL,
        predicted_cost_usd         DECIMAL(12,2) NULL,
        expected_exposure_usd      DECIMAL(12,2) NULL,
        risk_tier                  VARCHAR(10)   NOT NULL,
        recommended_action         NVARCHAR(400) NULL
    )""",

    # --- Populated in Phase 4/5 (agent logging = the evaluation evidence) ---
    """CREATE TABLE freight.agent_runs (
        run_id           VARCHAR(40)    NOT NULL PRIMARY KEY,
        started_at_utc   DATETIME2      NOT NULL,
        scenario         VARCHAR(40)    NULL,
        user_question    NVARCHAR(1000) NOT NULL,
        turns_used       INT            NULL,
        hit_turn_cap     BIT            NULL,
        completed        BIT            NULL,
        total_tokens     INT            NULL,
        latency_ms       INT            NULL,
        final_answer     NVARCHAR(MAX)  NULL
    )""",
    """CREATE TABLE freight.agent_tool_calls (
        call_id          INT IDENTITY(1,1) PRIMARY KEY,
        run_id           VARCHAR(40)    NOT NULL REFERENCES freight.agent_runs(run_id),
        turn_number      INT            NOT NULL,
        tool_name        VARCHAR(60)    NOT NULL,
        arguments_json   NVARCHAR(MAX)  NULL,
        succeeded        BIT            NOT NULL,
        error_message    NVARCHAR(1000) NULL,
        latency_ms       INT            NULL
    )""",

    # --- Views used by agent tools and Power BI ---
    """CREATE VIEW freight.v_carrier_lane_performance AS
    SELECT  s.carrier_id,
            c.carrier_name,
            s.lane_id,
            l.lane_name,
            COUNT(*)                                             AS shipments,
            CAST(1.0 - AVG(CAST(s.is_delayed AS FLOAT)) AS DECIMAL(5,4)) AS on_time_rate,
            CAST(AVG(CAST(s.delay_days AS FLOAT)) AS DECIMAL(5,2))       AS avg_delay_days,
            CAST(AVG(s.actual_cost_usd) AS DECIMAL(12,2))                AS avg_actual_cost_usd,
            CAST(AVG(s.actual_cost_usd - s.quoted_cost_usd) AS DECIMAL(12,2)) AS avg_cost_overrun_usd
    FROM    freight.shipments s
    JOIN    freight.carriers  c ON c.carrier_id = s.carrier_id
    JOIN    freight.lanes     l ON l.lane_id    = s.lane_id
    WHERE   s.status = 'Delivered'
    GROUP BY s.carrier_id, c.carrier_name, s.lane_id, l.lane_name""",

    """CREATE VIEW freight.v_active_shipments AS
    SELECT  s.shipment_id, s.status, s.ship_date, s.planned_delivery_date,
            s.lane_id, l.lane_name, l.distance_miles, l.is_port_origin, l.is_cross_border,
            l.origin_lat, l.origin_lon, l.dest_lat, l.dest_lon,
            s.carrier_id, c.carrier_name, s.equipment_type, s.weight_lbs,
            s.diesel_price_usd_gal, s.port_congestion_index,
            s.quote_currency, s.quoted_cost_native, s.quoted_cost_usd
    FROM    freight.shipments s
    JOIN    freight.carriers  c ON c.carrier_id = s.carrier_id
    JOIN    freight.lanes     l ON l.lane_id    = s.lane_id
    WHERE   s.status IN ('Booked', 'In Transit')""",
]

# Table load order respects foreign keys
LOAD_PLAN = [
    ("freight.carriers", "carriers.csv"),
    ("freight.lanes", "lanes.csv"),
    ("freight.fuel_prices", "fuel_prices.csv"),
    ("freight.shipments", "shipments.csv"),
]

EXPECTED_MIN_ROWS = {"carriers.csv": 6, "lanes.csv": 12, "fuel_prices.csv": 100, "shipments.csv": 1000}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_value(v):
    """Convert pandas NaN/NA to None so SQL receives NULL; numpy scalars to Python."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def read_csvs():
    frames = {}
    for _, filename in LOAD_PLAN:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run generate_freight_data.py first.")
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        if len(df) < EXPECTED_MIN_ROWS[filename]:
            sys.exit(f"{filename} has only {len(df)} rows - regenerate the data.")
        frames[filename] = df
    return frames


def get_connection():
    import pymssql  # imported here so --dry-run works without it installed

    server = os.environ.get("SQL_SERVER", "").strip()
    database = os.environ.get("SQL_DATABASE", "").strip()
    user = os.environ.get("SQL_USER", "").strip()
    password = os.environ.get("SQL_PASSWORD", "")
    missing = [n for n, v in [("SQL_SERVER", server), ("SQL_DATABASE", database),
                              ("SQL_USER", user), ("SQL_PASSWORD", password)] if not v]
    if missing:
        sys.exit(f"Set these environment variables first: {', '.join(missing)}")
    if not server.endswith(".database.windows.net"):
        server = f"{server}.database.windows.net"

    print(f"Connecting to {server} / {database} as {user} ...")
    return pymssql.connect(server=server, user=user, password=password,
                           database=database, port=1433, tds_version="7.4",
                           login_timeout=60, timeout=120)


def insert_frame(cursor, table, df, batch_size=500):
    cols = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(clean_value(v) for v in row) for row in df.itertuples(index=False, name=None)]
    for start in range(0, len(rows), batch_size):
        cursor.executemany(sql, rows[start:start + batch_size])
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create freight schema and load CSVs into Azure SQL.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSVs and print DDL without connecting")
    args = parser.parse_args()

    frames = read_csvs()
    print("CSV check:")
    for filename, df in frames.items():
        print(f"  {filename:<16} {len(df):>5} rows, {len(df.columns)} columns")

    if args.dry_run:
        print("\n-- DDL that would run --")
        for stmt in DROP_STATEMENTS + CREATE_STATEMENTS:
            print(stmt.strip() + "\n;")
        print("\nDry run complete. No connection was made.")
        return

    conn = get_connection()
    cursor = conn.cursor()
    try:
        print("Dropping old objects (if any) ...")
        cursor.execute("IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'freight') "
                       "EXEC('CREATE SCHEMA freight')")
        for stmt in DROP_STATEMENTS:
            cursor.execute(stmt)
        print("Creating tables and views ...")
        for stmt in CREATE_STATEMENTS:
            cursor.execute(stmt)
        conn.commit()

        for table, filename in LOAD_PLAN:
            n = insert_frame(cursor, table, frames[filename])
            conn.commit()
            print(f"  loaded {n:>5} rows -> {table}")

        print("\nValidation:")
        checks = [
            ("Shipments by status",
             "SELECT status, COUNT(*) FROM freight.shipments GROUP BY status ORDER BY status"),
            ("Historical delay rate",
             "SELECT CAST(AVG(CAST(is_delayed AS FLOAT)) AS DECIMAL(5,3)) "
             "FROM freight.shipments WHERE status = 'Delivered'"),
            ("Active shipments already past planned delivery",
             "SELECT COUNT(*) FROM freight.v_active_shipments "
             "WHERE status = 'In Transit' AND planned_delivery_date < '2026-11-07'"),
            ("Worst carrier-lane pairs (min 15 shipments)",
             "SELECT TOP 3 carrier_name, lane_name, shipments, on_time_rate "
             "FROM freight.v_carrier_lane_performance WHERE shipments >= 15 "
             "ORDER BY on_time_rate ASC"),
        ]
        for label, sql in checks:
            cursor.execute(sql)
            print(f"  {label}:")
            for row in cursor.fetchall():
                print("     ", " | ".join(str(x) for x in row))
        print("\nDone. Phase 1 database is ready.")
    except Exception as exc:
        conn.rollback()
        sys.exit(f"Load failed: {exc}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
