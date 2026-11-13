"""
freight_agent.py
Predictive Freight Cost & Delay Agent - Phase 4: the AI agent

Self-contained (no imports from other project files). Reads the saved models in ./models/ as data.
Requires:  pip install openai pymssql pandas numpy xgboost

The agent answers logistics questions by deciding which of four tools to call:
    1. get_shipment_context              active shipments, dates, status, overdue days      (Azure SQL)
    2. get_route_conditions              weather + routing along each lane                   (Azure SQL <- Open-Meteo/OSRM)
    3. get_carrier_performance           historical on-time rates by carrier/lane + diesel/FX (Azure SQL <- EIA/Frankfurter)
    4. calculate_risk_and_recommendation delay probability, exposure, carrier switches      (XGBoost models + rules)

The LLM never calculates risk or money itself: every number comes from a tool.

Guardrails:
    MAX_TURNS = 6           model calls per question; the 6th call is forced to answer (no more tools)
    MAX_TOKENS_PER_RUN      stops tool use and forces an answer if one question gets expensive
    DAILY_TOKEN_BUDGET      refuses to run once today's logged usage passes the cap (protects your card)
    SQL query timeout, input validation on every tool argument, retry limits on the model API,
    and every run + tool call logged to freight.agent_runs / freight.agent_tool_calls and logs/*.jsonl

Environment variables:
    Database (SQL Login):  SQL_SERVER, SQL_DATABASE, SQL_USER, SQL_PASSWORD
    Model - choose one with LLM_PROVIDER:
      LLM_PROVIDER=azure  (default)  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
                                     optional AZURE_OPENAI_API_VERSION (default 2024-10-21)
      LLM_PROVIDER=github (free)     GITHUB_TOKEN, optional GITHUB_MODEL (default openai/gpt-4.1-mini)

Usage:
    python freight_agent.py --check                         # test DB, models and all 4 tools - no model call, no cost
    python freight_agent.py "Which shipments are at high risk of delay this week, and what is the financial exposure?"
    python freight_agent.py --scenario B "Which shipments face severe weather on their route?"
    python freight_agent.py                                 # interactive mode (remembers the conversation)
    python freight_agent.py --tool get_carrier_performance --args '{"carrier_ids": ["C04"]}'   # one tool, no model
"""

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Guardrails and settings
# ---------------------------------------------------------------------------
MAX_TURNS = 6
MAX_TOKENS_PER_RUN = 40_000
DAILY_TOKEN_BUDGET = 400_000          # roughly $0.10-$0.30/day on a "mini" model; raise deliberately
MAX_OUTPUT_TOKENS = 1400            # room for report-style answers with tables
MAX_HISTORY_MESSAGES = 8           # earlier user/assistant messages kept for follow-up questions
SQL_TIMEOUT_SECONDS = 30
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 2
MAX_TOOL_RESULT_CHARS = 6_000         # keeps each tool result small (fits free-tier input limits too)
MODEL_DIR = "models"
LOG_DIR = "logs"

# Fuel surcharge convention used throughout the project
FSC_BASE_PRICE, FSC_MPG = 1.25, 6.5

AS_OF = date.today()                  # overridable with --as-of for reproducible evaluation runs


# ---------------------------------------------------------------------------
# Database helpers (pymssql - no ODBC driver needed)
# ---------------------------------------------------------------------------
_CONN = None
PARAM = "%s"                          # pymssql placeholder


def get_connection():
    global _CONN
    if _CONN is None:
        import pymssql
        server = os.environ.get("SQL_SERVER", "").strip()
        missing = [k for k in ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD") if not os.environ.get(k)]
        if missing:
            sys.exit(f"Set these environment variables first: {', '.join(missing)}")
        if not server.endswith(".database.windows.net"):
            server += ".database.windows.net"
        _CONN = pymssql.connect(server=server, user=os.environ["SQL_USER"], password=os.environ["SQL_PASSWORD"],
                                database=os.environ["SQL_DATABASE"], port=1433, tds_version="7.4",
                                login_timeout=30, timeout=SQL_TIMEOUT_SECONDS)
    return _CONN


def _clean_value(v):
    if isinstance(v, Decimal):
        return float(v)
    return v


def run_query(sql, params=()):
    cur = get_connection().cursor()
    cur.execute(sql, tuple(params))
    cols = [c[0] for c in cur.description] if cur.description else []
    rows = [[_clean_value(v) for v in r] for r in cur.fetchall()]
    return pd.DataFrame(rows, columns=cols)


def run_exec(sql, params=()):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    conn.commit()


def placeholders(n):
    return ", ".join([PARAM] * n)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
ID_PATTERNS = {"shipment_ids": r"^SHP-\d{6}$", "lane_ids": r"^L\d{2}$", "carrier_ids": r"^C\d{2}$"}


class ToolInputError(ValueError):
    pass


def v_list(args, key, max_len=100):
    value = args.get(key)
    if value in (None, [], ""):
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or len(value) > max_len:
        raise ToolInputError(f"{key} must be a list of at most {max_len} IDs")
    cleaned = [str(x).strip().upper() for x in value]
    bad = [x for x in cleaned if not re.match(ID_PATTERNS[key], x)]
    if bad:
        raise ToolInputError(f"invalid {key}: {bad[:5]} (expected format like "
                             f"{ {'shipment_ids': 'SHP-001234', 'lane_ids': 'L01', 'carrier_ids': 'C04'}[key] })")
    return cleaned


def v_date(args, key, default):
    value = args.get(key)
    if value in (None, ""):
        return default
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        raise ToolInputError(f"{key} must be a date in YYYY-MM-DD format")


def v_int(args, key, default, lo, hi):
    value = args.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{key} must be an integer")
    return max(lo, min(hi, value))


def v_number(args, key, default, lo, hi):
    value = args.get(key, default)
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{key} must be a number")
    if not lo <= value <= hi:
        raise ToolInputError(f"{key} must be between {lo} and {hi}")
    return value


def to_records(df, cols):
    out = []
    for rec in df[cols].to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            if isinstance(v, (pd.Timestamp, datetime, date)):
                v = pd.Timestamp(v).strftime("%Y-%m-%d")
            elif isinstance(v, (np.floating, float)):
                v = round(float(v), 4)
            elif isinstance(v, np.integer):
                v = int(v)
            clean[k] = v
        out.append(clean)
    return out


def fit_to_budget(result, list_key):
    """Shrink a list inside the result until the JSON fits MAX_TOOL_RESULT_CHARS."""
    text = json.dumps(result, default=str)
    while len(text) > MAX_TOOL_RESULT_CHARS and result.get(list_key):
        keep = max(1, len(result[list_key]) // 2)
        if keep == len(result[list_key]):
            break
        result[list_key] = result[list_key][:keep]
        result["truncated"] = f"{list_key} shortened to {keep} rows to fit size limit; narrow the filters for more"
        text = json.dumps(result, default=str)
    return result


# ---------------------------------------------------------------------------
# Tool 1: get_shipment_context
# ---------------------------------------------------------------------------
def get_shipment_context(args):
    status = args.get("status", "active")
    if status not in ("active", "Booked", "In Transit"):
        raise ToolInputError("status must be 'active', 'Booked' or 'In Transit'")
    d_from = v_date(args, "delivery_from", None)
    d_to = v_date(args, "delivery_to", None)
    lane_ids, carrier_ids = v_list(args, "lane_ids"), v_list(args, "carrier_ids")
    shipment_ids = v_list(args, "shipment_ids", 200)
    overdue_only = bool(args.get("overdue_only", False))
    limit = v_int(args, "limit", 20, 1, 50)

    sql = ("SELECT s.shipment_id, s.status, s.ship_date, s.planned_delivery_date, s.lane_id, l.lane_name, "
           "s.carrier_id, c.carrier_name, s.equipment_type, s.weight_lbs, s.quoted_cost_usd "
           "FROM freight.shipments s JOIN freight.lanes l ON l.lane_id = s.lane_id "
           "JOIN freight.carriers c ON c.carrier_id = s.carrier_id ")
    where, params = [], []
    if status == "active":
        where.append("s.status IN ('Booked', 'In Transit')")
    else:
        where.append(f"s.status = {PARAM}")
        params.append(status)
    if d_from:
        where.append(f"s.planned_delivery_date >= {PARAM}")
        params.append(d_from.isoformat())
    if d_to:
        where.append(f"s.planned_delivery_date <= {PARAM}")
        params.append(d_to.isoformat())
    for key, col, values in (("lane_ids", "s.lane_id", lane_ids), ("carrier_ids", "s.carrier_id", carrier_ids),
                             ("shipment_ids", "s.shipment_id", shipment_ids)):
        if values:
            where.append(f"{col} IN ({placeholders(len(values))})")
            params += values
    df = run_query(sql + "WHERE " + " AND ".join(where), params)
    if df.empty:
        return {"as_of": AS_OF.isoformat(), "total_matching": 0, "shipments": [],
                "note": "No shipments match these filters."}

    df["planned_delivery_date"] = pd.to_datetime(df["planned_delivery_date"])
    df["ship_date"] = pd.to_datetime(df["ship_date"])
    df["days_overdue"] = np.where((df["status"] == "In Transit") &
                                  (df["planned_delivery_date"] < pd.Timestamp(AS_OF)),
                                  (pd.Timestamp(AS_OF) - df["planned_delivery_date"]).dt.days, 0)
    if overdue_only:
        df = df[df["days_overdue"] > 0]
    df = df.sort_values(["days_overdue", "planned_delivery_date"], ascending=[False, True])

    by_carrier = []
    for (cid, cname), sub in df.groupby(["carrier_id", "carrier_name"]):
        by_carrier.append({"carrier_id": cid, "carrier_name": cname, "total": int(len(sub)),
                           "booked": int((sub["status"] == "Booked").sum()),
                           "in_transit": int((sub["status"] == "In Transit").sum()),
                           "overdue": int((sub["days_overdue"] > 0).sum())})
    by_carrier.sort(key=lambda x: -x["total"])
    result = {
        "as_of": AS_OF.isoformat(),
        "total_matching": int(len(df)),
        "by_status": {k: int(v) for k, v in df["status"].value_counts().items()},
        "by_carrier": by_carrier,
        "overdue_count": int((df["days_overdue"] > 0).sum()),
        "total_quoted_cost_usd": round(float(df["quoted_cost_usd"].sum()), 2),
        "lanes_involved": sorted(df["lane_id"].unique().tolist()),
        "carriers_involved": sorted(df["carrier_id"].unique().tolist()),
        "shipments_returned": min(limit, len(df)),
        "shipments": to_records(df.head(limit), ["shipment_id", "status", "ship_date", "planned_delivery_date",
                                                 "days_overdue", "lane_id", "lane_name", "carrier_id",
                                                 "carrier_name", "equipment_type", "quoted_cost_usd"]),
    }
    return fit_to_budget(result, "shipments")


# ---------------------------------------------------------------------------
# Tool 2: get_route_conditions
# ---------------------------------------------------------------------------
SEVERITY_LABEL = {0: "none", 1: "minor", 2: "moderate", 3: "severe"}


def get_route_conditions(args):
    lane_ids = v_list(args, "lane_ids", 12)
    if not lane_ids:
        raise ToolInputError("lane_ids is required (e.g. ['L01', 'L04'])")
    start = v_date(args, "start_date", AS_OF)
    end = v_date(args, "end_date", AS_OF + timedelta(days=7))
    if end < start or (end - start).days > 31:
        raise ToolInputError("end_date must be on/after start_date and within 31 days of it")

    lanes = run_query(f"SELECT lane_id, lane_name, is_port_origin, is_winter_exposed FROM freight.lanes "
                      f"WHERE lane_id IN ({placeholders(len(lane_ids))})", lane_ids)
    rc = run_query("SELECT lane_id, forecast_date, weather_severity_forecast, precipitation_mm, snowfall_cm, "
                   "wind_gust_kmh, route_distance_miles, route_duration_hours, as_of_utc, source "
                   f"FROM freight.route_conditions WHERE lane_id IN ({placeholders(len(lane_ids))}) "
                   f"AND forecast_date BETWEEN {PARAM} AND {PARAM}",
                   lane_ids + [start.isoformat(), end.isoformat()])
    congestion = run_query("SELECT lane_id, AVG(CAST(port_congestion_index AS FLOAT)) AS port_congestion_index "
                           f"FROM freight.shipments WHERE status IN ('Booked', 'In Transit') "
                           f"AND lane_id IN ({placeholders(len(lane_ids))}) GROUP BY lane_id", lane_ids)
    cong = dict(zip(congestion["lane_id"], congestion["port_congestion_index"])) if not congestion.empty else {}

    out = []
    for lane in lanes.itertuples(index=False):
        sub = rc[rc["lane_id"] == lane.lane_id] if not rc.empty else rc
        item = {"lane_id": lane.lane_id, "lane_name": lane.lane_name}
        if sub is None or sub.empty:
            item["forecast"] = "no forecast rows in this window (run fetch_external_data.py to refresh)"
        else:
            sub = sub.sort_values("forecast_date")
            worst = int(sub["weather_severity_forecast"].max())
            worst_days = sub[sub["weather_severity_forecast"] == worst]["forecast_date"]
            item.update({
                "worst_weather_severity": worst,
                "worst_weather_label": SEVERITY_LABEL.get(worst, "unknown"),
                "worst_weather_dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in worst_days][:5],
                "days_moderate_or_worse": int((sub["weather_severity_forecast"] >= 2).sum()),
                "max_snowfall_cm": round(float(sub["snowfall_cm"].max() or 0), 1),
                "max_wind_gust_kmh": round(float(sub["wind_gust_kmh"].max() or 0), 1),
                "route_distance_miles": None if pd.isna(sub["route_distance_miles"].iloc[0])
                else round(float(sub["route_distance_miles"].iloc[0]), 0),
                "route_drive_hours": None if pd.isna(sub["route_duration_hours"].iloc[0])
                else round(float(sub["route_duration_hours"].iloc[0]), 1),
                "forecast_as_of_utc": str(sub["as_of_utc"].max())[:16],
            })
        if int(lane.is_port_origin):
            item["port_congestion_index"] = round(float(cong.get(lane.lane_id, 0.0)), 3)
            item["port_congestion_note"] = "0-1 index, SIMULATED (no free live port API)"
        item["winter_exposed_route"] = bool(int(lane.is_winter_exposed))
        out.append(item)
    return fit_to_budget({"window": f"{start} to {end}", "severity_scale": "0 none, 1 minor, 2 moderate, 3 severe",
                          "lanes": out}, "lanes")


# ---------------------------------------------------------------------------
# Tool 3: get_carrier_performance
# ---------------------------------------------------------------------------
def get_carrier_performance(args):
    carrier_ids, lane_ids = v_list(args, "carrier_ids"), v_list(args, "lane_ids")
    min_shipments = v_int(args, "min_shipments", 5, 1, 100)

    sql = ("SELECT s.carrier_id, c.carrier_name, c.mode, COUNT(*) AS shipments, "
           "AVG(CAST(s.is_delayed AS FLOAT)) AS delay_rate, AVG(CAST(s.delay_days AS FLOAT)) AS avg_delay_days, "
           "AVG(CAST(s.actual_cost_usd - s.quoted_cost_usd AS FLOAT)) AS avg_cost_overrun_usd "
           "FROM freight.shipments s JOIN freight.carriers c ON c.carrier_id = s.carrier_id "
           "WHERE s.status = 'Delivered' ")
    params = []
    if carrier_ids:
        sql += f"AND s.carrier_id IN ({placeholders(len(carrier_ids))}) "
        params += carrier_ids
    overall = run_query(sql + "GROUP BY s.carrier_id, c.carrier_name, c.mode", params)
    overall["on_time_rate"] = 1 - overall["delay_rate"]

    sql2 = ("SELECT carrier_id, carrier_name, lane_id, lane_name, shipments, on_time_rate, avg_delay_days, "
            "avg_actual_cost_usd, avg_cost_overrun_usd FROM freight.v_carrier_lane_performance WHERE 1 = 1 ")
    params2 = []
    if carrier_ids:
        sql2 += f"AND carrier_id IN ({placeholders(len(carrier_ids))}) "
        params2 += carrier_ids
    if lane_ids:
        sql2 += f"AND lane_id IN ({placeholders(len(lane_ids))}) "
        params2 += lane_ids
    lanes = run_query(sql2, params2)
    if not lanes.empty:
        lanes = lanes[lanes["shipments"] >= min_shipments].sort_values("on_time_rate")
        carrier_overall = dict(zip(overall["carrier_id"], overall["on_time_rate"]))
        lanes["vs_carrier_overall_pts"] = (lanes["on_time_rate"] -
                                           lanes["carrier_id"].map(carrier_overall)).astype(float) * 100

    # Market context: latest live diesel (EIA) and USD->CAD (Frankfurter), if loaded
    market = {}
    try:
        diesel = run_query("SELECT week_start, diesel_price_usd_gal FROM freight.diesel_prices_eia")
        if not diesel.empty:
            diesel = diesel.sort_values("week_start")
            latest = float(diesel["diesel_price_usd_gal"].iloc[-1])
            prior = float(diesel["diesel_price_usd_gal"].iloc[max(0, len(diesel) - 13)])
            market.update({"diesel_usd_gal_latest": round(latest, 3),
                           "diesel_week": str(diesel["week_start"].iloc[-1])[:10],
                           "diesel_change_12w_pct": round((latest / prior - 1) * 100, 1),
                           "fuel_surcharge_usd_per_mile": round(max(0.0, latest - FSC_BASE_PRICE) / FSC_MPG, 3),
                           "diesel_source": "EIA (live)"})
    except Exception:
        pass
    if "diesel_usd_gal_latest" not in market:
        sim = run_query("SELECT week_start, diesel_price_usd_gal FROM freight.fuel_prices").sort_values("week_start")
        latest = float(sim["diesel_price_usd_gal"].iloc[-1])
        market.update({"diesel_usd_gal_latest": round(latest, 3), "diesel_week": str(sim["week_start"].iloc[-1])[:10],
                       "fuel_surcharge_usd_per_mile": round(max(0.0, latest - FSC_BASE_PRICE) / FSC_MPG, 3),
                       "diesel_source": "SIMULATED (EIA data not loaded)"})
    try:
        fx = run_query("SELECT rate_date, rate FROM freight.fx_rates WHERE quote_currency = 'CAD'")
        if not fx.empty:
            fx = fx.sort_values("rate_date")
            market.update({"usd_to_cad_latest": round(float(fx["rate"].iloc[-1]), 4),
                           "fx_date": str(fx["rate_date"].iloc[-1])[:10], "fx_source": "Frankfurter/ECB (live)"})
    except Exception:
        pass

    result = {
        "carriers_overall": to_records(overall.sort_values("on_time_rate"),
                                       ["carrier_id", "carrier_name", "mode", "shipments", "on_time_rate",
                                        "avg_delay_days", "avg_cost_overrun_usd"]),
        "carrier_lane_detail": [] if lanes.empty else to_records(
            lanes.head(30), ["carrier_id", "carrier_name", "lane_id", "lane_name", "shipments", "on_time_rate",
                             "vs_carrier_overall_pts", "avg_delay_days", "avg_cost_overrun_usd"]),
        "market": market,
        "note": "History = delivered shipments Jan 2024-Oct 2026 (synthetic). vs_carrier_overall_pts < 0 means "
                "the carrier performs worse on that lane than it does overall.",
    }
    return fit_to_budget(result, "carrier_lane_detail")


# ---------------------------------------------------------------------------
# Tool 4: calculate_risk_and_recommendation
# ---------------------------------------------------------------------------
_MODELS = None


def load_models():
    global _MODELS
    if _MODELS is None:
        import xgboost as xgb
        meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
        if not os.path.exists(meta_path):
            raise RuntimeError("models/ not found - run train_and_predict.py in this folder first")
        with open(meta_path) as f:
            meta = json.load(f)
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(MODEL_DIR, "delay_classifier.json"))
        _MODELS = (clf, meta)
    return _MODELS


def build_matrix(df, meta):
    X = df[meta["numeric_features"]].astype(float).copy()
    for col in meta["categorical_features"]:
        for value in meta["categories"][col]:
            X[f"{col}={value}"] = (df[col] == value).astype(float)
    return X[meta["feature_names"]]


def expected_extra_cost(p, meta):
    h = meta["accessorial_hurdle"]
    return p * h["late_mean_usd"] + (1 - p) * h["on_time_mean_usd"]


def alt_quote(history, carrier_id, equipment, miles, fallback_rate):
    pool = history[history["carrier_id"] == carrier_id]
    if len(pool):
        pool = pool[pool["ship_date"] >= pool["ship_date"].max() - pd.Timedelta(days=365)]
    same = pool[pool["equipment_type"] == equipment]
    pool = same if len(same) >= 3 else pool
    if len(pool) >= 3:
        return float((pool["quoted_cost_usd"] / pool["distance_miles"]).median() * miles)
    return float(fallback_rate * miles)


def recompute_switches(df, penalty, meta, carriers):
    """Re-run the expected-value carrier-switch rule for Booked Medium/High shipments at a new late penalty."""
    clf, _ = load_models()
    min_savings = float(meta.get("min_savings_usd", 100.0))
    info = carriers.set_index("carrier_id")
    out = {}
    targets = df[(df["status"] == "Booked") & (df["risk_tier"].isin(["High", "Medium"]))]
    for lane_id in targets["lane_id"].unique():
        hist = run_query("SELECT s.carrier_id, s.equipment_type, s.ship_date, s.quoted_cost_usd, l.distance_miles "
                         "FROM freight.shipments s JOIN freight.lanes l ON l.lane_id = s.lane_id "
                         f"WHERE s.status = 'Delivered' AND s.lane_id = {PARAM}", [lane_id])
        hist["ship_date"] = pd.to_datetime(hist["ship_date"])
        lane_carriers = sorted(hist["carrier_id"].unique())
        for _, row in targets[targets["lane_id"] == lane_id].iterrows():
            p_now = float(row["delay_probability"])
            current_total = float(row["quoted_cost_usd"]) + expected_extra_cost(p_now, meta)
            best = None
            for alt in lane_carriers:
                if alt == row["carrier_id"] or info.loc[alt, "mode"] == "Intermodal":
                    continue
                cand = row.to_frame().T.copy()
                cand["carrier_id"] = alt
                cand["reliability_baseline"] = float(info.loc[alt, "reliability_baseline"])
                if row["equipment_type"] == "Intermodal Container":
                    cand["equipment_type"] = "Dry Van"
                    cand["planned_transit_days"] = max(1, math.ceil(float(row["distance_miles"]) / 500)
                                                       + int(row["is_cross_border"]))
                p_alt = float(clf.predict_proba(build_matrix(cand, meta))[:, 1][0])
                alt_total = alt_quote(hist, alt, cand["equipment_type"].iloc[0], float(row["distance_miles"]),
                                      float(info.loc[alt, "base_rate_per_mile"])) + expected_extra_cost(p_alt, meta)
                net = (current_total + p_now * penalty) - (alt_total + p_alt * penalty)
                if p_alt <= p_now and net >= min_savings and (best is None or net > best["net_benefit_usd"]):
                    best = {"alt_carrier_id": alt, "alt_carrier_name": info.loc[alt, "carrier_name"],
                            "alt_delay_probability": round(p_alt, 4),
                            "freight_cost_change_usd": round(alt_total - current_total, 2),
                            "net_benefit_usd": round(net, 2)}
            out[row["shipment_id"]] = best
    return out


def calculate_risk_and_recommendation(args):
    clf, meta = load_models()
    shipment_ids = v_list(args, "shipment_ids", 200)
    lane_ids, carrier_ids = v_list(args, "lane_ids"), v_list(args, "carrier_ids")
    tiers = args.get("risk_tiers") or None
    if tiers is not None:
        tiers = [tiers] if isinstance(tiers, str) else tiers
        tiers = [str(t).capitalize() for t in tiers]
        if any(t not in ("High", "Medium", "Low") for t in tiers):
            raise ToolInputError("risk_tiers must contain only High, Medium, Low")
    stored_penalty = float(meta.get("late_penalty_usd", 1000.0))
    penalty = v_number(args, "late_penalty_usd", stored_penalty, 0, 20000)
    limit = v_int(args, "limit", 8, 1, 25)
    d_from = v_date(args, "delivery_from", None)
    d_to = v_date(args, "delivery_to", None)

    sql = ("SELECT p.shipment_id, p.delay_probability, p.model_delay_probability, p.days_overdue, "
           "p.expected_delay_days, p.predicted_cost_usd, p.expected_exposure_usd, p.risk_tier, "
           "p.recommended_action, p.weather_severity_used, p.is_overdue, p.alt_carrier_id, "
           "p.alt_delay_probability, p.alt_expected_cost_usd, p.expected_savings_usd, p.predicted_at_utc, "
           "s.status, s.lane_id, s.carrier_id, s.equipment_type, s.weight_lbs, s.ship_date, "
           "s.planned_transit_days, s.planned_delivery_date, s.diesel_price_usd_gal, s.port_congestion_index, "
           "s.quoted_cost_usd, l.lane_name, l.distance_miles, l.is_port_origin, l.is_winter_exposed, "
           "l.is_cross_border, c.carrier_name, c.reliability_baseline "
           "FROM freight.shipment_predictions p "
           "JOIN freight.shipments s ON s.shipment_id = p.shipment_id "
           "JOIN freight.lanes l ON l.lane_id = s.lane_id "
           "JOIN freight.carriers c ON c.carrier_id = s.carrier_id "
           f"WHERE p.model_version = {PARAM} ")
    params = [meta["model_version"]]
    if shipment_ids:
        sql += f"AND p.shipment_id IN ({placeholders(len(shipment_ids))}) "
        params += shipment_ids
    df = run_query(sql, params)
    if df.empty and shipment_ids:
        return {"note": f"None of these shipment IDs are active shipments with predictions: {shipment_ids}. "
                        "They may not exist, or may already be delivered.", "late_penalty_usd": penalty}
    if df.empty:
        return {"note": "No predictions found. Run train_and_predict.py and load outputs/predictions_load.sql."}
    # SQL NULLs arrive as None; a column that is NULL for every selected row becomes "object" dtype,
    # which breaks arithmetic (found by evaluation scenario E4). Coerce all numeric columns explicitly.
    numeric_cols = ["delay_probability", "model_delay_probability", "days_overdue", "expected_delay_days",
                    "predicted_cost_usd", "expected_exposure_usd", "weather_severity_used", "is_overdue",
                    "alt_delay_probability", "alt_expected_cost_usd", "expected_savings_usd", "quoted_cost_usd",
                    "weight_lbs", "planned_transit_days", "diesel_price_usd_gal", "port_congestion_index",
                    "distance_miles", "is_port_origin", "is_winter_exposed", "is_cross_border", "reliability_baseline"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if lane_ids:
        df = df[df["lane_id"].isin(lane_ids)]
    if carrier_ids:
        df = df[df["carrier_id"].isin(carrier_ids)]
    if tiers:
        df = df[df["risk_tier"].isin(tiers)]
    # Delivery window: applies to shipments not yet late; overdue shipments are always kept (still open)
    df["planned_delivery_date"] = pd.to_datetime(df["planned_delivery_date"])
    overdue_mask = df["is_overdue"].fillna(0).astype(int) == 1
    if d_from:
        df = df[overdue_mask | (df["planned_delivery_date"] >= pd.Timestamp(d_from))]
        overdue_mask = df["is_overdue"].fillna(0).astype(int) == 1
    if d_to:
        df = df[overdue_mask | (df["planned_delivery_date"] <= pd.Timestamp(d_to))]
    if df.empty:
        return {"note": "No shipments match these filters.", "late_penalty_usd": penalty}

    # Features needed if we re-score carrier alternatives
    df["ship_date"] = pd.to_datetime(df["ship_date"])
    df["ship_month"] = df["ship_date"].dt.month
    df["is_winter_month"] = df["ship_month"].isin([12, 1, 2]).astype(int)
    df["weather_severity"] = df["weather_severity_used"].fillna(0)
    carriers = run_query("SELECT carrier_id, carrier_name, mode, reliability_baseline, base_rate_per_mile "
                         "FROM freight.carriers")
    names = dict(zip(carriers["carrier_id"], carriers["carrier_name"]))

    recomputed = abs(penalty - stored_penalty) > 0.01
    if recomputed:
        switches = recompute_switches(df, penalty, meta, carriers)
        df["alt_carrier_id"] = df["shipment_id"].map(lambda s: (switches.get(s) or {}).get("alt_carrier_id"))
        df["alt_delay_probability"] = df["shipment_id"].map(lambda s: (switches.get(s) or {}).get("alt_delay_probability"))
        df["freight_cost_change_usd"] = df["shipment_id"].map(lambda s: (switches.get(s) or {}).get("freight_cost_change_usd"))
        df["net_benefit_usd"] = df["shipment_id"].map(lambda s: (switches.get(s) or {}).get("net_benefit_usd"))
        df["recommendation"] = df.apply(
            lambda r: (f"Switch to {names.get(r['alt_carrier_id'])} ({r['alt_carrier_id']}): delay risk "
                       f"{r['delay_probability']:.0%} -> {r['alt_delay_probability']:.0%}, freight cost change "
                       f"{r['freight_cost_change_usd']:+,.0f} USD, net benefit {r['net_benefit_usd']:,.0f} USD "
                       f"at {penalty:,.0f} USD per late delivery.")
            if isinstance(r["alt_carrier_id"], str) else
            (r["recommended_action"] if not str(r["recommended_action"]).startswith("Switch to")
             else "No carrier switch justified at this late-delivery cost; monitor closely."), axis=1)
    else:
        df["freight_cost_change_usd"] = -df["expected_savings_usd"]
        df["net_benefit_usd"] = np.nan
        df["recommendation"] = df["recommended_action"]
    df["alt_carrier_name"] = df["alt_carrier_id"].map(names)
    df["business_cost_of_lateness_usd"] = df["delay_probability"].astype(float) * penalty

    tier_summary = {}
    for t in ("High", "Medium", "Low"):
        sub = df[df["risk_tier"] == t]
        if len(sub):
            e = round(float(sub["expected_exposure_usd"].sum()), 0)
            b = round(float(sub["business_cost_of_lateness_usd"].sum()), 0)
            tier_summary[t] = {"shipments": int(len(sub)), "expected_extra_carrier_charges_usd": e,
                               "expected_business_cost_of_lateness_usd": b, "total_exposure_usd": e + b}
    switches_df = df[df["alt_carrier_id"].notna()].sort_values("net_benefit_usd", ascending=False, na_position="last")
    late = df[df["is_overdue"].fillna(0).astype(int) == 1].sort_values("days_overdue", ascending=False)
    at_risk = df[df["is_overdue"].fillna(0).astype(int) == 0].sort_values("delay_probability", ascending=False)

    def summarize(sub):
        extra = round(float(sub["expected_exposure_usd"].sum()), 0)
        lateness = round(float(sub["business_cost_of_lateness_usd"].sum()), 0)
        return {"shipments": int(len(sub)), "expected_extra_carrier_charges_usd": extra,
                "expected_business_cost_of_lateness_usd": lateness, "total_exposure_usd": extra + lateness}

    late_cols = ["shipment_id", "lane_name", "carrier_name", "days_overdue", "model_delay_probability",
                 "expected_exposure_usd", "recommendation"]
    risk_cols = ["shipment_id", "status", "lane_name", "carrier_name", "risk_tier", "delay_probability",
                 "weather_severity_used", "planned_delivery_date", "expected_exposure_usd", "recommendation"]
    switch_cols = ["shipment_id", "lane_name", "carrier_name", "delay_probability", "alt_carrier_id",
                   "alt_carrier_name", "alt_delay_probability", "freight_cost_change_usd", "net_benefit_usd"]

    result = {
        "model_version": meta["model_version"],
        "as_of": AS_OF.isoformat(),
        "predictions_generated_utc": str(df["predicted_at_utc"].max())[:16],
        "late_penalty_usd": penalty,
        "switches_recomputed_for_this_penalty": recomputed,
        "filters": {"risk_tiers": tiers, "lane_ids": lane_ids, "carrier_ids": carrier_ids,
                    "delivery_from": d_from and d_from.isoformat(),
                    "delivery_to": d_to and d_to.isoformat(),
                    "note": "overdue shipments are always included regardless of delivery window"},
        "shipments_evaluated": int(len(df)),
        "by_risk_tier": tier_summary,
        "totals": {**summarize(df),
                   "carrier_switches_recommended": int(len(switches_df)),
                   "freight_cost_change_if_all_switches_usd": round(float(switches_df["freight_cost_change_usd"].sum()), 0)
                   if len(switches_df) else 0.0},
        "already_late": {**summarize(late), "top": to_records(late.head(limit), late_cols)},
        "predicted_at_risk_not_yet_late": {**summarize(at_risk), "top": to_records(at_risk.head(limit), risk_cols)},
        "recommended_switches": to_records(switches_df.head(10), switch_cols),
        "definitions": "delay_probability is 1.0 for overdue shipments (already late); model_delay_probability is "
                       "the pre-trip model score. Extra carrier charges = model-expected detention/accessorials. "
                       "Business cost of lateness = delay_probability x late_penalty_usd (an assumption).",
    }
    # Shrink the three lists until the result fits the size cap (totals are always kept intact)
    cap = max(len(result["already_late"]["top"]), len(result["predicted_at_risk_not_yet_late"]["top"]),
              len(result["recommended_switches"]))
    text = json.dumps(result, default=str)
    while len(text) > MAX_TOOL_RESULT_CHARS and cap > 0:
        cap -= 1
        result["already_late"]["top"] = result["already_late"]["top"][:cap]
        result["predicted_at_risk_not_yet_late"]["top"] = result["predicted_at_risk_not_yet_late"]["top"][:cap]
        result["recommended_switches"] = result["recommended_switches"][:cap]
        result["truncated"] = f"lists shortened to {cap} rows to fit the size limit; totals cover all rows"
        text = json.dumps(result, default=str)
    return result


# ---------------------------------------------------------------------------
# Tool registry (schemas the model sees)
# ---------------------------------------------------------------------------
ID_LIST = lambda desc: {"type": "array", "items": {"type": "string"}, "description": desc}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_shipment_context",
        "description": "List active shipments (Booked / In Transit) with dates, lane, carrier, quoted cost and days "
                       "overdue, plus counts by status and by carrier (booked / in transit / overdue) for ALL "
                       "matching shipments. Use the by_carrier counts for 'how many' questions - one call is enough.",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["active", "Booked", "In Transit"]},
            "delivery_from": {"type": "string", "description": "Planned delivery on/after, YYYY-MM-DD"},
            "delivery_to": {"type": "string", "description": "Planned delivery on/before, YYYY-MM-DD"},
            "lane_ids": ID_LIST("Lane IDs like L01"),
            "carrier_ids": ID_LIST("Carrier IDs like C04"),
            "shipment_ids": ID_LIST("Shipment IDs like SHP-001234"),
            "overdue_only": {"type": "boolean"},
            "limit": {"type": "integer", "description": "Max shipments to list (1-50, default 20)"}}}}},
    {"type": "function", "function": {
        "name": "get_route_conditions",
        "description": "Forecast weather severity along each lane's route (worst of 5 points per day), route "
                       "distance/drive time, and port congestion for port-origin lanes.",
        "parameters": {"type": "object", "properties": {
            "lane_ids": ID_LIST("Lane IDs to check (required)"),
            "start_date": {"type": "string", "description": "YYYY-MM-DD, default today"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD, default today + 7 days"}},
            "required": ["lane_ids"]}}},
    {"type": "function", "function": {
        "name": "get_carrier_performance",
        "description": "Historical on-time rate, average delay days and cost overruns by carrier and by "
                       "carrier x lane, plus current diesel price, fuel surcharge per mile and USD->CAD rate.",
        "parameters": {"type": "object", "properties": {
            "carrier_ids": ID_LIST("Carrier IDs, omit for all"),
            "lane_ids": ID_LIST("Lane IDs, omit for all"),
            "min_shipments": {"type": "integer", "description": "Minimum history per carrier-lane (default 5)"}}}}},
    {"type": "function", "function": {
        "name": "calculate_risk_and_recommendation",
        "description": "Model-based delay probability, risk tier, expected extra charges, business cost of "
                       "lateness and carrier-switch recommendations. Filter by shipment, lane, carrier, risk tier and "
                       "delivery window; totals are computed for the filtered set. Returns three lists: already_late (overdue), "
                       "predicted_at_risk_not_yet_late, and recommended_switches. Use delivery_from/delivery_to for "
                       "time windows such as 'this week'. Pass late_penalty_usd to test a different cost of a late "
                       "delivery (switch decisions are recomputed).",
        "parameters": {"type": "object", "properties": {
            "shipment_ids": ID_LIST("Limit to these shipments, omit for all active"),
            "lane_ids": ID_LIST("Limit to these lanes, e.g. ['L01']"),
            "carrier_ids": ID_LIST("Limit to these carriers, e.g. ['C04']"),
            "risk_tiers": {"type": "array", "items": {"type": "string", "enum": ["High", "Medium", "Low"]}},
            "late_penalty_usd": {"type": "number", "description": "Business cost of one late delivery in USD"},
            "delivery_from": {"type": "string", "description": "Planned delivery on/after, YYYY-MM-DD (overdue shipments always included)"},
            "delivery_to": {"type": "string", "description": "Planned delivery on/before, YYYY-MM-DD (overdue shipments always included)"},
            "limit": {"type": "integer", "description": "Max rows per list (1-25, default 8)"}}}}},
]

TOOL_FUNCTIONS = {
    "get_shipment_context": get_shipment_context,
    "get_route_conditions": get_route_conditions,
    "get_carrier_performance": get_carrier_performance,
    "calculate_risk_and_recommendation": calculate_risk_and_recommendation,
}


def execute_tool(name, raw_arguments):
    """Validate and run one tool call. Never raises: errors go back to the model as data."""
    start = time.time()
    try:
        if name not in TOOL_FUNCTIONS:
            raise ToolInputError(f"unknown tool '{name}'")
        args = json.loads(raw_arguments or "{}")
        if not isinstance(args, dict):
            raise ToolInputError("arguments must be a JSON object")
        result = TOOL_FUNCTIONS[name](args)
        ok, error = True, None
    except ToolInputError as exc:
        result, ok, error = {"error": f"Invalid input: {exc}"}, False, str(exc)
    except json.JSONDecodeError:
        result, ok, error = {"error": "Arguments were not valid JSON"}, False, "invalid JSON"
    except Exception as exc:                      # DB timeout, connection drop, model file missing ...
        result, ok, error = {"error": f"Tool failed: {type(exc).__name__}: {str(exc)[:300]}"}, False, str(exc)[:900]
    text = json.dumps(result, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        # Last resort: keep the output valid JSON so the model (and the evaluator) can still parse it
        text = json.dumps({"truncated": True, "note": "result exceeded the size limit; narrow the filters",
                           "partial_result_text": text[:MAX_TOOL_RESULT_CHARS - 200]})
    return text, ok, error, int((time.time() - start) * 1000)


# ---------------------------------------------------------------------------
# Model client (Azure OpenAI in Microsoft Foundry, or GitHub Models)
# ---------------------------------------------------------------------------
def make_client():
    provider = os.environ.get("LLM_PROVIDER", "azure").lower()
    if provider == "github":
        from openai import OpenAI
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            sys.exit("Set GITHUB_TOKEN (a fine-grained token with Models: read-only).")
        client = OpenAI(base_url="https://models.github.ai/inference", api_key=token,
                        timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)
        return client, os.environ.get("GITHUB_MODEL", "openai/gpt-4.1-mini"), "github"
    from openai import AzureOpenAI, OpenAI
    missing = [k for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
               if not os.environ.get(k)]
    if missing:
        sys.exit(f"Set these environment variables first: {', '.join(missing)}  (or set LLM_PROVIDER=github)")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].strip()
    if "/openai/v1" in endpoint:
        # Newer Foundry "v1" endpoint: standard OpenAI client, no api-version parameter
        base_url = endpoint[:endpoint.index("/openai/v1")] + "/openai/v1/"
        client = OpenAI(base_url=base_url, api_key=os.environ["AZURE_OPENAI_API_KEY"],
                        timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)
        return client, os.environ["AZURE_OPENAI_DEPLOYMENT"], "azure-v1"
    client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                         api_key=os.environ["AZURE_OPENAI_API_KEY"],
                         api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                         timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)
    return client, os.environ["AZURE_OPENAI_DEPLOYMENT"], "azure"


_DIRECTORY = None


def directory_text():
    """Lane and carrier IDs, so the model can map 'Chicago' -> L01 without spending a tool call."""
    global _DIRECTORY
    if _DIRECTORY is None:
        try:
            lanes = run_query("SELECT lane_id, lane_name FROM freight.lanes").sort_values("lane_id")
            carriers = run_query("SELECT carrier_id, carrier_name, mode FROM freight.carriers").sort_values("carrier_id")
            _DIRECTORY = ("Lanes: " + "; ".join(f"{r.lane_id} {r.lane_name}" for r in lanes.itertuples()) +
                          "\nCarriers: " + "; ".join(f"{r.carrier_id} {r.carrier_name} ({r.mode})"
                                                     for r in carriers.itertuples()))
        except Exception:
            _DIRECTORY = ""
    return _DIRECTORY


def system_prompt():
    return f"""You are a freight risk analyst assistant for a US logistics team. Today is {AS_OF.isoformat()} ({AS_OF.strftime('%A')}).
You answer questions about active shipments using ONLY these tools: get_shipment_context, get_route_conditions,
get_carrier_performance, calculate_risk_and_recommendation.

Rules:
1. Never invent shipment IDs, probabilities, costs, weather or carrier figures. Every number you state must
   appear in a tool result from this conversation. If the tools do not provide it, say so.
2. Delay probabilities, risk tiers, exposure and carrier-switch decisions come only from
   calculate_risk_and_recommendation. Do not compute or adjust them yourself.
3. Call only the tools the question needs. You may call several tools in one turn. You have at most
   {MAX_TURNS} turns in total; on the last one you must answer.
4. "This week" means the next 7 days: delivery_from = today, delivery_to = today + 6 days, unless the user
   says otherwise. Pass these dates to calculate_risk_and_recommendation for any time-bound question.
5. Financial exposure has two parts: expected extra carrier charges, and the business cost of lateness
   (delay probability x late-delivery cost assumption). Quote total_exposure_usd for combined figures; never add,
   subtract or average numbers yourself; if you need a subtotal (e.g. one carrier on one lane), call the tool
   again with lane_ids / carrier_ids filters. Always state the late-delivery cost used.
9. When you compare (a lane vs. a carrier's overall record, one carrier vs. another), state both figures.
10. Questions about whether to keep, use, drop or switch a carrier, or about carrier reliability, need the
    carrier's track record: call get_carrier_performance (with the carrier and lane) as well as any risk tool.
6. Carrier switches are only possible for Booked shipments; in-transit loads cannot change carrier.
7. If a question is not about these shipments, carriers, lanes or costs, say it is outside your scope.
8. If the user asks you to skip the tools, guess, or "estimate yourself", do not comply: explain in one sentence
   that risk and cost figures come only from the validated model, then call the tool and report its result.

Conversation: earlier messages may be included for follow-up questions ("what about Redline?", "summarize
this"). Figures stated in your own earlier answers came from tools and may be repeated; for anything new, call
the tools. For "summarize our conversation" requests, summarize only what was already established.

Reports: if the user asks for a report, scorecard, summary or table, you may use up to 500 words and markdown
tables (one row per shipment, carrier or lane), still using only tool figures.

Answer format for ordinary questions (under 300 words):
- one-line headline with the time window and totals;
- "Already late": count, exposure, and the top few by days overdue (ID, lane, carrier, days late);
- "Predicted at risk (not yet late)": count, exposure, and the top shipments (ID, lane, carrier, probability, why
  if a tool shows it, e.g. weather severity or a weak carrier-lane record);
- "Recommended carrier switches": each one with ID, current -> new carrier, risk change and cost change;
- recommended next actions;
- a final "Data notes" line that ALWAYS includes the late-delivery cost used and these notes
(shipments and carriers are synthetic; weather, routing, diesel and FX are live where loaded; port congestion is simulated).

{directory_text()}"""


def assistant_message_to_dict(msg):
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]
    return d


def call_model(client, model, messages, tool_choice):
    kwargs = dict(model=model, messages=messages, tools=TOOLS, tool_choice=tool_choice,
                  temperature=0.1, max_tokens=MAX_OUTPUT_TOKENS)
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Newer reasoning models reject temperature / max_tokens: retry once with compatible parameters
        text = str(exc).lower()
        if "max_tokens" in text or "temperature" in text or "unsupported" in text:
            kwargs.pop("temperature", None)
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            return client.chat.completions.create(**kwargs)
        raise


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def tokens_used_today():
    try:
        midnight = datetime.now(timezone.utc).strftime("%Y-%m-%d 00:00:00")
        df = run_query(f"SELECT SUM(total_tokens) AS t FROM freight.agent_runs WHERE started_at_utc >= {PARAM}",
                       [midnight])
        value = df["t"].iloc[0] if not df.empty else 0
        return int(value) if value is not None and not pd.isna(value) else 0
    except Exception:
        return 0


def db_log(sql, params):
    try:
        run_exec(sql, params)
    except Exception as exc:
        print(f"  (logging to SQL failed, continuing: {str(exc)[:120]})")


def write_jsonl(record):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "agent_runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
def run_agent(question, client, model, provider, scenario=None, verbose=True, history=None, return_details=False):
    """Answer one question. `history` = earlier [{"role": "user"|"assistant", "content": ...}] for follow-ups.
    Returns the answer text, or (return_details=True) a dict with the answer, tool calls and usage."""
    used_today = tokens_used_today()
    if used_today >= DAILY_TOKEN_BUDGET:
        msg = (f"Daily token budget reached ({used_today:,} of {DAILY_TOKEN_BUDGET:,}). "
               "Raise DAILY_TOKEN_BUDGET in freight_agent.py deliberately if you need more today.")
        return {"answer": msg, "run_id": None, "tool_calls": [], "total_tokens": 0, "turns_used": 0,
                "completed": False, "latency_ms": 0} if return_details else msg

    run_id = f"run-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
    started = time.time()
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    db_log("INSERT INTO freight.agent_runs (run_id, started_at_utc, scenario, user_question) "
           f"VALUES ({placeholders(4)})", [run_id, started_utc, scenario, question[:1000]])

    prior = []
    for m in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            prior.append({"role": m["role"], "content": str(m["content"])[:3000]})
    messages = [{"role": "system", "content": system_prompt()}] + prior + [{"role": "user", "content": question}]
    total_tokens, turns_used, hit_cap, final_answer, completed = 0, 0, False, None, False
    trace = []
    force_answer = False

    for turn in range(1, MAX_TURNS + 1):
        turns_used = turn
        last_turn = turn == MAX_TURNS or force_answer
        try:
            response = call_model(client, model, messages, "none" if last_turn else "auto")
        except Exception as exc:
            final_answer = f"The model call failed ({type(exc).__name__}: {str(exc)[:200]}). No answer produced."
            break
        if getattr(response, "usage", None):
            total_tokens += int(response.usage.total_tokens or 0)
        msg = response.choices[0].message

        if msg.tool_calls and not last_turn:
            messages.append(assistant_message_to_dict(msg))
            for tc in msg.tool_calls:
                result_text, ok, error, ms = execute_tool(tc.function.name, tc.function.arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                trace.append({"turn": turn, "tool": tc.function.name, "arguments": tc.function.arguments,
                              "ok": ok, "error": error, "latency_ms": ms, "result_chars": len(result_text),
                              "result": result_text})
                db_log("INSERT INTO freight.agent_tool_calls (run_id, turn_number, tool_name, arguments_json, "
                       f"succeeded, error_message, latency_ms) VALUES ({placeholders(7)})",
                       [run_id, turn, tc.function.name, (tc.function.arguments or "")[:3900], int(ok),
                        (error or None) and error[:1000], ms])
                if verbose:
                    status = "ok" if ok else f"ERROR: {error[:80]}"
                    print(f"  turn {turn}: {tc.function.name}({(tc.function.arguments or '')[:90]}) -> {status} [{ms} ms]")
            if total_tokens >= MAX_TOKENS_PER_RUN:
                force_answer = True
                if verbose:
                    print(f"  token cap reached ({total_tokens:,}) - forcing a final answer")
            continue

        final_answer = msg.content or "(The model returned an empty answer.)"
        completed = True
        hit_cap = turn == MAX_TURNS and bool(trace)
        break

    latency_ms = int((time.time() - started) * 1000)
    if final_answer is None:
        final_answer = "Maximum agent turns reached without an answer."
    db_log("UPDATE freight.agent_runs SET turns_used = %s, hit_turn_cap = %s, completed = %s, total_tokens = %s, "
           "latency_ms = %s, final_answer = %s WHERE run_id = %s".replace("%s", PARAM),
           [turns_used, int(hit_cap), int(completed), total_tokens, latency_ms, final_answer, run_id])
    write_jsonl({"run_id": run_id, "started_at_utc": started_utc, "provider": provider, "model": model,
                 "scenario": scenario, "question": question, "turns_used": turns_used, "hit_turn_cap": hit_cap,
                 "completed": completed, "total_tokens": total_tokens, "latency_ms": latency_ms,
                 "tool_calls": trace, "final_answer": final_answer})
    if verbose:
        print(f"  [{run_id}] turns {turns_used}/{MAX_TURNS}, tool calls {len(trace)}, tokens {total_tokens:,}, "
              f"{latency_ms / 1000:.1f}s, today's usage {used_today + total_tokens:,}/{DAILY_TOKEN_BUDGET:,}")
    if return_details:
        return {"answer": final_answer, "run_id": run_id, "tool_calls": trace, "total_tokens": total_tokens,
                "turns_used": turns_used, "completed": completed, "latency_ms": latency_ms,
                "tokens_used_today": used_today + total_tokens, "daily_budget": DAILY_TOKEN_BUDGET}
    return final_answer


# ---------------------------------------------------------------------------
# --check: exercise DB, models and every tool without calling the model (free)
# ---------------------------------------------------------------------------
def self_check():
    print("Checking database connection ...")
    run_query("SELECT COUNT(*) AS n FROM freight.shipments")
    print("  ok")
    print("Checking saved models ...")
    _, meta = load_models()
    print(f"  ok - {meta['model_version']}, late penalty at training: ${meta.get('late_penalty_usd', 1000):,.0f}")
    checks = [
        ("get_shipment_context", {"overdue_only": True, "limit": 3}),
        ("get_route_conditions", {"lane_ids": ["L01", "L04"]}),
        ("get_carrier_performance", {"carrier_ids": ["C04"], "lane_ids": ["L01"]}),
        ("calculate_risk_and_recommendation", {"risk_tiers": ["High"], "limit": 3}),
        ("calculate_risk_and_recommendation", {"risk_tiers": ["High", "Medium"], "late_penalty_usd": 2500, "limit": 3}),
        ("get_shipment_context", {"lane_ids": ["Chicago"]}),          # should be rejected by validation
    ]
    for name, args in checks:
        text, ok, error, ms = execute_tool(name, json.dumps(args))
        print(f"\n{name}({args}) -> {'ok' if ok else 'rejected: ' + error} [{ms} ms, {len(text)} chars]")
        print("  " + text[:600] + (" ..." if len(text) > 600 else ""))
    print("\nAll checks finished. If every tool says ok (and the last one is rejected), the agent is ready.")


def main():
    global AS_OF
    parser = argparse.ArgumentParser(description="Freight cost & delay AI agent")
    parser.add_argument("question", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--check", action="store_true", help="Test DB, models and tools without calling the model")
    parser.add_argument("--scenario", default=None, help="Label stored with the run (e.g. A, B, C)")
    parser.add_argument("--as-of", default=None, help="Override today's date (YYYY-MM-DD)")
    parser.add_argument("--tool", default=None, help="Run one tool directly and print its JSON (no model call)")
    parser.add_argument("--args", default="{}", help="JSON arguments for --tool")
    parser.add_argument("--json-io", action="store_true",
                        help="Read {question, history} as JSON on stdin and print one JSON result (used by the chat app)")
    args = parser.parse_args()
    if args.as_of:
        AS_OF = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    if args.tool:
        text, ok, error, _ = execute_tool(args.tool, args.args)
        print(text)
        sys.exit(0 if ok else 1)

    if args.check:
        self_check()
        return

    if args.json_io:
        payload = json.loads(sys.stdin.read() or "{}")
        client, model, provider = make_client()
        result = run_agent(str(payload.get("question", "")), client, model, provider,
                           scenario=payload.get("scenario", "CHAT"), verbose=False,
                           history=payload.get("history"), return_details=True)
        result.update({"provider": provider, "model": model})
        print("JSONRESULT " + json.dumps(result, default=str))
        return

    client, model, provider = make_client()
    print(f"Freight agent ready ({provider}: {model}). Max {MAX_TURNS} turns per question.\n")
    if args.question:
        print(run_agent(" ".join(args.question), client, model, provider, scenario=args.scenario))
        return
    history = []                      # interactive mode remembers the conversation for follow-ups
    print("Interactive mode: follow-up questions remember earlier answers. Type 'new' to start over, 'exit' to quit.")
    while True:
        try:
            q = input("\nAsk a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", ""):
            break
        if q.lower() == "new":
            history = []
            print("(conversation cleared)")
            continue
        answer = run_agent(q, client, model, provider, scenario=args.scenario or "INTERACTIVE", history=history)
        print("\n" + answer)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
