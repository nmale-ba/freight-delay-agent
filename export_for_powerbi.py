"""
export_for_powerbi.py
Predictive Freight Cost & Delay Agent - Phase 6: data for the Power BI dashboard

Self-contained (no imports from other project files). Requires:  pip install pymssql pandas openpyxl
Uses the same SQL_* environment variables as the other scripts. No model calls, no cost.

Writes outputs/freight_dashboard_data.xlsx with one named Excel table per sheet, already joined and
with business fields computed, so the Power BI report (web or Desktop) needs no relationships and
only simple aggregations:

    ActiveShipments        100 rows   one per active shipment: risk, exposure, weather, action, switch
    ShipmentHistory      1,200 rows   delivered shipments: delays and cost overruns by carrier/lane/month
    CarrierLanePerf                    on-time rate by carrier x lane vs. the carrier's overall rate
    RouteConditions                    weather severity by lane and day (Open-Meteo along OSRM routes)
    DieselPrices                       weekly diesel and the implied fuel surcharge per mile (EIA)
    SwitchSensitivity                  carrier switches and cost vs. the assumed cost of a late delivery
    ModelEvaluation                    Phase 2 model metrics vs. baselines
    AgentEvaluation                    Phase 5 scenario results, agent vs. no-tools baseline
    EvaluationProgress                 Run 1-4 summary (from the README)
    AgentRuns                          every logged agent run: turns, tokens, latency
    About                              as-of date, sources, data notes

Usage:
    python export_for_powerbi.py
"""

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

OUT_PATH = os.path.join("outputs", "freight_dashboard_data.xlsx")
AGENT_SCRIPT = "freight_agent.py"
MODEL_META = os.path.join("models", "model_metadata.json")
TODAY = date.today()
SENSITIVITY_PENALTIES = [0, 500, 1000, 1500, 2500, 5000]
SEVERITY_LABEL = {0: "None", 1: "Minor", 2: "Moderate", 3: "Severe"}

# Results of the four evaluation runs, as recorded in the README (Run 1 had no correctness metric yet)
EVALUATION_PROGRESS = [
    {"run": "Run 1", "run_order": 1, "agent_correctness": None, "baseline_correctness": None,
     "agent_grounding": 1.000, "tool_selection_accuracy": 0.917, "avg_tokens": 6498, "issues_found": 3,
     "key_change": "Baseline evaluation; found prompt-override and efficiency gaps"},
    {"run": "Run 2", "run_order": 2, "agent_correctness": 0.875, "baseline_correctness": 0.278,
     "agent_grounding": 0.977, "tool_selection_accuracy": 1.000, "avg_tokens": 5389, "issues_found": 3,
     "key_change": "Correctness metric added; found a NULL-handling bug in the risk tool"},
    {"run": "Run 3", "run_order": 3, "agent_correctness": 0.889, "baseline_correctness": 0.278,
     "agent_grounding": 1.000, "tool_selection_accuracy": 0.917, "avg_tokens": 4585, "issues_found": 3,
     "key_change": "Lane/carrier filters; found run-to-run variance on carrier decisions"},
    {"run": "Run 4 (x3)", "run_order": 4, "agent_correctness": 1.000, "baseline_correctness": 0.306,
     "agent_grounding": 1.000, "tool_selection_accuracy": 1.000, "avg_tokens": 4852, "issues_found": 0,
     "key_change": "12/12 scenarios passed all 3 repetitions"},
]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_CONN = None


def get_connection():
    global _CONN
    if _CONN is None:
        import pymssql
        missing = [k for k in ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD") if not os.environ.get(k)]
        if missing:
            sys.exit(f"Set these environment variables first: {', '.join(missing)}")
        server = os.environ["SQL_SERVER"].strip()
        if not server.endswith(".database.windows.net"):
            server += ".database.windows.net"
        _CONN = pymssql.connect(server=server, user=os.environ["SQL_USER"], password=os.environ["SQL_PASSWORD"],
                                database=os.environ["SQL_DATABASE"], port=1433, tds_version="7.4",
                                login_timeout=30, timeout=60)
    return _CONN


def run_query(sql, params=()):
    cur = get_connection().cursor()
    cur.execute(sql, tuple(params))
    cols = [c[0] for c in cur.description]
    rows = [[float(v) if isinstance(v, Decimal) else v for v in r] for r in cur.fetchall()]
    return pd.DataFrame(rows, columns=cols)


def safe_query(sql, label):
    try:
        return run_query(sql)
    except Exception as exc:
        print(f"  ! {label}: not available ({str(exc)[:100]})")
        return pd.DataFrame()


def to_date(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c]).dt.date
    return df


def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def late_penalty_used():
    try:
        with open(MODEL_META) as f:
            return float(json.load(f).get("late_penalty_usd", 1000.0))
    except Exception:
        return 1000.0


def active_shipments(penalty):
    df = run_query(
        "SELECT p.shipment_id, s.status, s.ship_date, s.planned_delivery_date, l.lane_id, l.lane_name, "
        "l.origin_city, l.dest_city, l.distance_miles, l.is_port_origin, c.carrier_id, c.carrier_name, "
        "s.equipment_type, s.weight_lbs, s.quoted_cost_usd, p.risk_tier, p.delay_probability, "
        "p.model_delay_probability, p.is_overdue, p.days_overdue, p.weather_severity_used, p.expected_delay_days, "
        "p.expected_exposure_usd, p.predicted_cost_usd, p.recommended_action, p.alt_carrier_id, "
        "p.alt_delay_probability, p.expected_savings_usd, p.model_version, p.predicted_at_utc "
        "FROM freight.shipment_predictions p "
        "JOIN freight.shipments s ON s.shipment_id = p.shipment_id "
        "JOIN freight.lanes l ON l.lane_id = s.lane_id "
        "JOIN freight.carriers c ON c.carrier_id = s.carrier_id")
    carriers = run_query("SELECT carrier_id, carrier_name FROM freight.carriers")
    names = dict(zip(carriers["carrier_id"], carriers["carrier_name"]))
    num(df, ["delay_probability", "model_delay_probability", "is_overdue", "days_overdue", "weather_severity_used",
             "expected_delay_days", "expected_exposure_usd", "predicted_cost_usd", "alt_delay_probability",
             "expected_savings_usd", "quoted_cost_usd", "weight_lbs", "distance_miles", "is_port_origin"])
    to_date(df, ["ship_date", "planned_delivery_date"])

    df["is_overdue"] = df["is_overdue"].fillna(0).astype(int)
    df["days_overdue"] = df["days_overdue"].fillna(0).astype(int)
    df["risk_group"] = df.apply(lambda r: "Already late" if r["is_overdue"] == 1
                                else ("Predicted at risk" if r["risk_tier"] in ("High", "Medium") else "Low risk"),
                                axis=1)
    df["risk_tier_order"] = df["risk_tier"].map({"High": 1, "Medium": 2, "Low": 3})
    week_end = TODAY + timedelta(days=6)

    def window(r):
        if r["is_overdue"] == 1:
            return "Overdue"
        d = r["planned_delivery_date"]
        if d <= week_end:
            return "Next 7 days"
        if d <= week_end + timedelta(days=7):
            return "8-14 days"
        return "Later"
    df["delivery_window"] = df.apply(window, axis=1)
    df["delivery_window_order"] = df["delivery_window"].map({"Overdue": 1, "Next 7 days": 2, "8-14 days": 3, "Later": 4})
    df["weather_label"] = df["weather_severity_used"].map(SEVERITY_LABEL)
    df["late_penalty_usd"] = penalty
    df["business_cost_of_lateness_usd"] = (df["delay_probability"] * penalty).round(2)
    df["total_exposure_usd"] = (df["expected_exposure_usd"] + df["business_cost_of_lateness_usd"]).round(2)
    df["alt_carrier_name"] = df["alt_carrier_id"].map(names)
    df["has_switch"] = df["alt_carrier_id"].notna().astype(int)
    df["freight_cost_change_usd"] = -df["expected_savings_usd"]
    df["risk_reduction_pts"] = ((df["delay_probability"] - df["alt_delay_probability"]) * 100).round(1)
    df["delay_probability_pct"] = (df["delay_probability"] * 100).round(1)
    df["action_type"] = df["recommended_action"].fillna("").map(
        lambda a: "Switch carrier" if a.startswith("Switch to") else
        "Chase overdue" if "past planned delivery" in a else
        "Proactive ETA check" if "in transit" in a.lower() else
        "Build delivery buffer" if "buffer" in a else
        "Monitor" if a.startswith("Monitor") else "No action")
    df["predicted_at_utc"] = df["predicted_at_utc"].astype(str).str[:16]
    return df.sort_values(["risk_tier_order", "delay_probability"], ascending=[True, False])


def shipment_history():
    df = run_query(
        "SELECT s.shipment_id, s.ship_date, s.actual_delivery_date, l.lane_name, c.carrier_name, c.mode, "
        "s.equipment_type, s.weight_lbs, s.planned_transit_days, s.actual_transit_days, s.delay_days, s.is_delayed, "
        "s.weather_severity, s.port_congestion_index, s.diesel_price_usd_gal, s.quoted_cost_usd, "
        "s.accessorial_cost_usd, s.actual_cost_usd "
        "FROM freight.shipments s JOIN freight.lanes l ON l.lane_id = s.lane_id "
        "JOIN freight.carriers c ON c.carrier_id = s.carrier_id WHERE s.status = 'Delivered'")
    num(df, ["weight_lbs", "planned_transit_days", "actual_transit_days", "delay_days", "is_delayed",
             "weather_severity", "port_congestion_index", "diesel_price_usd_gal", "quoted_cost_usd",
             "accessorial_cost_usd", "actual_cost_usd"])
    to_date(df, ["ship_date", "actual_delivery_date"])
    df["ship_month"] = pd.to_datetime(df["ship_date"]).dt.to_period("M").dt.to_timestamp().dt.date
    df["on_time"] = 1 - df["is_delayed"].astype(int)
    df["cost_overrun_usd"] = (df["actual_cost_usd"] - df["quoted_cost_usd"]).round(2)
    df["weather_label"] = df["weather_severity"].map(SEVERITY_LABEL)
    return df


def carrier_lane_perf():
    lanes = run_query("SELECT carrier_id, carrier_name, lane_id, lane_name, shipments, on_time_rate, avg_delay_days, "
                      "avg_actual_cost_usd, avg_cost_overrun_usd FROM freight.v_carrier_lane_performance")
    overall = run_query("SELECT carrier_id, COUNT(*) AS carrier_shipments, "
                        "1.0 - AVG(CAST(is_delayed AS FLOAT)) AS carrier_on_time_rate "
                        "FROM freight.shipments WHERE status = 'Delivered' GROUP BY carrier_id")
    num(lanes, ["shipments", "on_time_rate", "avg_delay_days", "avg_actual_cost_usd", "avg_cost_overrun_usd"])
    num(overall, ["carrier_shipments", "carrier_on_time_rate"])
    df = lanes.merge(overall, on="carrier_id", how="left")
    df["vs_carrier_overall_pts"] = ((df["on_time_rate"] - df["carrier_on_time_rate"]) * 100).round(1)
    df["on_time_pct"] = (df["on_time_rate"] * 100).round(1)
    df["flag"] = df.apply(lambda r: "Weak lane" if r["vs_carrier_overall_pts"] <= -15 and r["shipments"] >= 15
                          else "", axis=1)
    return df.sort_values("on_time_rate")


def route_conditions():
    df = safe_query("SELECT r.lane_id, l.lane_name, r.forecast_date, r.weather_severity_forecast, r.precipitation_mm, "
                    "r.snowfall_cm, r.wind_gust_kmh, r.route_distance_miles, r.route_duration_hours, r.as_of_utc "
                    "FROM freight.route_conditions r JOIN freight.lanes l ON l.lane_id = r.lane_id", "route conditions")
    if df.empty:
        return df
    num(df, ["weather_severity_forecast", "precipitation_mm", "snowfall_cm", "wind_gust_kmh",
             "route_distance_miles", "route_duration_hours"])
    to_date(df, ["forecast_date"])
    df["weather_label"] = df["weather_severity_forecast"].map(SEVERITY_LABEL)
    df["period"] = df["forecast_date"].map(lambda d: "Forecast" if d >= TODAY else "Observed (past 7 days)")
    df["as_of_utc"] = df["as_of_utc"].astype(str).str[:16]
    return df.sort_values(["lane_id", "forecast_date"])


def diesel_prices():
    df = safe_query("SELECT week_start, diesel_price_usd_gal, source FROM freight.diesel_prices_eia", "EIA diesel")
    if df.empty:
        df = run_query("SELECT week_start, diesel_price_usd_gal, data_source AS source FROM freight.fuel_prices")
        df["source"] = "SIMULATED"
    num(df, ["diesel_price_usd_gal"])
    to_date(df, ["week_start"])
    df["fuel_surcharge_usd_per_mile"] = ((df["diesel_price_usd_gal"] - 1.25).clip(lower=0) / 6.5).round(3)
    return df.sort_values("week_start")


def switch_sensitivity():
    rows = []
    for penalty in SENSITIVITY_PENALTIES:
        args = {"risk_tiers": ["High", "Medium"], "late_penalty_usd": penalty, "limit": 2}
        try:
            proc = subprocess.run([sys.executable, AGENT_SCRIPT, "--tool", "calculate_risk_and_recommendation",
                                   "--args", json.dumps(args)], capture_output=True, text=True, timeout=180)
            line = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")][-1]
            r = json.loads(line)
            if "error" in r:
                raise RuntimeError(r["error"])
            t = r["totals"]
            rows.append({"late_penalty_usd": penalty,
                         "carrier_switches": t["carrier_switches_recommended"],
                         "freight_cost_change_usd": t["freight_cost_change_if_all_switches_usd"],
                         "expected_business_cost_of_lateness_usd": t["expected_business_cost_of_lateness_usd"],
                         "expected_extra_carrier_charges_usd": t["expected_extra_carrier_charges_usd"],
                         "shipments_evaluated": r["shipments_evaluated"]})
        except Exception as exc:
            print(f"  ! sensitivity at ${penalty}: {str(exc)[:120]}")
    return pd.DataFrame(rows)


def model_evaluation():
    df = safe_query("SELECT model_version, model, metric, model_value, baseline_value, baseline, test_rows, "
                    "test_delay_rate FROM freight.model_evaluation", "model evaluation")
    num(df, ["model_value", "baseline_value", "test_rows", "test_delay_rate"])
    if not df.empty:
        df["improvement_vs_baseline"] = (df["model_value"] - df["baseline_value"]).round(4)
    return df


def agent_evaluation():
    df = safe_query("SELECT batch_id, evaluated_at_utc, mode, scenario, scenario_group, question, tools_called, "
                    "tool_calls, tool_selection_ok, arguments_ok, completed, turns, tokens, latency_ms, numeric_claims, "
                    "unsupported_claims, grounding_rate, correctness, missing_facts, repetition, passed, answer "
                    "FROM freight.agent_evaluation", "agent evaluation")
    if df.empty:
        return df
    num(df, ["tool_calls", "turns", "tokens", "latency_ms", "numeric_claims", "unsupported_claims",
             "grounding_rate", "correctness", "repetition"])
    for c in ("tool_selection_ok", "arguments_ok", "completed", "passed"):
        df[c] = df[c].map(lambda v: None if v is None else int(bool(v)))
    df["evaluated_at_utc"] = df["evaluated_at_utc"].astype(str).str[:16]
    latest = df["batch_id"].max()
    df["is_latest_batch"] = (df["batch_id"] == latest).astype(int)
    df["mode_label"] = df["mode"].map({"agent": "Agent (with tools)", "baseline_no_tools": "Same model, no tools"})
    df["latency_s"] = (df["latency_ms"] / 1000).round(2)
    df["answer"] = df["answer"].astype(str).str.slice(0, 1500)
    return df


def agent_runs():
    df = safe_query("SELECT run_id, started_at_utc, scenario, user_question, turns_used, hit_turn_cap, completed, "
                    "total_tokens, latency_ms FROM freight.agent_runs", "agent runs")
    if df.empty:
        return df
    num(df, ["turns_used", "total_tokens", "latency_ms"])
    for c in ("hit_turn_cap", "completed"):
        df[c] = df[c].map(lambda v: None if v is None else int(bool(v)))
    df["started_at_utc"] = pd.to_datetime(df["started_at_utc"])
    df["run_date"] = df["started_at_utc"].dt.date
    df["started_at_utc"] = df["started_at_utc"].astype(str).str[:16]
    df["run_type"] = df["scenario"].fillna("").map(lambda s: "Evaluation" if str(s).startswith("EVAL") else "Interactive")
    df["latency_s"] = (df["latency_ms"] / 1000).round(2)
    return df


# ---------------------------------------------------------------------------
# Excel writer (one named table per sheet)
# ---------------------------------------------------------------------------
def write_workbook(tables, path):
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            if df is None or df.empty:
                df = pd.DataFrame({"note": [f"No data available for {name}"]})
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
            table = Table(displayName=name, ref=ref)
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
            ws.add_table(table)
            for i, col in enumerate(df.columns, start=1):
                lengths = sorted(len(str(v)) for v in df.iloc[:, i - 1].tolist()) or [10]
                p90 = lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))]
                ws.column_dimensions[get_column_letter(i)].width = min(60, max(10, len(str(col)) + 2, p90 + 2))
            ws.freeze_panes = "A2"


def main():
    print("Exporting dashboard data from Azure SQL ...")
    penalty = late_penalty_used()
    tables = {}
    steps = [
        ("ActiveShipments", lambda: active_shipments(penalty)),
        ("ShipmentHistory", shipment_history),
        ("CarrierLanePerf", carrier_lane_perf),
        ("RouteConditions", route_conditions),
        ("DieselPrices", diesel_prices),
        ("SwitchSensitivity", switch_sensitivity),
        ("ModelEvaluation", model_evaluation),
        ("AgentEvaluation", agent_evaluation),
        ("EvaluationProgress", lambda: pd.DataFrame(EVALUATION_PROGRESS)),
        ("AgentRuns", agent_runs),
    ]
    for name, fn in steps:
        df = fn()
        tables[name] = df
        print(f"  {name:<20} {len(df):>6} rows")

    a = tables["ActiveShipments"]
    tables["About"] = pd.DataFrame([
        {"item": "Exported (UTC)", "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")},
        {"item": "As-of date (overdue / delivery windows)", "value": TODAY.isoformat()},
        {"item": "Predictions generated (UTC)", "value": a["predicted_at_utc"].max() if len(a) else ""},
        {"item": "Late-delivery cost assumption (USD)", "value": f"{penalty:,.0f}"},
        {"item": "Synthetic data", "value": "Shipments, carriers (fictional names) and history"},
        {"item": "Live data", "value": "Weather (Open-Meteo), routing (OSRM), diesel (EIA), FX (Frankfurter)"},
        {"item": "Simulated data", "value": "Port congestion index"},
        {"item": "Exposure definition", "value": "Expected extra carrier charges + delay probability x late-delivery cost"},
    ])
    write_workbook(tables, OUT_PATH)
    print(f"\nWrote {OUT_PATH}  ({os.path.getsize(OUT_PATH) / 1024:.0f} KB)")
    print("Next: upload it at app.powerbi.com (see DASHBOARD_GUIDE.md).")


if __name__ == "__main__":
    main()
