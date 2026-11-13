"""
train_and_predict.py
Predictive Freight Cost & Delay Agent - Phase 2: prediction engine

Self-contained (no imports from other project files).
Requires: pandas, numpy, scikit-learn, xgboost
    pip install pandas numpy scikit-learn xgboost

Reads (from the project folder):
    data/shipments.csv, data/lanes.csv, data/carriers.csv
    data/ground_truth.csv              (optional - used ONLY to report the theoretical ceiling)
    external/route_conditions.csv      (optional - live weather from fetch_external_data.py)

What it does:
    1. Time-based split: train on shipments before 2026-04-01, test on later ones
       (the model never sees the "future" - mirrors how it would be used in practice).
    2. Trains two XGBoost models plus a two-part cost model:
         delay classifier     -> probability the shipment is late
         delay-days regressor -> how late, if late
         extra-cost model     -> two-part (hurdle) model built on the delay probability:
                                 P(late) x avg extra when late + P(on time) x avg extra when on time
    3. Compares each model to a simple business baseline (carrier's historical average).
    4. Scores the 100 active shipments using the worst forecast weather on each trip.
    5. Applies deterministic business rules: overdue shipments are High risk; for Booked
       high/medium-risk shipments it tests every alternative carrier on the lane and
       recommends a switch only if it does not raise delay risk AND lowers
       expected cost + P(late) x late-delivery cost (--late-penalty, default $1,000) by $100+.
       Severe forecast weather is never 'Low'.
    6. Writes outputs/ CSVs, models/ files (reused by the agent in Phase 4),
       and outputs/predictions_load.sql to run in VS Code.

Usage:
    python train_and_predict.py
    python train_and_predict.py --as-of 2026-11-08     # override "today"
    python train_and_predict.py --late-penalty 1500    # what a late delivery costs the business
"""

import argparse
import json
import os
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)

MODEL_VERSION = "xgb-v1"
TEST_START = "2026-04-01"
RANDOM_STATE = 42

# Risk tiers on delay probability (overdue shipments are forced to High)
HIGH_THRESHOLD = 0.45
MEDIUM_THRESHOLD = 0.25

# Carrier switch rule (expected value). A late delivery costs more than carrier accessorials:
# customer penalties, expediting, missed sales. LATE_PENALTY_USD is that business cost per late
# shipment - an ASSUMPTION the user sets (--late-penalty). A switch is recommended only if it does not
# raise delay risk and lowers  expected total cost + P(late) x LATE_PENALTY_USD  by MIN_SAVINGS_USD+.
DEFAULT_LATE_PENALTY_USD = 1000.0
MIN_SAVINGS_USD = 100.0

NUMERIC_FEATURES = ["distance_miles", "weight_lbs", "planned_transit_days", "diesel_price_usd_gal",
                    "weather_severity", "port_congestion_index", "is_port_origin",
                    "is_winter_exposed", "is_cross_border", "ship_month", "is_winter_month",
                    "reliability_baseline"]
CATEGORICAL_FEATURES = ["carrier_id", "lane_id", "equipment_type"]

# Domain knowledge encoded as monotonic constraints: +1 = can only raise delay risk, -1 = can only lower it
MONOTONE = {"weather_severity": 1, "port_congestion_index": 1, "reliability_baseline": -1,
            "is_winter_month": 1, "distance_miles": 1}

DATA_DIR, EXT_DIR, OUT_DIR, MODEL_DIR = "data", "external", "outputs", "models"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_data():
    shipments = pd.read_csv(os.path.join(DATA_DIR, "shipments.csv"),
                            parse_dates=["ship_date", "planned_delivery_date", "actual_delivery_date"])
    lanes = pd.read_csv(os.path.join(DATA_DIR, "lanes.csv"))
    carriers = pd.read_csv(os.path.join(DATA_DIR, "carriers.csv"))
    df = shipments.merge(lanes, on="lane_id", how="left").merge(
        carriers[["carrier_id", "carrier_name", "mode", "reliability_baseline"]], on="carrier_id", how="left")
    df["ship_month"] = df["ship_date"].dt.month
    df["is_winter_month"] = df["ship_month"].isin([12, 1, 2]).astype(int)
    return df, lanes, carriers


def build_matrix(df, categories):
    """Numeric features + one-hot categoricals with a FIXED column set (so scoring matches training)."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    for col in CATEGORICAL_FEATURES:
        for value in categories[col]:
            X[f"{col}={value}"] = (df[col] == value).astype(float)
    return X


def attach_live_weather(active, as_of):
    """Worst forecast severity on each trip window from external/route_conditions.csv."""
    path = os.path.join(EXT_DIR, "route_conditions.csv")
    active = active.copy()
    if not os.path.exists(path) or os.path.getsize(path) < 10:
        print("  ! external/route_conditions.csv not found - using lane/month historical weather instead")
        active["weather_severity"] = np.nan
        active["weather_source"] = "HISTORICAL_FALLBACK"
        return active

    rc = pd.read_csv(path, parse_dates=["forecast_date"])
    values, sources = [], []
    for row in active.itertuples(index=False):
        # Trip window: ship date (or today, if already moving) through planned delivery.
        # Overdue shipments are still on the road, so extend to today + 2 days.
        start = max(row.ship_date, pd.Timestamp(as_of)) if row.status == "In Transit" else row.ship_date
        end = max(row.planned_delivery_date, pd.Timestamp(as_of) + pd.Timedelta(days=2))
        window = rc[(rc["lane_id"] == row.lane_id) & (rc["forecast_date"] >= start) & (rc["forecast_date"] <= end)]
        if window.empty:
            window = rc[rc["lane_id"] == row.lane_id]     # beyond forecast range: use what we have
        if window.empty:
            values.append(np.nan)
            sources.append("HISTORICAL_FALLBACK")
        else:
            values.append(int(window["weather_severity_forecast"].max()))
            sources.append("OPEN_METEO_FORECAST")
    active["weather_severity"] = values
    active["weather_source"] = sources
    return active


def fill_weather_fallback(active, history):
    """Where no forecast exists, use the rounded historical mean severity for that lane and month."""
    lane_month = history.groupby(["lane_id", "ship_month"])["weather_severity"].mean()
    lane_only = history.groupby("lane_id")["weather_severity"].mean()
    filled = []
    for row in active.itertuples(index=False):
        if pd.notna(row.weather_severity):
            filled.append(row.weather_severity)
        else:
            v = lane_month.get((row.lane_id, row.ship_month), lane_only.get(row.lane_id, 0))
            filled.append(int(round(v)))
    active = active.copy()
    active["weather_severity"] = filled
    return active


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_models(train, categories):
    X = build_matrix(train, categories)
    y_delay = train["is_delayed"].astype(int)
    pos_weight = (y_delay == 0).sum() / max((y_delay == 1).sum(), 1)

    delay_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
        min_child_weight=3, reg_lambda=2.0, eval_metric="logloss", random_state=RANDOM_STATE,
        monotone_constraints=tuple(MONOTONE.get(c, 0) for c in X.columns))
    delay_clf.fit(X, y_delay)

    late = train[train["is_delayed"] == 1]
    days_reg = xgb.XGBRegressor(
        n_estimators=60, max_depth=1, learning_rate=0.05, subsample=0.8,
        min_child_weight=10, reg_lambda=10.0, random_state=RANDOM_STATE)
    days_reg.fit(build_matrix(late, categories), late["delay_days"].astype(float))

    # Two-part ("hurdle") cost model: extra charges are mostly $0 and spike when a load runs late,
    # so expected extra cost = P(late) * avg extra when late + (1 - P(late)) * avg extra when on time.
    acc_params = {
        "late_mean_usd": float(late["accessorial_cost_usd"].mean()),
        "on_time_mean_usd": float(train.loc[train["is_delayed"] == 0, "accessorial_cost_usd"].mean()),
    }
    return delay_clf, days_reg, acc_params, float(pos_weight)


def expected_accessorial(p, acc_params):
    return p * acc_params["late_mean_usd"] + (1 - p) * acc_params["on_time_mean_usd"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def evaluate(models, train, test, categories, ground_truth):
    delay_clf, days_reg, acc_params, _ = models
    X_test = build_matrix(test, categories)
    y = test["is_delayed"].astype(int).values
    p = delay_clf.predict_proba(X_test)[:, 1]

    # Baseline: each carrier's historical delay rate from the training period
    carrier_rate = train.groupby("carrier_id")["is_delayed"].mean()
    p_base = test["carrier_id"].map(carrier_rate).fillna(train["is_delayed"].mean()).values

    flagged = (p >= MEDIUM_THRESHOLD).astype(int)
    flagged_base = (p_base >= MEDIUM_THRESHOLD).astype(int)

    acc_true = test["accessorial_cost_usd"].astype(float).values
    acc_pred = expected_accessorial(p, acc_params)
    acc_base = test["carrier_id"].map(train.groupby("carrier_id")["accessorial_cost_usd"].mean()).values
    cost_true = test["actual_cost_usd"].astype(float).values
    cost_pred = test["quoted_cost_usd"].values + acc_pred
    cost_base_quote_only = test["quoted_cost_usd"].values

    late = test[test["is_delayed"] == 1]
    days_mae = mean_absolute_error(late["delay_days"], days_reg.predict(build_matrix(late, categories))) \
        if len(late) else float("nan")
    days_base_mae = mean_absolute_error(late["delay_days"], np.full(len(late), train.loc[
        train["is_delayed"] == 1, "delay_days"].mean())) if len(late) else float("nan")

    metrics = [
        # (model, metric, model_value, baseline_value, baseline_description)
        ("delay_classifier", "roc_auc", roc_auc_score(y, p), roc_auc_score(y, p_base), "carrier historical delay rate"),
        ("delay_classifier", "pr_auc", average_precision_score(y, p), average_precision_score(y, p_base), "carrier historical delay rate"),
        ("delay_classifier", "precision_at_medium_plus", precision_score(y, flagged, zero_division=0),
         precision_score(y, flagged_base, zero_division=0), "carrier historical delay rate"),
        ("delay_classifier", "recall_at_medium_plus", recall_score(y, flagged), recall_score(y, flagged_base),
         "carrier historical delay rate"),
        ("delay_classifier", "f1_at_medium_plus", f1_score(y, flagged), f1_score(y, flagged_base),
         "carrier historical delay rate"),
        ("delay_days_regressor", "mae_days", days_mae, days_base_mae, "average delay days"),
        ("accessorial_hurdle", "rmse_usd", rmse(acc_true, acc_pred), rmse(acc_true, acc_base),
         "carrier average accessorial"),
        ("accessorial_hurdle", "portfolio_total_error_pct", abs(acc_pred.sum() - acc_true.sum()) / acc_true.sum(),
         1.0, "quoted cost only (assumes no extra charges)"),
        ("total_cost", "rmse_usd", rmse(cost_true, cost_pred), rmse(cost_true, cost_base_quote_only),
         "quoted cost only"),
        ("total_cost", "mape", float(np.mean(np.abs(cost_pred - cost_true) / cost_true)),
         float(np.mean(np.abs(cost_base_quote_only - cost_true) / cost_true)), "quoted cost only"),
    ]
    if ground_truth is not None:
        gt = test[["shipment_id"]].merge(ground_truth, on="shipment_id", how="left")["true_delay_probability"]
        if gt.notna().all():
            metrics.append(("delay_classifier", "roc_auc_theoretical_ceiling", roc_auc_score(y, gt), None,
                            "true generating probability (synthetic data only)"))

    report = pd.DataFrame(metrics, columns=["model", "metric", "model_value", "baseline_value", "baseline"])
    report["test_rows"] = len(test)
    report["test_delay_rate"] = round(float(y.mean()), 4)
    return report


# ---------------------------------------------------------------------------
# Scoring and recommendations
# ---------------------------------------------------------------------------
def score(models, frame, categories):
    delay_clf, days_reg, acc_params, _ = models
    X = build_matrix(frame, categories)
    p = delay_clf.predict_proba(X)[:, 1]
    days_if_late = np.clip(days_reg.predict(X), 1, None)
    acc = expected_accessorial(p, acc_params)
    return p, p * days_if_late, acc


def tier(p, overdue, weather):
    """Model probability sets the tier; two deterministic guardrails can only raise it."""
    if overdue or p >= HIGH_THRESHOLD:
        return "High"
    if p >= MEDIUM_THRESHOLD or weather >= 3:      # severe forecast weather is never 'Low'
        return "Medium"
    return "Low"


def estimate_alt_quote(history, carrier_id, lane_id, equipment, miles, fallback_rate):
    """Recent median quoted cost per mile for this carrier on this lane (last 12 months of history)."""
    recent = history[(history["carrier_id"] == carrier_id) & (history["lane_id"] == lane_id)]
    recent = recent[recent["ship_date"] >= recent["ship_date"].max() - pd.Timedelta(days=365)] if len(recent) else recent
    same_equipment = recent[recent["equipment_type"] == equipment]
    pool = same_equipment if len(same_equipment) >= 3 else recent
    if len(pool) >= 3:
        return float((pool["quoted_cost_usd"] / pool["distance_miles"]).median() * miles)
    return float(fallback_rate * miles)


def recommend(active, history, carriers, models, categories, as_of, late_penalty):
    p, exp_days, acc = score(models, active, categories)
    active = active.copy()
    active["delay_probability"] = p
    active["expected_delay_days"] = exp_days
    active["predicted_accessorial_usd"] = acc
    active["predicted_cost_usd"] = active["quoted_cost_usd"] + acc
    active["expected_exposure_usd"] = acc
    active["is_overdue"] = ((active["status"] == "In Transit") &
                            (active["planned_delivery_date"] < pd.Timestamp(as_of))).astype(int)
    # Overdue shipments are already late: the pre-trip probability no longer applies.
    # Keep the model's score in model_delay_probability for transparency; set the operational
    # probability to 1.0 and price exposure as a late shipment.
    active["model_delay_probability"] = active["delay_probability"]
    active["days_overdue"] = np.where(active["is_overdue"] == 1,
                                      (pd.Timestamp(as_of) - active["planned_delivery_date"]).dt.days, 0)
    od = active["is_overdue"] == 1
    active.loc[od, "delay_probability"] = 1.0
    active.loc[od, "expected_delay_days"] = np.maximum(active.loc[od, "expected_delay_days"],
                                                       active.loc[od, "days_overdue"]).astype(float)
    late_extra = models[2]["late_mean_usd"]
    active.loc[od, "expected_exposure_usd"] = late_extra
    active.loc[od, "predicted_cost_usd"] = active.loc[od, "quoted_cost_usd"] + late_extra

    active["risk_tier"] = [tier(pi, od, w) for pi, od, w in
                           zip(active["delay_probability"], active["is_overdue"], active["weather_severity"])]

    lane_carriers = history.groupby("lane_id")["carrier_id"].unique().to_dict()
    carrier_info = carriers.set_index("carrier_id")
    alt_cols = {"alt_carrier_id": [], "alt_delay_probability": [], "alt_expected_cost_usd": [],
                "expected_savings_usd": [], "recommended_action": []}

    for idx, row in active.iterrows():
        current_total = row["predicted_cost_usd"]
        best = None
        if row["status"] == "Booked" and row["risk_tier"] in ("High", "Medium"):
            for alt in lane_carriers.get(row["lane_id"], []):
                if alt == row["carrier_id"]:
                    continue
                cand = active.loc[[idx]].copy()
                cand["carrier_id"] = alt
                cand["reliability_baseline"] = carrier_info.loc[alt, "reliability_baseline"]
                if carrier_info.loc[alt, "mode"] == "Intermodal":
                    continue          # truck -> rail changes the delivery date; out of scope for v1
                if row["equipment_type"] == "Intermodal Container":
                    # rail -> truck: score the alternative as a dry-van truckload with truck transit time
                    cand["equipment_type"] = "Dry Van"
                    cand["planned_transit_days"] = max(1, int(np.ceil(row["distance_miles"] / 500))
                                                       + int(row["is_cross_border"]))
                ap, _, aacc = score(models, cand, categories)
                alt_quote = estimate_alt_quote(history, alt, row["lane_id"], cand["equipment_type"].iloc[0],
                                               row["distance_miles"],
                                               carrier_info.loc[alt, "base_rate_per_mile"])
                alt_total = alt_quote + float(aacc[0])
                savings = current_total - alt_total
                risk_drop = row["delay_probability"] - float(ap[0])
                # Never recommend a carrier that raises delay risk on a shipment flagged for risk
                eff_current = current_total + row["delay_probability"] * late_penalty
                eff_alt = alt_total + float(ap[0]) * late_penalty
                net_benefit = eff_current - eff_alt
                if risk_drop >= 0 and net_benefit >= MIN_SAVINGS_USD and (best is None or net_benefit > best[5]):
                    best = (alt, float(ap[0]), alt_total, savings, risk_drop, net_benefit)

        if best:
            alt, ap, alt_total, savings, risk_drop, net_benefit = best
            alt_name = carrier_info.loc[alt, "carrier_name"]
            action = (f"Switch to {alt_name} ({alt}): delay risk {row['delay_probability']:.0%} -> {ap:.0%}, "
                      f"freight cost change {-savings:+,.0f} USD, net benefit {net_benefit:,.0f} USD "
                      f"incl. {late_penalty:,.0f} USD late-delivery cost.")
            alt_cols["alt_carrier_id"].append(alt)
            alt_cols["alt_delay_probability"].append(round(ap, 4))
            alt_cols["alt_expected_cost_usd"].append(round(alt_total, 2))
            alt_cols["expected_savings_usd"].append(round(savings, 2))
        else:
            if row["is_overdue"]:
                d = int(row["days_overdue"])
                action = (f"{d} day{'s' if d != 1 else ''} past planned delivery (pre-trip model risk "
                          f"{row['model_delay_probability']:.0%}): contact carrier for updated ETA and notify the customer.")
            elif row["risk_tier"] == "High" and row["status"] == "In Transit":
                action = "High risk in transit: request proactive ETA check-in and pre-alert the consignee."
            elif row["risk_tier"] == "High":
                action = "High risk, no better carrier on this lane: build buffer into delivery commitment."
            elif row["risk_tier"] == "Medium":
                action = "Monitor: re-check weather and carrier status 24 hours before pickup or delivery."
            else:
                action = "No action needed."
            for k in ["alt_carrier_id", "alt_delay_probability", "alt_expected_cost_usd", "expected_savings_usd"]:
                alt_cols[k].append(None)
        alt_cols["recommended_action"].append(action)

    for k, v in alt_cols.items():
        active[k] = v
    return active


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def sql_literal(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "NULL"
    if isinstance(v, (pd.Timestamp, datetime, date)):
        v = pd.Timestamp(v).strftime("%Y-%m-%d")
    return "N'" + str(v).replace("'", "''") + "'"


def insert_blocks(table, df, batch=200):
    cols = ", ".join(df.columns)
    rows = ["(" + ", ".join(sql_literal(v) for v in rec) + ")"
            for rec in df.astype(object).itertuples(index=False, name=None)]
    out = []
    for i in range(0, len(rows), batch):
        out += [f"INSERT INTO {table} ({cols}) VALUES\n" + ",\n".join(rows[i:i + batch]) + ";", "GO"]
    return out


def write_sql(predictions, report, path):
    lines = [
        "/* Phase 2 predictions - generated by train_and_predict.py",
        f"   Model version: {MODEL_VERSION}   Generated (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M} */",
        "SET NOCOUNT ON;", "GO",
        # Extra columns on shipment_predictions (added once; safe to re-run)
        "IF COL_LENGTH('freight.shipment_predictions', 'weather_severity_used') IS NULL\n"
        "ALTER TABLE freight.shipment_predictions ADD\n"
        "    weather_severity_used TINYINT NULL, weather_source VARCHAR(40) NULL, is_overdue BIT NULL,\n"
        "    alt_carrier_id VARCHAR(10) NULL, alt_delay_probability DECIMAL(5,4) NULL,\n"
        "    alt_expected_cost_usd DECIMAL(12,2) NULL, expected_savings_usd DECIMAL(12,2) NULL;", "GO",
        "IF COL_LENGTH('freight.shipment_predictions', 'model_delay_probability') IS NULL\n"
        "ALTER TABLE freight.shipment_predictions ADD\n"
        "    model_delay_probability DECIMAL(5,4) NULL, days_overdue INT NULL;", "GO",
        "IF OBJECT_ID('freight.model_evaluation') IS NULL\n"
        "CREATE TABLE freight.model_evaluation (\n"
        "    model_version VARCHAR(30) NOT NULL, model VARCHAR(40) NOT NULL, metric VARCHAR(40) NOT NULL,\n"
        "    model_value DECIMAL(12,4) NULL, baseline_value DECIMAL(12,4) NULL, baseline NVARCHAR(100) NULL,\n"
        "    test_rows INT NULL, test_delay_rate DECIMAL(5,4) NULL, evaluated_at_utc DATETIME2 NOT NULL,\n"
        "    CONSTRAINT pk_model_evaluation PRIMARY KEY (model_version, model, metric));", "GO",
        f"DELETE FROM freight.shipment_predictions WHERE model_version = '{MODEL_VERSION}';", "GO",
        f"DELETE FROM freight.model_evaluation WHERE model_version = '{MODEL_VERSION}';", "GO",
    ]
    lines += insert_blocks("freight.shipment_predictions", predictions)
    lines += insert_blocks("freight.model_evaluation", report)
    lines += [
        "/* Validation */",
        f"SELECT risk_tier, COUNT(*) AS shipments, CAST(SUM(expected_exposure_usd) AS DECIMAL(12,0)) AS exposure_usd "
        f"FROM freight.shipment_predictions WHERE model_version = '{MODEL_VERSION}' GROUP BY risk_tier;",
        "SELECT TOP 10 p.shipment_id, s.status, l.lane_name, c.carrier_name, p.delay_probability, "
        "p.model_delay_probability, p.days_overdue, p.weather_severity_used, p.risk_tier, p.recommended_action "
        "FROM freight.shipment_predictions p JOIN freight.shipments s ON s.shipment_id = p.shipment_id "
        "JOIN freight.lanes l ON l.lane_id = s.lane_id JOIN freight.carriers c ON c.carrier_id = s.carrier_id "
        f"WHERE p.model_version = '{MODEL_VERSION}' ORDER BY p.is_overdue DESC, p.delay_probability DESC;",
        "SELECT model, metric, model_value, baseline_value, baseline FROM freight.model_evaluation "
        f"WHERE model_version = '{MODEL_VERSION}' ORDER BY model, metric;",
        "GO",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_models(models, categories, feature_names, training_info):
    os.makedirs(MODEL_DIR, exist_ok=True)
    delay_clf, days_reg, acc_params, _ = models
    delay_clf.save_model(os.path.join(MODEL_DIR, "delay_classifier.json"))
    days_reg.save_model(os.path.join(MODEL_DIR, "delay_days_regressor.json"))
    meta = {"accessorial_hurdle": acc_params, "model_version": MODEL_VERSION, "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES, "categories": categories,
            "feature_names": feature_names, "high_threshold": HIGH_THRESHOLD,
            "medium_threshold": MEDIUM_THRESHOLD, **training_info}
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train freight models and score active shipments.")
    parser.add_argument("--late-penalty", type=float, default=DEFAULT_LATE_PENALTY_USD,
                        help="Business cost of one late delivery in USD (default 1000)")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="'Today' for overdue checks (YYYY-MM-DD)")
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of)

    df, lanes, carriers = load_data()
    history = df[df["status"] == "Delivered"].copy()
    active = df[df["status"] != "Delivered"].copy()
    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}

    train = history[history["ship_date"] < TEST_START]
    test = history[history["ship_date"] >= TEST_START]
    print(f"Training on {len(train)} shipments (before {TEST_START}), testing on {len(test)} later shipments")

    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    ground_truth = pd.read_csv(gt_path) if os.path.exists(gt_path) else None

    # 1) Honest evaluation: models trained on the past, scored on the future
    eval_models = train_models(train, categories)
    report = evaluate(eval_models, train, test, categories, ground_truth)

    # 2) Production models: retrain on ALL history before scoring active shipments
    final_models = train_models(history, categories)
    feature_names = build_matrix(history.head(1), categories).columns.tolist()

    importance = pd.Series(final_models[0].feature_importances_, index=feature_names).sort_values(ascending=False)

    print("Attaching live weather to active shipments ...")
    active = attach_live_weather(active, as_of)
    active = fill_weather_fallback(active, history)
    scored = recommend(active, history, carriers, final_models, categories, as_of, args.late_penalty)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    predictions = pd.DataFrame({
        "shipment_id": scored["shipment_id"],
        "predicted_at_utc": now_utc,
        "model_version": MODEL_VERSION,
        "delay_probability": scored["delay_probability"].astype(float).round(4),
        "model_delay_probability": scored["model_delay_probability"].astype(float).round(4),
        "days_overdue": scored["days_overdue"].astype(int),
        "expected_delay_days": scored["expected_delay_days"].astype(float).round(2),
        "predicted_cost_usd": scored["predicted_cost_usd"].astype(float).round(2),
        "expected_exposure_usd": scored["expected_exposure_usd"].astype(float).round(2),
        "risk_tier": scored["risk_tier"],
        "recommended_action": scored["recommended_action"].str.slice(0, 400),
        "weather_severity_used": scored["weather_severity"].astype(int),
        "weather_source": scored["weather_source"],
        "is_overdue": scored["is_overdue"],
        "alt_carrier_id": scored["alt_carrier_id"],
        "alt_delay_probability": scored["alt_delay_probability"],
        "alt_expected_cost_usd": scored["alt_expected_cost_usd"],
        "expected_savings_usd": scored["expected_savings_usd"],
    })
    report_out = report.copy()
    report_out.insert(0, "model_version", MODEL_VERSION)
    report_out["model_value"] = report_out["model_value"].astype(float).round(4)
    report_out["baseline_value"] = pd.to_numeric(report_out["baseline_value"]).round(4)
    report_out["evaluated_at_utc"] = now_utc

    os.makedirs(OUT_DIR, exist_ok=True)
    predictions.to_csv(os.path.join(OUT_DIR, "shipment_predictions.csv"), index=False)
    report_out.to_csv(os.path.join(OUT_DIR, "model_evaluation.csv"), index=False)
    importance.rename("importance").to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
    write_sql(predictions, report_out, os.path.join(OUT_DIR, "predictions_load.sql"))
    save_models(final_models, categories, feature_names,
                {"trained_rows": int(len(history)), "test_start": TEST_START, "as_of": args.as_of,
                 "late_penalty_usd": args.late_penalty, "min_savings_usd": MIN_SAVINGS_USD})

    # ---- Console summary ----
    print("\n=== Model evaluation (test set: shipments from", TEST_START, "onward) ===")
    for r in report.itertuples(index=False):
        base = "" if r.baseline_value is None or pd.isna(r.baseline_value) else f"   baseline {r.baseline_value:.3f}"
        print(f"  {r.model:<22} {r.metric:<28} {r.model_value:>9.3f}{base}")
    print("\nTop 8 delay drivers (feature importance):")
    for name, val in importance.head(8).items():
        print(f"  {name:<28} {val:.3f}")
    print("\n=== Active shipments ===")
    print(f"  weather source: {scored['weather_source'].value_counts().to_dict()}")
    tiers = predictions.groupby("risk_tier").agg(shipments=("shipment_id", "count"),
                                                 exposure=("expected_exposure_usd", "sum"))
    for t in ["High", "Medium", "Low"]:
        if t in tiers.index:
            print(f"  {t:<7} {int(tiers.loc[t, 'shipments']):>3} shipments   expected exposure ${tiers.loc[t, 'exposure']:,.0f}")
    print(f"  overdue: {int(predictions['is_overdue'].sum())}   carrier switches recommended: "
          f"{int(predictions['alt_carrier_id'].notna().sum())}")
    print(f"\nRun outputs/predictions_load.sql in VS Code to load results into Azure SQL.")


if __name__ == "__main__":
    main()
