"""
generate_freight_data.py
Predictive Freight Cost & Delay Agent - Phase 1: synthetic dataset

Self-contained (no imports from other project files). Requires: numpy, pandas.

Writes five CSVs to ./data/:
    carriers.csv      6 fictional carriers (one Canadian, quoting in CAD)
    lanes.csv         12 lanes, mostly out of Los Angeles / Long Beach / Oakland
    fuel_prices.csv   weekly diesel price series (SIMULATED - replace with EIA API in Phase 3)
    shipments.csv     ~1,200 delivered (historical) + 100 active shipments
    ground_truth.csv  the true delay probability used to generate each historical shipment

Why the data has a planted causal structure:
    Delays and costs are generated from known rules (carrier reliability, weather,
    port congestion, winter on northern lanes, distance, weight, and one weak
    carrier-lane pairing). Because the rules are known, Phase 2 can check whether
    the XGBoost model actually recovers them - a stronger validation story than
    "the model scored well on data I made up".

Usage:
    python generate_freight_data.py            # default seed 42
    python generate_freight_data.py --seed 7
"""

import argparse
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_HISTORICAL = 1200
N_ACTIVE = 100
HIST_START = date(2024, 1, 1)
HIST_END = date(2026, 10, 31)
AS_OF_DATE = date(2026, 11, 7)          # "today" for the active shipments
ACTIVE_SHIP_START = date(2026, 11, 2)
ACTIVE_SHIP_END = date(2026, 11, 13)

# Standard DOE-style fuel surcharge: (diesel - base) / mpg, per mile
FSC_BASE_PRICE = 1.25
FSC_MPG = 6.5
CAD_PER_USD_BASE = 1.36                 # simulated FX level for history

# Fictional carriers. reliability = baseline on-time tendency (lower = more delays)
CARRIERS = [
    # id,   name,                     mode,          country, currency, rate/mi, reliability
    ("C01", "Pacific Crest Freight",  "FTL",         "US", "USD", 2.35, 0.93),
    ("C02", "Redline Logistics",      "FTL",         "US", "USD", 2.05, 0.85),
    ("C03", "Summit Ridge Transport", "FTL",         "US", "USD", 2.20, 0.90),
    ("C04", "Ironhorse Carriers",     "FTL",         "US", "USD", 1.85, 0.78),
    ("C05", "Bluewater Intermodal",   "Intermodal",  "US", "USD", 1.70, 0.88),
    ("C06", "Maple Leaf Haulage",     "FTL",         "CA", "CAD", 3.05, 0.89),
]

# Lanes. Distances are approximate road miles (replace with OSRM / Azure Maps in Phase 3).
# winter_exposed = route crosses mountain passes or northern states (Dec-Feb weather risk)
LANES = [
    # id,   origin,           o_st, o_lat,  o_lon,    dest,             d_st, d_country, d_lat, d_lon,   miles, port, winter
    ("L01", "Los Angeles",    "CA", 34.05, -118.24, "Chicago",        "IL", "US", 41.88,  -87.63, 2015, 0, 1),
    ("L02", "Los Angeles",    "CA", 34.05, -118.24, "Dallas",         "TX", "US", 32.78,  -96.80, 1435, 0, 0),
    ("L03", "Los Angeles",    "CA", 34.05, -118.24, "Phoenix",        "AZ", "US", 33.45, -112.07,  373, 0, 0),
    ("L04", "Los Angeles",    "CA", 34.05, -118.24, "Denver",         "CO", "US", 39.74, -104.99, 1020, 0, 1),
    ("L05", "Los Angeles",    "CA", 34.05, -118.24, "Atlanta",        "GA", "US", 33.75,  -84.39, 2175, 0, 0),
    ("L06", "Los Angeles",    "CA", 34.05, -118.24, "Seattle",        "WA", "US", 47.61, -122.33, 1135, 0, 1),
    ("L07", "Long Beach",     "CA", 33.77, -118.19, "Memphis",        "TN", "US", 35.15,  -90.05, 1800, 1, 0),
    ("L08", "Long Beach",     "CA", 33.77, -118.19, "Columbus",       "OH", "US", 39.96,  -83.00, 2240, 1, 1),
    ("L09", "Oakland",        "CA", 37.80, -122.27, "Salt Lake City", "UT", "US", 40.76, -111.89,  740, 1, 1),
    ("L10", "Chicago",        "IL", 41.88,  -87.63, "New York",       "NY", "US", 40.71,  -74.01,  790, 0, 1),
    ("L11", "Dallas",         "TX", 32.78,  -96.80, "Atlanta",        "GA", "US", 33.75,  -84.39,  780, 0, 0),
    ("L12", "Los Angeles",    "CA", 34.05, -118.24, "Toronto",        "ON", "CA", 43.65,  -79.38, 2520, 0, 1),
]

# Which carriers run which lanes
LANE_CARRIERS = {
    "L01": ["C01", "C02", "C04", "C05"],
    "L02": ["C01", "C02", "C03", "C04"],
    "L03": ["C02", "C03", "C04"],
    "L04": ["C01", "C03", "C04"],
    "L05": ["C01", "C02", "C05"],
    "L06": ["C01", "C03", "C04"],
    "L07": ["C02", "C04", "C05"],
    "L08": ["C01", "C03", "C05"],
    "L09": ["C02", "C03", "C04"],
    "L10": ["C01", "C02", "C03", "C06"],
    "L11": ["C02", "C03", "C04"],
    "L12": ["C01", "C06"],
}

# One deliberately weak carrier-lane pairing, so "history of delays on this route"
# is a real signal distinct from a carrier's overall reliability.
WEAK_PAIRS = {("C04", "L01"): 1.1, ("C02", "L06"): 0.8}

EQUIPMENT_FTL = ["Dry Van", "Dry Van", "Dry Van", "Reefer", "Flatbed"]
EQUIPMENT_RATE_MULT = {"Dry Van": 1.00, "Reefer": 1.15, "Flatbed": 1.10, "Intermodal Container": 1.00}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def planned_transit_days(miles, mode, cross_border):
    if mode == "Intermodal":
        days = math.ceil(miles / 350) + 1
    else:
        days = math.ceil(miles / 500)
    return max(1, days + (1 if cross_border else 0))


def week_start(d):
    return d - timedelta(days=d.weekday())


def build_fuel_prices(rng):
    """Weekly diesel price, bounded random walk with mild seasonality. SIMULATED."""
    weeks = []
    d = week_start(HIST_START)
    end = week_start(ACTIVE_SHIP_END)
    price = 3.95
    while d <= end:
        seasonal = 0.06 * math.sin(2 * math.pi * (d.timetuple().tm_yday / 365.0))
        price += rng.normal(0, 0.035) + 0.15 * (3.85 - price) * 0.1
        price = min(max(price, 3.30), 4.70)
        weeks.append((d, round(price + seasonal, 3)))
        d += timedelta(days=7)
    return pd.DataFrame(weeks, columns=["week_start", "diesel_price_usd_gal"])


def port_congestion_index(rng, d):
    """0-1 congestion index for port-origin lanes: peak season Aug-Oct, plus noise."""
    month_effect = {8: 0.25, 9: 0.30, 10: 0.25, 11: 0.10, 3: 0.10, 4: 0.10}.get(d.month, 0.0)
    year_effect = 0.10 if d.year == 2024 and d.month <= 6 else 0.0   # early-2024 backlog
    value = 0.25 + month_effect + year_effect + rng.normal(0, 0.12)
    return round(float(min(max(value, 0.0), 1.0)), 3)


def sample_weather(rng, d, winter_exposed):
    """0 = none, 1 = minor, 2 = moderate, 3 = severe."""
    winter = d.month in (12, 1, 2)
    if winter and winter_exposed:
        p = [0.40, 0.28, 0.20, 0.12]
    elif winter or d.month in (3, 11):
        p = [0.60, 0.25, 0.11, 0.04]
    else:
        p = [0.75, 0.18, 0.06, 0.01]
    return int(rng.choice([0, 1, 2, 3], p=p))


def cad_per_usd(rng, d):
    drift = 0.03 * math.sin((d - HIST_START).days / 180.0)
    return round(CAD_PER_USD_BASE + drift + rng.normal(0, 0.008), 4)


def true_delay_probability(reliability, weather, port, congestion, winter_exposed,
                           month, miles, weight, pair_penalty):
    z = (-3.55
         + 14.0 * (0.90 - reliability)
         + 1.00 * weather
         + 1.80 * port * congestion
         + (0.70 if (winter_exposed and month in (12, 1, 2)) else 0.0)
         + 0.25 * (miles / 1000.0)
         + (0.40 if weight > 40000 else 0.0)
         + pair_penalty)
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate(seed):
    rng = np.random.default_rng(seed)

    carriers = pd.DataFrame(CARRIERS, columns=[
        "carrier_id", "carrier_name", "mode", "country", "quote_currency",
        "base_rate_per_mile", "reliability_baseline"])
    carriers["data_source"] = "SYNTHETIC"

    lanes = pd.DataFrame(LANES, columns=[
        "lane_id", "origin_city", "origin_state", "origin_lat", "origin_lon",
        "dest_city", "dest_state", "dest_country", "dest_lat", "dest_lon",
        "distance_miles", "is_port_origin", "is_winter_exposed"])
    lanes["is_cross_border"] = (lanes["dest_country"] != "US").astype(int)
    lanes["lane_name"] = lanes["origin_city"] + ", " + lanes["origin_state"] + " -> " + \
        lanes["dest_city"] + ", " + lanes["dest_state"]

    fuel = build_fuel_prices(rng)
    fuel["data_source"] = "SIMULATED"
    fuel_lookup = dict(zip(fuel["week_start"], fuel["diesel_price_usd_gal"]))

    carrier_by_id = carriers.set_index("carrier_id").to_dict("index")
    lane_by_id = lanes.set_index("lane_id").to_dict("index")
    # Lane popularity: LA -> Chicago / Dallas and port lanes are busiest
    lane_ids = list(lane_by_id.keys())
    lane_weights = np.array([14, 11, 9, 7, 8, 7, 10, 9, 7, 7, 6, 5], dtype=float)
    lane_weights /= lane_weights.sum()

    hist_days = (HIST_END - HIST_START).days
    rows, truth = [], []

    def make_shipment(idx, ship_date, is_active):
        lane_id = str(rng.choice(lane_ids, p=lane_weights))
        lane = lane_by_id[lane_id]
        carrier_id = str(rng.choice(LANE_CARRIERS[lane_id]))
        carrier = carrier_by_id[carrier_id]
        miles = lane["distance_miles"]

        equipment = "Intermodal Container" if carrier["mode"] == "Intermodal" \
            else str(rng.choice(EQUIPMENT_FTL))
        weight = int(rng.normal(32000, 7000))
        weight = int(min(max(weight, 8000), 44500))
        if equipment == "Reefer":
            weight = min(weight, 42000)

        p_days = planned_transit_days(miles, carrier["mode"], lane["is_cross_border"])
        planned_delivery = ship_date + timedelta(days=p_days)
        diesel = fuel_lookup[week_start(ship_date)]
        congestion = port_congestion_index(rng, ship_date) if lane["is_port_origin"] else 0.0

        # Currency handling: only the Canadian carrier quotes in CAD
        currency = carrier["quote_currency"]
        fx_to_usd = 1.0 if currency == "USD" else round(1.0 / cad_per_usd(rng, ship_date), 5)

        # Freight cost (quoted): linehaul + fuel surcharge + drayage for port intermodal
        market_noise = rng.normal(1.0, 0.06)
        linehaul_native = miles * carrier["base_rate_per_mile"] * EQUIPMENT_RATE_MULT[equipment] * market_noise
        fsc_usd = miles * max(0.0, diesel - FSC_BASE_PRICE) / FSC_MPG
        weight_adj = 0.08 if weight > 40000 else 0.0
        drayage_usd = 350.0 if (carrier["mode"] == "Intermodal" and lane["is_port_origin"]) else 0.0
        linehaul_usd = linehaul_native * fx_to_usd
        quoted_usd = linehaul_usd * (1 + weight_adj) + fsc_usd + drayage_usd
        quoted_native = quoted_usd / fx_to_usd

        record = {
            "shipment_id": f"SHP-{idx:06d}",
            "lane_id": lane_id,
            "carrier_id": carrier_id,
            "equipment_type": equipment,
            "weight_lbs": weight,
            "ship_date": ship_date,
            "planned_transit_days": p_days,
            "planned_delivery_date": planned_delivery,
            "diesel_price_usd_gal": diesel,
            "port_congestion_index": congestion,
            "quote_currency": currency,
            "fx_rate_to_usd": fx_to_usd,
            "quoted_cost_native": round(quoted_native, 2),
            "quoted_cost_usd": round(quoted_usd, 2),
        }

        if is_active:
            record.update({
                "status": "In Transit" if ship_date <= AS_OF_DATE else "Booked",
                "weather_severity": None,          # filled from live forecast in Phase 3
                "actual_delivery_date": None,
                "actual_transit_days": None,
                "delay_days": None,
                "is_delayed": None,
                "accessorial_cost_usd": None,
                "actual_cost_usd": None,
            })
            return record, None

        weather = sample_weather(rng, ship_date, lane["is_winter_exposed"])
        pair_penalty = WEAK_PAIRS.get((carrier_id, lane_id), 0.0)
        p_delay = true_delay_probability(
            carrier["reliability_baseline"], weather, lane["is_port_origin"], congestion,
            lane["is_winter_exposed"], ship_date.month, miles, weight, pair_penalty)
        delayed = bool(rng.random() < p_delay)
        delay_days = 0
        if delayed:
            delay_days = int(min(1 + rng.poisson(0.4 + 0.35 * weather), 6))

        # Accessorials: detention/layover when delayed, plus occasional minor charges
        accessorial = 0.0
        if delayed:
            accessorial += delay_days * rng.uniform(150, 225)
        if rng.random() < 0.10:
            accessorial += rng.uniform(75, 250)
        actual_cost = quoted_usd + accessorial + rng.normal(0, 0.01 * quoted_usd)

        actual_transit = p_days + delay_days
        record.update({
            "status": "Delivered",
            "weather_severity": weather,
            "actual_delivery_date": ship_date + timedelta(days=actual_transit),
            "actual_transit_days": actual_transit,
            "delay_days": delay_days,
            "is_delayed": int(delayed),
            "accessorial_cost_usd": round(accessorial, 2),
            "actual_cost_usd": round(actual_cost, 2),
        })
        return record, {"shipment_id": record["shipment_id"],
                        "true_delay_probability": round(p_delay, 4)}

    for i in range(1, N_HISTORICAL + 1):
        ship_date = HIST_START + timedelta(days=int(rng.integers(0, hist_days + 1)))
        rec, gt = make_shipment(i, ship_date, is_active=False)
        rows.append(rec)
        truth.append(gt)

    active_days = (ACTIVE_SHIP_END - ACTIVE_SHIP_START).days
    for j in range(1, N_ACTIVE + 1):
        ship_date = ACTIVE_SHIP_START + timedelta(days=int(rng.integers(0, active_days + 1)))
        rec, _ = make_shipment(N_HISTORICAL + j, ship_date, is_active=True)
        rows.append(rec)

    shipments = pd.DataFrame(rows).sort_values(["ship_date", "shipment_id"]).reset_index(drop=True)
    for col in ["weather_severity", "actual_transit_days", "delay_days", "is_delayed"]:
        shipments[col] = shipments[col].astype("Int64")
    shipments["data_source"] = "SYNTHETIC"
    column_order = [
        "shipment_id", "lane_id", "carrier_id", "equipment_type", "weight_lbs", "status",
        "ship_date", "planned_transit_days", "planned_delivery_date",
        "actual_delivery_date", "actual_transit_days", "delay_days", "is_delayed",
        "diesel_price_usd_gal", "weather_severity", "port_congestion_index",
        "quote_currency", "fx_rate_to_usd", "quoted_cost_native", "quoted_cost_usd",
        "accessorial_cost_usd", "actual_cost_usd", "data_source"]
    shipments = shipments[column_order]

    lanes = lanes[["lane_id", "lane_name", "origin_city", "origin_state", "origin_lat", "origin_lon",
                   "dest_city", "dest_state", "dest_country", "dest_lat", "dest_lon",
                   "distance_miles", "is_port_origin", "is_winter_exposed", "is_cross_border"]]
    ground_truth = pd.DataFrame(truth)
    return carriers, lanes, fuel, shipments, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic freight dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    carriers, lanes, fuel, shipments, ground_truth = generate(args.seed)
    os.makedirs(args.out, exist_ok=True)
    carriers.to_csv(os.path.join(args.out, "carriers.csv"), index=False)
    lanes.to_csv(os.path.join(args.out, "lanes.csv"), index=False)
    fuel.to_csv(os.path.join(args.out, "fuel_prices.csv"), index=False)
    shipments.to_csv(os.path.join(args.out, "shipments.csv"), index=False)
    ground_truth.to_csv(os.path.join(args.out, "ground_truth.csv"), index=False)

    hist = shipments[shipments["status"] == "Delivered"]
    active = shipments[shipments["status"] != "Delivered"]
    print(f"Wrote CSVs to ./{args.out}/")
    print(f"  carriers:     {len(carriers)}")
    print(f"  lanes:        {len(lanes)}")
    print(f"  fuel weeks:   {len(fuel)}")
    print(f"  shipments:    {len(shipments)}  ({len(hist)} delivered, {len(active)} active)")
    print(f"  delay rate (historical): {hist['is_delayed'].mean():.1%}")
    print(f"  avg actual cost (USD):   ${hist['actual_cost_usd'].mean():,.0f}")
    print("\nOn-time rate by carrier (historical):")
    otr = 1 - hist.groupby("carrier_id")["is_delayed"].mean()
    for cid, rate in otr.items():
        print(f"  {cid}  {rate:.1%}")


if __name__ == "__main__":
    main()
