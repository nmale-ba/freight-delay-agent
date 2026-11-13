"""Rebuild the Freight Risk Dashboard report pages (PBIR) with clean, correct visuals.
Usage: python3 build_pbir.py <report_definition_dir>   (replaces <dir>/pages)
"""
import json, sys, shutil, hashlib, pathlib

VC = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.14.0/schema.json"
PG = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.1.0/schema.json"
PM = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.1.0/schema.json"

# palette
NAVY, BLUE, TEAL, GREY = "#1E3A5F", "#2B6CB0", "#319795", "#A0AEC0"
RED, AMBER, GREEN = "#C53030", "#DD6B20", "#2F855A"
INK, MUTED, BG, CARD = "#1A202C", "#4A5568", "#F3F5F8", "#FFFFFF"
AGG = {"sum": 0, "avg": 1, "min": 3, "max": 4, "count": 5}
AGGNAME = {0: "Sum", 1: "Avg", 3: "Min", 4: "Max", 5: "CountNonNull"}

def uid(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]

def lit(v):
    if isinstance(v, bool): return {"expr": {"Literal": {"Value": "true" if v else "false"}}}
    if isinstance(v, (int, float)): return {"expr": {"Literal": {"Value": f"{v}D"}}}
    return {"expr": {"Literal": {"Value": "'" + str(v).replace("'", "''") + "'"}}}

def color(c): return {"solid": {"color": lit(c)}}

# ---------- field helpers ----------
def col(t, c): return {"Column": {"Expression": {"SourceRef": {"Entity": t}}, "Property": c}}
def meas(m): return {"Measure": {"Expression": {"SourceRef": {"Entity": "_Measures"}}, "Property": m}}
def agg(t, c, how): return {"Aggregation": {"Expression": col(t, c), "Function": AGG[how]}}

def qref(f):
    if "Measure" in f: return "_Measures." + f["Measure"]["Property"]
    if "Column" in f: return f["Column"]["Expression"]["SourceRef"]["Entity"] + "." + f["Column"]["Property"]
    a = f["Aggregation"]; c = a["Expression"]["Column"]
    return f'{AGGNAME[a["Function"]]}({c["Expression"]["SourceRef"]["Entity"]}.{c["Property"]})'

def proj(f, name=None, active=None):
    p = {"field": f, "queryRef": qref(f), "nativeQueryRef": name or qref(f).split(".")[-1].rstrip(")")}
    if name: p["displayName"] = name
    if active is not None: p["active"] = active
    return p

# ---------- filters ----------
def _where(t, c, cond):
    return {"Version": 2, "From": [{"Name": "s", "Entity": t, "Type": 0}], "Where": [{"Condition": cond}]}

def f_in(t, c, values, negate=False):
    vals = [[{"Literal": {"Value": (f"{v}L" if isinstance(v, int) else "'" + v + "'")}}] for v in values]
    cond = {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": c}}], "Values": vals}}
    if negate: cond = {"Not": {"Expression": cond}}
    return {"name": uid(t, c, str(values), str(negate)), "field": col(t, c), "type": "Categorical",
            "filter": _where(t, c, cond), "howCreated": "User"}

# ---------- formatting objects ----------
def container(title, subtitle=None):
    o = {"title": [{"properties": {"show": lit(True), "text": lit(title), "fontColor": color(INK),
                                    "fontSize": lit(13), "bold": lit(True)}}],
         "background": [{"properties": {"show": lit(True), "color": color(CARD), "transparency": lit(0)}}],
         "border": [{"properties": {"show": lit(True), "color": color("#E2E8F0"), "radius": lit(10)}}],
         "dropShadow": [{"properties": {"show": lit(False)}}],
         "padding": [{"properties": {"top": lit(12), "bottom": lit(12), "left": lit(14), "right": lit(14)}}]}
    if subtitle:
        o["subTitle"] = [{"properties": {"show": lit(True), "text": lit(subtitle), "fontColor": color(MUTED), "fontSize": lit(10)}}]
    return o

def fill_for(t, c, value, hexcolor):
    return {"properties": {"fill": color(hexcolor)},
            "selector": {"data": [{"scopeId": {"Comparison": {"ComparisonKind": 0, "Left": col(t, c),
                                                               "Right": {"Literal": {"Value": "'" + value + "'"}}}}}]}}

def fill_for_series(queryref, hexcolor):
    return {"properties": {"fill": color(hexcolor)}, "selector": {"metadata": queryref}}

def gradient(f, lo, hi, mid=None):
    rule = {"min": {"color": {"Literal": {"Value": f"'{lo}'"}}}, "max": {"color": {"Literal": {"Value": f"'{hi}'"}}},
            "nullColoringStrategy": {"strategy": {"Literal": {"Value": "'asZero'"}}}}
    key = "linearGradient2"
    if mid:
        rule["mid"] = {"color": {"Literal": {"Value": f"'{mid}'"}}}; key = "linearGradient3"
    return {"properties": {"backColor": {"solid": {"color": {"expr": {"FillRule": {"Input": f, "FillRule": {key: rule}}}}}}},
            "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}], "metadata": qref(f)}}

# ---------- visual builder ----------
class Page:
    def __init__(self, key, name):
        self.key, self.name, self.visuals, self.z = key, name, [], 0
        self.id = uid("page", key)

    def add(self, vtype, pos, query=None, objects=None, cobj=None, filters=None, sort=None, extra=None):
        self.z += 1000
        x, y, w, h = pos
        v = {"visualType": vtype}
        if query is not None:
            v["query"] = {"queryState": {role: {"projections": ps} for role, ps in query.items()}}
            if sort:
                v["query"]["sortDefinition"] = {"sort": [{"field": f, "direction": d} for f, d in sort], "isDefaultSort": True}
        if objects: v["objects"] = objects
        if cobj: v["visualContainerObjects"] = cobj
        if extra: v.update(extra)
        v["drillFilterOtherVisuals"] = True
        name = uid(self.key, vtype, str(pos))
        d = {"$schema": VC, "name": name,
             "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w, "tabOrder": self.z}, "visual": v}
        if filters: d["filterConfig"] = {"filters": filters}
        self.visuals.append(d)

    # convenience
    def header(self, title, subtitle):
        runs = [{"textRuns": [{"value": title, "textStyle": {"fontWeight": "bold", "fontSize": "22pt", "color": NAVY}}]},
                {"textRuns": [{"value": subtitle, "textStyle": {"fontSize": "11pt", "color": MUTED}}]}]
        self.add("textbox", (24, 12, 1300, 84),
                 objects={"general": [{"properties": {"paragraphs": runs}}]},
                 cobj={"background": [{"properties": {"show": lit(False)}}]})

    def cards(self, measures, pos, title=None):
        self.add("cardVisual", pos, query={"Data": [proj(meas(m), m) for m in measures]},
                 cobj=container(title) if title else {**container(""), "title": [{"properties": {"show": lit(False)}}]})

    def slicer(self, t, c, label, pos):
        self.add("slicer", pos, query={"Values": [proj(col(t, c), label, True)]},
                 objects={"data": [{"properties": {"mode": lit("Dropdown")}}],
                          "header": [{"properties": {"show": lit(True), "text": lit(label), "fontColor": color(MUTED)}}]},
                 cobj={"background": [{"properties": {"show": lit(True), "color": color(CARD)}}],
                       "border": [{"properties": {"show": lit(True), "color": color("#E2E8F0"), "radius": lit(10)}}]})

def bar_objects(fill=BLUE, labels=True, extra=None):
    o = {"dataPoint": [{"properties": {"fill": color(fill)}}],
         "labels": [{"properties": {"show": lit(labels), "color": color(INK), "fontSize": lit(9)}}],
         "valueAxis": [{"properties": {"show": lit(False), "gridlineShow": lit(False)}}],
         "categoryAxis": [{"properties": {"showAxisTitle": lit(False), "fontSize": lit(10), "labelColor": color(MUTED)}}]}
    if extra: o.update(extra)
    return o

AS, SH, CL, RC, DP, SS, ME, AE = ("ActiveShipments", "ShipmentHistory", "CarrierLanePerf", "RouteConditions",
                                  "DieselPrices", "SwitchSensitivity", "ModelEvaluation", "AgentEvaluation")
M = 24; G = 16

def build():
    pages = []

    # ---------------- 1. Shipment Risk Overview ----------------
    p = Page("p1", "1. Shipment Risk Overview"); pages.append(p)
    p.header("Shipment Risk Overview", "100 active shipments scored by the XGBoost delay model. Exposure = extra carrier charges + delay probability × late-delivery cost.")
    for i, (c, lab) in enumerate([("delivery_window", "Delivery window"), ("carrier_name", "Carrier"), ("lane_name", "Lane")]):
        p.slicer(AS, c, lab, (1340 + i * 188, 20, 176, 72))
    p.cards(["Active Shipments", "Already Late", "Predicted At Risk", "% Shipments Needing Attention",
             "Total Exposure", "At-Risk Exposure"], (M, 108, 1872, 140))
    rg = col(AS, "risk_group")
    p.add("donutChart", (M, 264, 520, 380),
          query={"Category": [proj(rg, "Risk group", True)], "Y": [proj(meas("Active Shipments"), "Shipments")]},
          objects={"dataPoint": [fill_for(AS, "risk_group", "Already late", RED),
                                 fill_for(AS, "risk_group", "Predicted at risk", AMBER),
                                 fill_for(AS, "risk_group", "Low risk", GREEN)],
                   "legend": [{"properties": {"show": lit(True), "position": lit("Bottom")}}],
                   "labels": [{"properties": {"show": lit(True), "labelStyle": lit("Data value, percent of total")}}]},
          cobj=container("Where the active shipments stand", "Count of shipments by risk group"))
    p.add("columnChart", (M + 520 + G, 264, 640, 380),
          query={"Category": [proj(col(AS, "delivery_window"), "Delivery window", True)],
                 "Series": [proj(rg, "Risk group")], "Y": [proj(meas("Active Shipments"), "Shipments")]},
          objects=bar_objects(extra={"dataPoint": [fill_for(AS, "risk_group", "Already late", RED),
                                                   fill_for(AS, "risk_group", "Predicted at risk", AMBER),
                                                   fill_for(AS, "risk_group", "Low risk", GREEN)],
                                     "legend": [{"properties": {"show": lit(True), "position": lit("Top")}}]}),
          cobj=container("Risk by delivery window", "Overdue loads first, then the next 7 and 8-14 days"))
    p.add("clusteredBarChart", (M + 1160 + 2 * G, 264, 680, 380),
          query={"Category": [proj(col(AS, "lane_name"), "Lane", True)], "Y": [proj(meas("Total Exposure"), "Total exposure")]},
          sort=[(meas("Total Exposure"), "Descending")], objects=bar_objects(NAVY),
          cobj=container("Exposure by lane ($)", "Expected cost of delays on the active book"))
    p.add("tableEx", (M, 660, 1872, 404),
          query={"Values": [proj(col(AS, "shipment_id"), "Shipment"), proj(col(AS, "lane_name"), "Lane"),
                            proj(col(AS, "carrier_name"), "Carrier"), proj(col(AS, "delivery_window"), "Window"),
                            proj(col(AS, "risk_group"), "Risk group"), proj(col(AS, "weather_label"), "Weather"),
                            proj(meas("Avg Delay Probability"), "Delay probability"),
                            proj(meas("Total Exposure"), "Exposure"), proj(col(AS, "recommended_action"), "Recommended action")]},
          sort=[(meas("Total Exposure"), "Descending")],
          objects={"values": [gradient(meas("Avg Delay Probability"), "#FFFFFF", "#F6AD55"), ],
                   "columnHeaders": [{"properties": {"bold": lit(True), "fontColor": color(INK), "backColor": color("#EDF2F7")}}],
                   "grid": [{"properties": {"rowPadding": lit(4), "gridHorizontal": lit(True)}}]},
          cobj=container("Shipments needing attention", "Already late or predicted at risk, highest exposure first"),
          filters=[f_in(AS, "risk_group", ["Low risk"], negate=True)])

    # ---------------- 2. Carrier Performance ----------------
    p = Page("p2", "2. Carrier Performance"); pages.append(p)
    p.header("Carrier Performance", "1,200 delivered shipments: who delivers on time, on which lanes, and at what cost.")
    p.slicer(SH, "carrier_name", "Carrier", (1528, 20, 176, 72)); p.slicer(SH, "lane_name", "Lane", (1716, 20, 176, 72))
    p.cards(["Historical Shipments", "Historical On-Time %", "Historical Delay Rate", "Avg Delay Days (Late Loads)",
             "Avg Cost Overrun", "Weak Lanes"], (M, 108, 1872, 140))
    p.add("clusteredBarChart", (M, 264, 600, 380),
          query={"Category": [proj(col(SH, "carrier_name"), "Carrier", True)], "Y": [proj(meas("Historical On-Time %"), "On-time %")]},
          sort=[(meas("Historical On-Time %"), "Descending")], objects=bar_objects(TEAL),
          cobj=container("On-time rate by carrier", "Share of loads delivered on or before the planned date"))
    p.add("lineChart", (M + 600 + G, 264, 1256, 380),
          query={"Category": [proj(col(SH, "ship_month"), "Month", True)],
                 "Y": [proj(meas("Historical Delay Rate"), "Delay rate")]},
          objects={"dataPoint": [{"properties": {"fill": color(RED)}}],
                   "lineStyles": [{"properties": {"strokeWidth": lit(3), "showMarker": lit(True)}}],
                   "categoryAxis": [{"properties": {"showAxisTitle": lit(False)}}],
                   "valueAxis": [{"properties": {"showAxisTitle": lit(False), "gridlineColor": color("#EDF2F7")}}]},
          cobj=container("Delay rate by ship month", "Winter months run higher (weather effect)"))
    p.add("pivotTable", (M, 660, 1100, 404),
          query={"Rows": [proj(col(CL, "carrier_name"), "Carrier", True)], "Columns": [proj(col(CL, "lane_id"), "Lane", True)],
                 "Values": [proj(agg(CL, "on_time_pct", "avg"), "On-time %")]},
          objects={"values": [gradient(agg(CL, "on_time_pct", "avg"), "#FED7D7", "#C6F6D5", "#FEFCBF")],
                   "subTotals": [{"properties": {"rowSubtotals": lit(False), "columnSubtotals": lit(False)}}],
                   "columnHeaders": [{"properties": {"bold": lit(True), "backColor": color("#EDF2F7")}}]},
          cobj=container("On-time % heatmap: carrier × lane", "Red cells are carrier-lane pairs to renegotiate or avoid"))
    p.add("tableEx", (M + 1100 + G, 660, 756, 404),
          query={"Values": [proj(col(CL, "carrier_name"), "Carrier"), proj(col(CL, "lane_name"), "Lane"),
                            proj(agg(CL, "shipments", "sum"), "Loads"), proj(agg(CL, "on_time_pct", "avg"), "On-time %"),
                            proj(agg(CL, "vs_carrier_overall_pts", "avg"), "vs carrier avg (pts)")]},
          objects={"columnHeaders": [{"properties": {"bold": lit(True), "backColor": color("#EDF2F7")}}]},
          cobj=container("Weak lanes", "Carrier-lane pairs well below that carrier's own average"),
          filters=[f_in(CL, "flag", ["Weak lane"])])

    # ---------------- 3. Route Conditions & Fuel ----------------
    p = Page("p3", "3. Route Conditions & Fuel"); pages.append(p)
    p.header("Route Conditions & Fuel Market", "Open-Meteo 16-day route forecast (5 points per lane) and EIA weekly diesel prices.")
    p.cards(["Worst Forecast Severity", "Lanes With Moderate+ Weather", "Latest Diesel Price",
             "Latest Fuel Surcharge per Mile", "Diesel Change 12 Weeks %"], (M, 108, 1872, 140))
    fc = [f_in(RC, "period", ["Forecast"])]
    p.add("pivotTable", (M, 264, 1872, 420),
          query={"Rows": [proj(col(RC, "lane_name"), "Lane", True)], "Columns": [proj(col(RC, "forecast_date"), "Date", True)],
                 "Values": [proj(agg(RC, "weather_severity_forecast", "max"), "Severity")]},
          objects={"values": [gradient(agg(RC, "weather_severity_forecast", "max"), "#F0FFF4", "#E53E3E", "#FBD38D")],
                   "subTotals": [{"properties": {"rowSubtotals": lit(False), "columnSubtotals": lit(False)}}],
                   "columnHeaders": [{"properties": {"bold": lit(True), "backColor": color("#EDF2F7")}}]},
          cobj=container("Forecast weather severity by lane and day", "0 = none, 1 = minor, 2 = moderate, 3 = severe"),
          filters=fc)
    p.add("clusteredBarChart", (M, 700, 928, 364),
          query={"Category": [proj(col(RC, "lane_name"), "Lane", True)],
                 "Y": [proj(agg(RC, "precipitation_mm", "sum"), "Rain (mm)"), proj(agg(RC, "snowfall_cm", "sum"), "Snow (cm)")]},
          sort=[(agg(RC, "precipitation_mm", "sum"), "Descending")],
          objects=bar_objects(extra={"dataPoint": [fill_for_series("Sum(RouteConditions.precipitation_mm)", BLUE),
                                                   fill_for_series("Sum(RouteConditions.snowfall_cm)", GREY)],
                                     "legend": [{"properties": {"show": lit(True), "position": lit("Top")}}]}),
          cobj=container("Forecast precipitation and snowfall by lane", "Totals over the 16-day forecast"), filters=fc)
    p.add("lineChart", (M + 928 + G, 700, 928, 364),
          query={"Category": [proj(col(DP, "week_start"), "Week", True)],
                 "Y": [proj(agg(DP, "diesel_price_usd_gal", "avg"), "Diesel $/gal")]},
          objects={"dataPoint": [{"properties": {"fill": color(NAVY)}}],
                   "lineStyles": [{"properties": {"strokeWidth": lit(3)}}],
                   "categoryAxis": [{"properties": {"showAxisTitle": lit(False)}}],
                   "valueAxis": [{"properties": {"showAxisTitle": lit(False), "gridlineColor": color("#EDF2F7")}}]},
          cobj=container("US on-highway diesel price ($/gal)", "Weekly EIA series; drives the fuel surcharge"))

    # ---------------- 4. Recommended Actions ----------------
    p = Page("p4", "4. Recommended Actions"); pages.append(p)
    p.header("Recommended Actions", "Carrier switches use an expected-value rule: switch only if lower risk × late cost outweighs the freight cost change.")
    p.cards(["Carrier Switches", "Avg Risk Before Switch", "Avg Risk After Switch", "Switch Freight Cost Change",
             "Expected Lateness Cost Avoided", "Net Benefit of Switches"], (M, 108, 1872, 140))
    p.add("tableEx", (M, 264, 1872, 400),
          query={"Values": [proj(col(AS, "shipment_id"), "Shipment"), proj(col(AS, "lane_name"), "Lane"),
                            proj(col(AS, "carrier_name"), "Current carrier"), proj(col(AS, "alt_carrier_name"), "Switch to"),
                            proj(meas("Avg Risk Before Switch"), "Risk now"), proj(meas("Avg Risk After Switch"), "Risk after"),
                            proj(meas("Avg Risk Reduction (pts)"), "Reduction (pts)"),
                            proj(meas("Switch Freight Cost Change"), "Freight cost change"),
                            proj(meas("Net Benefit of Switches"), "Net benefit")]},
          sort=[(meas("Net Benefit of Switches"), "Descending")],
          objects={"values": [gradient(meas("Net Benefit of Switches"), "#FFFFFF", "#9AE6B4")],
                   "columnHeaders": [{"properties": {"bold": lit(True), "backColor": color("#EDF2F7")}}]},
          cobj=container("Recommended carrier switches", "Highest net benefit first"),
          filters=[f_in(AS, "has_switch", [1])])
    p.add("clusteredBarChart", (M, 680, 760, 384),
          query={"Category": [proj(col(AS, "action_type"), "Action", True)], "Y": [proj(meas("Active Shipments"), "Shipments")]},
          sort=[(meas("Active Shipments"), "Descending")], objects=bar_objects(BLUE),
          cobj=container("Shipments by recommended action", "What the planner should do today"),
          filters=[f_in(AS, "action_type", ["No action"], negate=True)])
    ss_sw, ss_cost = agg(SS, "carrier_switches", "sum"), agg(SS, "freight_cost_change_usd", "sum")
    p.add("lineClusteredColumnComboChart", (M + 760 + G, 680, 1096, 384),
          query={"Category": [proj(col(SS, "late_penalty_usd"), "Late-delivery cost ($)", True)],
                 "Y": [proj(ss_sw, "Carrier switches")], "Y2": [proj(ss_cost, "Freight cost change ($)")]},
          sort=[(col(SS, "late_penalty_usd"), "Ascending")],
          objects={"dataPoint": [fill_for_series(qref(ss_sw), TEAL), fill_for_series(qref(ss_cost), AMBER)],
                   "categoryAxis": [{"properties": {"axisType": lit("Categorical"), "showAxisTitle": lit(True)}}],
                   "labels": [{"properties": {"show": lit(True), "fontSize": lit(9)}}],
                   "legend": [{"properties": {"show": lit(True), "position": lit("Top")}}]},
          cobj=container("Sensitivity: how the late-delivery cost changes the advice",
                         "Higher assumed cost of lateness → more switches justified"))

    # ---------------- 5. AI Agent & Model Evaluation ----------------
    p = Page("p5", "5. AI Agent & Model Evaluation"); pages.append(p)
    p.header("AI Agent & Model Evaluation", "12 test scenarios × 3 repeats on gpt-4.1-mini with tools, vs the same model without tools; XGBoost holdout metrics.")
    p.cards(["Agent Correctness", "Baseline Correctness", "Correctness Lift (pts)", "Agent Pass Rate",
             "Agent Grounding Rate", "Avg Tokens per Question", "Avg Response Time (s)"], (M, 108, 1316, 140), )
    p.cards(["Prediction ROC-AUC", "Baseline ROC-AUC"], (M + 1316 + G, 108, 540, 140))
    latest = [f_in(AE, "is_latest_batch", [1])]
    corr = agg(AE, "correctness", "avg")
    p.add("clusteredBarChart", (M, 264, 928, 400),
          query={"Category": [proj(col(AE, "scenario_group"), "Scenario group", True)],
                 "Series": [proj(col(AE, "mode_label"), "Mode")], "Y": [proj(corr, "Correctness")]},
          objects=bar_objects(extra={"dataPoint": [fill_for(AE, "mode_label", "Agent (with tools)", BLUE),
                                                   fill_for(AE, "mode_label", "Same model, no tools", GREY)],
                                     "legend": [{"properties": {"show": lit(True), "position": lit("Top")}}]}),
          cobj=container("Answer correctness: agent vs no-tools baseline", "Share of answer-key facts present in the reply"),
          filters=latest)
    mv, bv = agg(ME, "model_value", "max"), agg(ME, "baseline_value", "max")
    p.add("clusteredColumnChart", (M + 928 + G, 264, 928, 400),
          query={"Category": [proj(col(ME, "metric"), "Metric", True)], "Y": [proj(mv, "XGBoost"), proj(bv, "Baseline")]},
          objects=bar_objects(extra={"dataPoint": [fill_for_series(qref(mv), NAVY), fill_for_series(qref(bv), GREY)],
                                     "legend": [{"properties": {"show": lit(True), "position": lit("Top")}}]}),
          cobj=container("Delay model vs baseline (holdout test set)", "Higher is better for every metric shown"),
          filters=[f_in(ME, "model", ["delay_classifier"]),
                   f_in(ME, "metric", ["roc_auc", "pr_auc", "precision_at_medium_plus", "recall_at_medium_plus", "f1_at_medium_plus"])])
    p.add("tableEx", (M, 680, 1872, 384),
          query={"Values": [proj(col(AE, "scenario"), "Scenario"), proj(col(AE, "question"), "Question"),
                            proj(col(AE, "tools_called"), "Tools called"), proj(agg(AE, "correctness", "avg"), "Correctness"),
                            proj(agg(AE, "grounding_rate", "avg"), "Grounding"), proj(agg(AE, "passed", "sum"), "Passed (of 3)"),
                            proj(agg(AE, "tokens", "avg"), "Avg tokens"), proj(agg(AE, "latency_s", "avg"), "Avg seconds")]},
          objects={"values": [gradient(agg(AE, "correctness", "avg"), "#FED7D7", "#C6F6D5")],
                   "columnHeaders": [{"properties": {"bold": lit(True), "backColor": color("#EDF2F7")}}]},
          cobj=container("Agent results by scenario (latest batch)", "One row per test question, averaged over repeats"),
          filters=latest + [f_in(AE, "mode", ["agent"])])
    return pages

def write(defdir):
    defdir = pathlib.Path(defdir); pdir = defdir / "pages"
    if pdir.exists(): shutil.rmtree(pdir)
    pages = build()
    for p in pages:
        d = pdir / p.id; (d / "visuals").mkdir(parents=True)
        page = {"$schema": PG, "name": p.id, "displayName": p.name, "displayOption": "FitToPage", "height": 1080, "width": 1920,
                "objects": {"background": [{"properties": {"color": color(BG), "transparency": lit(0)}}],
                            "outspace": [{"properties": {"color": color(BG)}}]}}
        (d / "page.json").write_text(json.dumps(page, indent=2))
        for v in p.visuals:
            vd = d / "visuals" / v["name"]; vd.mkdir()
            (vd / "visual.json").write_text(json.dumps(v, indent=2))
    (pdir / "pages.json").write_text(json.dumps({"$schema": PM, "pageOrder": [p.id for p in pages],
                                                 "activePageName": pages[0].id}, indent=2))
    print(f"wrote {len(pages)} pages, {sum(len(p.visuals) for p in pages)} visuals")

if __name__ == "__main__":
    write(sys.argv[1])
