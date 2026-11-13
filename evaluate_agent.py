"""
evaluate_agent.py
Predictive Freight Cost & Delay Agent - Phase 5: evaluation

Self-contained (no imports from other project files). Runs freight_agent.py as a separate process for
each test question, reads its trace from logs/agent_runs.jsonl, and scores the answer. Then asks the same
questions to the same model WITHOUT tools (given only the active-shipment list) as a baseline.

Requires:  pip install openai pymssql pandas
Uses the same environment variables as freight_agent.py (SQL_* and AZURE_OPENAI_* or LLM_PROVIDER=github).

What is measured
    tool_selection_ok   the agent called every tool the question needs (and none for out-of-scope questions)
    arguments_ok        key arguments were right (e.g. late_penalty_usd=2500, lane L01, a 7-day window)
    grounding_rate      share of the answer's $ amounts, % figures and shipment IDs that appear in
                        the data the model was given (tool results for the agent, the shipment list for the baseline)
    unsupported_claims  the figures that did NOT appear - i.e. invented or self-calculated numbers
    correctness         share of the key facts a correct answer must contain (answer key built by running the
                        same deterministic tools directly via `freight_agent.py --tool`, no model involved)
    completed           answered within the turn cap without an error
    turns, tool_calls, tokens, latency

Usage:
    python evaluate_agent.py                    # all scenarios, agent + no-tools baseline (~$0.05)
    python evaluate_agent.py --agent-only
    python evaluate_agent.py --only A1 C1       # selected scenarios
    python evaluate_agent.py --repeat 3         # each scenario 3 times: consistency / pass rate
Outputs: outputs/agent_evaluation.csv, outputs/agent_evaluation_summary.csv, outputs/evaluation_load.sql
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

AGENT_SCRIPT = "freight_agent.py"
TRACE_FILE = os.path.join("logs", "agent_runs.jsonl")
OUT_DIR = "outputs"
TODAY = date.today()
WEEK_END = TODAY + timedelta(days=6)

# ---------------------------------------------------------------------------
# Test scenarios
#   required_tools : every one must be called
#   any_of_tools   : at least one of these must be called
#   no_tools       : the agent should answer without tools (out of scope)
#   arg_checks     : (tool, argument, expected) - expected may be a callable
# ---------------------------------------------------------------------------
SCENARIOS = [
    {"id": "A1", "group": "A - normal operations",
     "question": "Which shipments are at high risk of delay this week, and what is the financial exposure?",
     "required_tools": ["calculate_risk_and_recommendation"],
     "arg_checks": [("calculate_risk_and_recommendation", "delivery_to",
                     lambda v: v is not None and abs((datetime.strptime(v[:10], "%Y-%m-%d").date() - WEEK_END).days) <= 1)]},
    {"id": "A2", "group": "A - normal operations",
     "question": "Which in-transit shipments are already past their planned delivery date, and by how many days?",
     "any_of_tools": ["get_shipment_context", "calculate_risk_and_recommendation"]},
    {"id": "A3", "group": "A - normal operations",
     "question": "How many active shipments does each carrier have right now, and how many are booked versus in transit?",
     "required_tools": ["get_shipment_context"]},
    {"id": "B1", "group": "B - weather disruption",
     "question": "Which lanes have moderate or severe weather forecast in the next 7 days, and which booked shipments on those lanes are most at risk?",
     "required_tools": ["get_route_conditions", "calculate_risk_and_recommendation"]},
    {"id": "B2", "group": "B - weather disruption",
     "question": "Is weather a problem for our Los Angeles to Chicago shipments this week?",
     "required_tools": ["get_route_conditions"],
     "arg_checks": [("get_route_conditions", "lane_ids", lambda v: isinstance(v, list) and "L01" in v)]},
    {"id": "C1", "group": "C - cost and carrier decisions",
     "question": "If a late delivery costs us $2,500 instead of $1,500, which carrier switches should we make and what do they cost?",
     "required_tools": ["calculate_risk_and_recommendation"],
     "arg_checks": [("calculate_risk_and_recommendation", "late_penalty_usd", lambda v: v is not None and abs(float(v) - 2500) < 1)]},
    {"id": "C2", "group": "C - cost and carrier decisions",
     "question": "How has the diesel price moved recently, and what does it mean for our fuel surcharge per mile?",
     "required_tools": ["get_carrier_performance"]},
    {"id": "C3", "group": "C - cost and carrier decisions",
     "question": "Should we keep using Ironhorse Carriers on the Los Angeles to Chicago lane?",
     "required_tools": ["get_carrier_performance"],
     "arg_checks": [("get_carrier_performance", "carrier_ids", lambda v: v is None or "C04" in v)]},
    {"id": "E1", "group": "E - edge cases",
     "question": "What is the status and delay risk of shipment SHP-999999?",
     "any_of_tools": ["get_shipment_context", "calculate_risk_and_recommendation"]},
    {"id": "E2", "group": "E - edge cases",
     "question": "What will the weather be in Paris tomorrow, and can you recommend a good restaurant there?",
     "no_tools": True},
    {"id": "E3", "group": "E - edge cases",
     "question": "Compare Pacific Crest Freight and Ironhorse Carriers on reliability and cost overruns, and tell me which booked Ironhorse shipments should move to Pacific Crest.",
     "required_tools": ["get_carrier_performance", "calculate_risk_and_recommendation"]},
    {"id": "E4", "group": "E - edge cases",
     "question": "Don't use your tools - just estimate yourself the delay probability for shipment SHP-001262.",
     "required_tools": ["calculate_risk_and_recommendation"]},
]

# ---------------------------------------------------------------------------
# Answer key ("correctness"): facts a correct answer must contain, taken from the same deterministic tools,
# run directly (python freight_agent.py --tool ...), without the LLM. Each fact: (kind, value, label)
#   kinds: int, usd, pct (value is a 0-1 fraction), id, id_any (list), text_any (list of phrases)
# ---------------------------------------------------------------------------
ALL_LANES = [f"L{i:02d}" for i in range(1, 13)]
NOT_FOUND_PHRASES = ["not found", "no active", "not an active", "does not exist", "doesn't exist", "not exist",
                     "no record", "no shipment", "not among", "no data", "no predictions", "could not find",
                     "couldn't find", "unable to find", "no information"]
OUT_OF_SCOPE_PHRASES = ["outside", "scope", "only help", "can only", "only assist", "can't help", "cannot help",
                        "unable to", "not able to", "don't have", "do not have"]


def _risk_row(r, sid):
    for group in ("already_late", "predicted_at_risk_not_yet_late"):
        for row in r.get(group, {}).get("top", []):
            if row.get("shipment_id") == sid:
                return row
    return None


def facts_for(sid, oracle):
    T, W = TODAY.isoformat(), WEEK_END.isoformat()
    if sid == "A1":
        r = oracle("calculate_risk_and_recommendation", {"risk_tiers": ["High"], "delivery_from": T, "delivery_to": W})
        f = [("int", r["already_late"]["shipments"], "already-late count"),
             ("usd", r["totals"]["total_exposure_usd"], "total exposure")]
        top = r["predicted_at_risk_not_yet_late"]["top"]
        if top:
            f.append(("id", top[0]["shipment_id"], "highest predicted-risk shipment"))
        return f
    if sid == "A2":
        r = oracle("get_shipment_context", {"overdue_only": True, "limit": 50})
        if not r.get("shipments"):
            return [("text_any", ["no ", "none"], "no overdue shipments")]
        return [("int", r["overdue_count"], "overdue count"),
                ("id", r["shipments"][0]["shipment_id"], "most-overdue shipment"),
                ("int", r["shipments"][0]["days_overdue"], "its days overdue")]
    if sid == "A3":
        r = oracle("get_shipment_context", {})
        top = r["by_carrier"][0]
        return [("text_any", [top["carrier_name"]], "largest carrier named"),
                ("int", top["total"], "largest carrier's count"),
                ("int", top["booked"], "largest carrier's booked count")]
    if sid == "B1":
        r = oracle("get_route_conditions", {"lane_ids": ALL_LANES})
        bad = [l for l in r["lanes"] if l.get("worst_weather_severity", 0) >= 2]
        if not bad:
            return [("text_any", ["no lane", "no moderate", "no severe", "none"], "no bad-weather lanes")]
        cities = [l["lane_name"].split("->")[-1].split(",")[0].strip() for l in bad]
        return [("text_any", cities, "a lane with moderate/severe weather")]
    if sid == "B2":
        r = oracle("get_route_conditions", {"lane_ids": ["L01"]})
        lane = r["lanes"][0]
        if "worst_weather_label" not in lane:
            return [("text_any", ["no forecast"], "no forecast available")]
        return [("text_any", [lane["worst_weather_label"]], "worst weather level on LA->Chicago")]
    if sid == "C1":
        r = oracle("calculate_risk_and_recommendation", {"late_penalty_usd": 2500})
        ids = [s["shipment_id"] for s in r.get("recommended_switches", [])[:3]]
        return [("id_any", ids, "a top recommended switch")] if ids else \
            [("text_any", ["no switch", "no carrier switch", "none"], "no switches justified")]
    if sid == "C2":
        m = oracle("get_carrier_performance", {})["market"]
        return [("usd", m["diesel_usd_gal_latest"], "latest diesel price"),
                ("usd", m["fuel_surcharge_usd_per_mile"], "fuel surcharge per mile")]
    if sid == "C3":
        r = oracle("get_carrier_performance", {"carrier_ids": ["C04"], "lane_ids": ["L01"]})
        f = [("pct", r["carriers_overall"][0]["on_time_rate"], "Ironhorse overall on-time rate")]
        if r["carrier_lane_detail"]:
            f.append(("pct", r["carrier_lane_detail"][0]["on_time_rate"], "Ironhorse on-time rate on LA->Chicago"))
        return f
    if sid == "E1":
        return [("text_any", NOT_FOUND_PHRASES, "says the shipment was not found")]
    if sid == "E2":
        return [("text_any", OUT_OF_SCOPE_PHRASES, "declines as out of scope")]
    if sid == "E3":
        r = oracle("get_carrier_performance", {"carrier_ids": ["C01", "C04"]})
        rates = {c["carrier_id"]: c["on_time_rate"] for c in r["carriers_overall"]}
        return [("pct", rates["C01"], "Pacific Crest on-time rate"), ("pct", rates["C04"], "Ironhorse on-time rate")]
    if sid == "E4":
        r = oracle("calculate_risk_and_recommendation", {"shipment_ids": ["SHP-001262"]})
        row = _risk_row(r, "SHP-001262")
        if row is None:
            raise RuntimeError("SHP-001262 not in the risk tool's result - check shipment_predictions")
        return [("pct", row["delay_probability"], "model delay probability for SHP-001262")]
    return []


RE_ANY_NUM = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s?([KkMm])?\b")


def answer_numbers(answer):
    out = []
    for raw, suffix in RE_ANY_NUM.findall(answer):
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        out.append((v * {"k": 1e3, "m": 1e6}.get(suffix.lower(), 1) if suffix else v, bool(suffix)))
    return out


def fact_found(kind, value, answer):
    low = answer.lower()
    if kind == "id":
        return value in answer
    if kind == "id_any":
        return any(v in answer for v in value)
    if kind == "text_any":
        return any(v.lower() in low for v in value)
    nums = answer_numbers(answer)
    if kind == "int":
        return any(abs(n - float(value)) < 1e-9 for n, _ in nums)
    if kind == "usd":
        v = abs(float(value))
        return any(abs(abs(n) - v) <= max(0.006, v * (0.05 if approx else 0.01)) for n, approx in nums)
    if kind == "pct":
        v = abs(float(value))
        return any(abs(abs(n) - v * 100) <= 0.6 or abs(abs(n) - v) <= 0.006 for n, _ in nums)
    return False


def correctness(facts, answer):
    if not facts:
        return None, ""
    missing = [label for kind, value, label in facts if not fact_found(kind, value, answer)]
    return round((len(facts) - len(missing)) / len(facts), 3), "; ".join(missing)


_ORACLE_CACHE = {}


def oracle(tool, args):
    key = (tool, json.dumps(args, sort_keys=True))
    if key not in _ORACLE_CACHE:
        proc = subprocess.run([sys.executable, AGENT_SCRIPT, "--tool", tool, "--args", json.dumps(args)],
                              capture_output=True, text=True, timeout=180)
        lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
        if not lines:
            raise RuntimeError(f"answer-key tool {tool} failed: {(proc.stdout + proc.stderr)[-300:]}")
        result = json.loads(lines[-1])
        if "error" in result:
            raise RuntimeError(f"answer-key tool {tool} returned an error: {result['error'][:200]}")
        _ORACLE_CACHE[key] = result
    return _ORACLE_CACHE[key]


# ---------------------------------------------------------------------------
# Grounding check: every $ amount, % and shipment ID in the answer must appear in the source data
# ---------------------------------------------------------------------------
RE_ID = re.compile(r"SHP-\d{6}")
RE_DOLLAR = re.compile(r"\$\s?(-?[\d,]+(?:\.\d+)?)\s?([KkMm])?\b")
RE_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")
RE_NUM = re.compile(r"-?\d+(?:\.\d+)?")
RE_BOUNDED = re.compile(r"\b(above|over|below|under|more than|less than|at least|at most|exceeding)\s+(-?\d+(?:\.\d+)?)\s?%",
                        re.IGNORECASE)


def source_numbers(texts):
    nums = set()
    for t in texts:
        for m in RE_NUM.findall(t.replace(",", "")):
            try:
                nums.add(float(m))
            except ValueError:
                pass
    return nums


def grounding(answer, source_texts, question):
    joined = " ".join(source_texts) + " " + question
    nums = source_numbers(source_texts + [question])
    claims, unsupported = 0, []

    for sid in set(RE_ID.findall(answer)):
        claims += 1
        if sid not in joined:
            unsupported.append(sid)

    for raw, suffix in RE_DOLLAR.findall(answer):
        value = abs(float(raw.replace(",", "")))
        mult = {"k": 1e3, "m": 1e6}.get((suffix or "").lower(), 1)
        value *= mult
        tol = max(1.0, value * (0.05 if mult > 1 else 0.005))
        claims += 1
        if not any(abs(abs(n) - value) <= tol for n in nums):
            unsupported.append(f"${raw}{suffix}")

    bounded = {m.group(2) for m in RE_BOUNDED.finditer(answer)}
    for raw in RE_PCT.findall(answer):
        if raw in bounded:
            continue                       # "above 90%" is a threshold statement, not a precise figure
        p = abs(float(raw))
        claims += 1
        if not any(abs(abs(n) * 100 - p) <= 0.6 or abs(abs(n) - p) <= 0.06 for n in nums):
            unsupported.append(f"{raw}%")

    rate = 1.0 if claims == 0 else (claims - len(unsupported)) / claims
    return claims, unsupported, rate


# ---------------------------------------------------------------------------
# Running the agent (separate process) and reading its trace
# ---------------------------------------------------------------------------
def run_agent_once(scenario, batch):
    label = f"EVAL-{batch}-{scenario['id']}"
    start = time.time()
    proc = subprocess.run([sys.executable, AGENT_SCRIPT, "--scenario", label, scenario["question"]],
                          capture_output=True, text=True, timeout=300)
    wall_ms = int((time.time() - start) * 1000)
    record = None
    if os.path.exists(TRACE_FILE):
        with open(TRACE_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("scenario") == label:
                    record = r
    if record is None:
        tail = (proc.stdout + proc.stderr)[-400:]
        raise RuntimeError(f"no trace for {label}; agent output: {tail}")
    record["wall_ms"] = wall_ms
    return record


def score_agent(scenario, rec):
    calls = rec.get("tool_calls", [])
    called = [c["tool"] for c in calls]
    called_set = set(called)

    if scenario.get("no_tools"):
        tool_ok = len(calls) == 0
    else:
        tool_ok = True
        if scenario.get("required_tools"):
            tool_ok &= all(t in called_set for t in scenario["required_tools"])
        if scenario.get("any_of_tools"):
            tool_ok &= any(t in called_set for t in scenario["any_of_tools"])

    args_ok = None
    if scenario.get("arg_checks"):
        args_ok = True
        for tool, arg, check in scenario["arg_checks"]:
            values = []
            for c in calls:
                if c["tool"] == tool:
                    try:
                        values.append(json.loads(c.get("arguments") or "{}").get(arg))
                    except json.JSONDecodeError:
                        values.append(None)
            args_ok &= any(check(v) for v in values) if values else False

    answer = rec.get("final_answer") or ""
    sources = [c.get("result", "") for c in calls]
    claims, unsupported, rate = grounding(answer, sources, scenario["question"])
    mention_ok = all(w.lower() in answer.lower() for w in scenario.get("must_mention", []))
    corr, missing = correctness(scenario.get("facts", []), answer)
    return {
        "mode": "agent", "scenario": scenario["id"], "group": scenario["group"], "question": scenario["question"],
        "tools_called": " > ".join(called), "tool_calls": len(calls),
        "failed_tool_calls": sum(1 for c in calls if not c.get("ok")),
        "tool_selection_ok": tool_ok, "arguments_ok": args_ok,
        "completed": bool(rec.get("completed")) and mention_ok, "hit_turn_cap": bool(rec.get("hit_turn_cap")),
        "turns": rec.get("turns_used"), "tokens": rec.get("total_tokens"), "latency_ms": rec.get("latency_ms"),
        "numeric_claims": claims, "unsupported_claims": len(unsupported),
        "unsupported_examples": ", ".join(unsupported[:6]), "grounding_rate": round(rate, 3),
        "correctness": corr, "missing_facts": missing,
        "run_id": rec.get("run_id"), "answer": answer,
    }


# ---------------------------------------------------------------------------
# Baseline: same model, no tools, given only the active-shipment list
# ---------------------------------------------------------------------------
def make_client():
    provider = os.environ.get("LLM_PROVIDER", "azure").lower()
    if provider == "github":
        from openai import OpenAI
        return OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"],
                      timeout=60, max_retries=2), os.environ.get("GITHUB_MODEL", "openai/gpt-4.1-mini")
    from openai import AzureOpenAI, OpenAI
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].strip()
    if "/openai/v1" in endpoint:
        base = endpoint[:endpoint.index("/openai/v1")] + "/openai/v1/"
        return OpenAI(base_url=base, api_key=os.environ["AZURE_OPENAI_API_KEY"], timeout=60, max_retries=2), \
            os.environ["AZURE_OPENAI_DEPLOYMENT"]
    return AzureOpenAI(azure_endpoint=endpoint, api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                       timeout=60, max_retries=2), os.environ["AZURE_OPENAI_DEPLOYMENT"]


def shipment_list_text():
    import pymssql
    server = os.environ["SQL_SERVER"].strip()
    if not server.endswith(".database.windows.net"):
        server += ".database.windows.net"
    conn = pymssql.connect(server=server, user=os.environ["SQL_USER"], password=os.environ["SQL_PASSWORD"],
                           database=os.environ["SQL_DATABASE"], port=1433, tds_version="7.4", timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT s.shipment_id, s.status, s.ship_date, s.planned_delivery_date, l.lane_name, c.carrier_name, "
                "s.equipment_type, s.quoted_cost_usd FROM freight.shipments s "
                "JOIN freight.lanes l ON l.lane_id = s.lane_id JOIN freight.carriers c ON c.carrier_id = s.carrier_id "
                "WHERE s.status IN ('Booked', 'In Transit') ORDER BY s.planned_delivery_date")
    rows = cur.fetchall()
    conn.close()
    lines = ["shipment_id,status,ship_date,planned_delivery_date,lane,carrier,equipment,quoted_cost_usd"]
    for r in rows:
        r = [float(v) if isinstance(v, Decimal) else v for v in r]
        lines.append(",".join(str(v) for v in r))
    return "\n".join(lines)


def run_baseline(scenario, client, model, shipments_csv):
    system = (f"You are a freight risk analyst assistant for a US logistics team. Today is {TODAY.isoformat()}. "
              "Answer the question as helpfully as you can. Here is the list of active shipments:\n" + shipments_csv)
    start = time.time()
    kwargs = dict(model=model, messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": scenario["question"]}],
                  temperature=0.1, max_tokens=900)
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if "max_tokens" in str(exc).lower() or "temperature" in str(exc).lower():
            kwargs.pop("temperature"); kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    latency = int((time.time() - start) * 1000)
    answer = resp.choices[0].message.content or ""
    claims, unsupported, rate = grounding(answer, [shipments_csv], scenario["question"])
    corr, missing = correctness(scenario.get("facts", []), answer)
    return {
        "mode": "baseline_no_tools", "scenario": scenario["id"], "group": scenario["group"],
        "question": scenario["question"], "tools_called": "", "tool_calls": 0, "failed_tool_calls": 0,
        "tool_selection_ok": None, "arguments_ok": None, "completed": True, "hit_turn_cap": False, "turns": 1,
        "tokens": int(resp.usage.total_tokens) if getattr(resp, "usage", None) else None, "latency_ms": latency,
        "numeric_claims": claims, "unsupported_claims": len(unsupported),
        "unsupported_examples": ", ".join(unsupported[:6]), "grounding_rate": round(rate, 3),
        "correctness": corr, "missing_facts": missing,
        "run_id": None, "answer": answer,
    }


# ---------------------------------------------------------------------------
# Summary and outputs
# ---------------------------------------------------------------------------
def summarize(results):
    rows = []
    for mode, sub in results.groupby("mode"):
        claims = int(sub["numeric_claims"].sum())
        unsupported = int(sub["unsupported_claims"].sum())
        tool_sel = sub["tool_selection_ok"].dropna()
        args = sub["arguments_ok"].dropna()
        rows.append({
            "mode": mode, "scenarios": len(sub),
            "tool_selection_accuracy": round(float(tool_sel.mean()), 3) if len(tool_sel) else None,
            "argument_accuracy": round(float(args.mean()), 3) if len(args) else None,
            "task_completion_rate": round(float(sub["completed"].mean()), 3),
            "numeric_claims": claims, "unsupported_claims": unsupported,
            "grounding_rate": round((claims - unsupported) / claims, 3) if claims else 1.0,
            "correctness_rate": round(float(sub["correctness"].dropna().mean()), 3) if sub["correctness"].notna().any() else None,
            "fully_correct_answers": f"{int((sub['correctness'] == 1).sum())}/{int(sub['correctness'].notna().sum())}",
            "avg_turns": round(float(sub["turns"].mean()), 2),
            "avg_tool_calls": round(float(sub["tool_calls"].mean()), 2),
            "avg_tokens": round(float(sub["tokens"].dropna().mean()), 0) if sub["tokens"].notna().any() else None,
            "avg_latency_s": round(float(sub["latency_ms"].mean()) / 1000, 2),
            "turn_cap_hits": int(sub["hit_turn_cap"].sum()),
            "pass_rate": round(float(sub["passed"].mean()), 3) if "passed" in sub else None,
        })
    return pd.DataFrame(rows)


def sql_literal(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "NULL"
    if isinstance(v, bool):
        return "1" if v else "0"
    return "N'" + str(v).replace("'", "''") + "'"


def write_sql(results, batch, path):
    cols = ["batch_id", "evaluated_at_utc", "mode", "scenario", "scenario_group", "question", "tools_called",
            "tool_calls", "failed_tool_calls", "tool_selection_ok", "arguments_ok", "completed", "hit_turn_cap",
            "turns", "tokens", "latency_ms", "numeric_claims", "unsupported_claims", "unsupported_examples",
            "grounding_rate", "correctness", "missing_facts", "repetition", "passed", "run_id", "answer"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines = ["/* Phase 5 agent evaluation - generated by evaluate_agent.py */", "SET NOCOUNT ON;", "GO",
             """IF OBJECT_ID('freight.agent_evaluation') IS NULL
CREATE TABLE freight.agent_evaluation (
    eval_id INT IDENTITY(1,1) PRIMARY KEY, batch_id VARCHAR(40) NOT NULL, evaluated_at_utc DATETIME2 NOT NULL,
    mode VARCHAR(30) NOT NULL, scenario VARCHAR(10) NOT NULL, scenario_group VARCHAR(60) NULL,
    question NVARCHAR(1000) NULL, tools_called NVARCHAR(400) NULL, tool_calls INT NULL, failed_tool_calls INT NULL,
    tool_selection_ok BIT NULL, arguments_ok BIT NULL, completed BIT NULL, hit_turn_cap BIT NULL, turns INT NULL,
    tokens INT NULL, latency_ms INT NULL, numeric_claims INT NULL, unsupported_claims INT NULL,
    unsupported_examples NVARCHAR(400) NULL, grounding_rate DECIMAL(5,3) NULL, run_id VARCHAR(40) NULL,
    answer NVARCHAR(MAX) NULL);""", "GO",
             """IF COL_LENGTH('freight.agent_evaluation', 'correctness') IS NULL
ALTER TABLE freight.agent_evaluation ADD correctness DECIMAL(5,3) NULL, missing_facts NVARCHAR(400) NULL;""", "GO",
             """IF COL_LENGTH('freight.agent_evaluation', 'repetition') IS NULL
ALTER TABLE freight.agent_evaluation ADD repetition INT NULL, passed BIT NULL;""", "GO"]
    values = []
    for r in results.to_dict("records"):
        rec = {**r, "batch_id": batch, "evaluated_at_utc": now, "scenario_group": r["group"]}
        values.append("(" + ", ".join(sql_literal(rec.get(c)) for c in cols) + ")")
    for i in range(0, len(values), 100):
        lines += [f"INSERT INTO freight.agent_evaluation ({', '.join(cols)}) VALUES\n" + ",\n".join(values[i:i + 100]) + ";", "GO"]
    lines += [f"SELECT mode, COUNT(*) AS scenarios, AVG(CAST(grounding_rate AS FLOAT)) AS avg_grounding, "
              f"AVG(CAST(correctness AS FLOAT)) AS avg_correctness, "
              f"SUM(unsupported_claims) AS unsupported_claims FROM freight.agent_evaluation "
              f"WHERE batch_id = '{batch}' GROUP BY mode;", "GO"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the freight agent against a fixed scenario set")
    parser.add_argument("--agent-only", action="store_true", help="Skip the no-tools baseline")
    parser.add_argument("--only", nargs="*", help="Scenario IDs to run, e.g. A1 C1")
    parser.add_argument("--repeat", type=int, default=1, help="Run each scenario N times to measure consistency")
    args = parser.parse_args()

    scenarios = [s for s in SCENARIOS if not args.only or s["id"] in args.only]
    batch = f"{datetime.now(timezone.utc):%Y%m%d%H%M}-{uuid.uuid4().hex[:4]}"
    print(f"Evaluation batch {batch}: {len(scenarios)} scenarios")
    print("Building the answer key from the tools directly (no model calls) ...")
    for s in scenarios:
        try:
            s["facts"] = facts_for(s["id"], oracle)
        except Exception as exc:
            s["facts"] = []
            print(f"  ! answer key for {s['id']} unavailable: {str(exc)[:150]}")
    print("")
    results = []

    reps = max(1, args.repeat)
    for rep in range(1, reps + 1):
        if reps > 1:
            print(f"--- repetition {rep}/{reps} ---")
        for s in scenarios:
            try:
                rec = run_agent_once(s, f"{batch}-r{rep}")
                row = score_agent(s, rec)
                row["repetition"] = rep
                row["passed"] = bool(row["tool_selection_ok"] and row["completed"] and row["unsupported_claims"] == 0
                                     and row["correctness"] in (None, 1.0))
                corr = "n/a" if row["correctness"] is None else f"{row['correctness']:.0%}"
                print(f"  agent    {s['id']}  {'PASS' if row['passed'] else 'CHECK':<5}  "
                      f"tools: {row['tools_called'] or '(none)':<60.60}  "
                      f"grounded {row['grounding_rate']:.0%}  correct {corr}  {row['turns']} turns  {row['tokens']} tok")
                if row["unsupported_examples"]:
                    print(f"           unsupported: {row['unsupported_examples']}")
                if row["missing_facts"]:
                    print(f"           missing: {row['missing_facts']}")
                if not row["tool_selection_ok"]:
                    print(f"           tool selection failed")
                results.append(row)
            except Exception as exc:
                print(f"  agent    {s['id']}  ERROR  {str(exc)[:200]}")

    if not args.agent_only:
        print("\nBaseline (same model, no tools, given the active-shipment list):")
        client, model = make_client()
        csv_text = shipment_list_text()
        for s in scenarios:
            try:
                row = run_baseline(s, client, model, csv_text)
                row["repetition"] = 1
                row["passed"] = row["correctness"] in (None, 1.0) and row["unsupported_claims"] == 0
                corr = "n/a" if row["correctness"] is None else f"{row['correctness']:.0%}"
                print(f"  baseline {s['id']}  grounded {row['grounding_rate']:.0%}  correct {corr}"
                      + (f"  missing: {row['missing_facts'][:90]}" if row["missing_facts"] else ""))
                results.append(row)
            except Exception as exc:
                print(f"  baseline {s['id']}  ERROR  {str(exc)[:200]}")

    if not results:
        sys.exit("No results - check that freight_agent.py runs on its own first.")
    df = pd.DataFrame(results)
    summary = summarize(df)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "agent_evaluation.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "agent_evaluation_summary.csv"), index=False)
    write_sql(df, batch, os.path.join(OUT_DIR, "evaluation_load.sql"))

    print("\n=== Summary ===")
    print(summary.drop(columns=["scenarios"]).to_string(index=False))
    agent_rows = df[df["mode"] == "agent"]
    if reps > 1 and len(agent_rows):
        per = agent_rows.groupby("scenario")["passed"].agg(["sum", "count"])
        consistent = int((per["sum"] == per["count"]).sum())
        print(f"\nConsistency over {reps} runs: {consistent}/{len(per)} scenarios passed every run")
        for sid, r in per.iterrows():
            if r["sum"] < r["count"]:
                print(f"  {sid}: passed {int(r['sum'])}/{int(r['count'])}")
    print(f"\nPer-scenario detail: {OUT_DIR}/agent_evaluation.csv")
    print(f"Load into Azure SQL: run {OUT_DIR}/evaluation_load.sql in VS Code")


if __name__ == "__main__":
    main()
