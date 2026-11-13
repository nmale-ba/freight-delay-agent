"""
freight_chat_app.py
Predictive Freight Cost & Delay Agent - chat app (runs locally in your browser)

Self-contained (no imports from other project files). Each question is answered by running
`freight_agent.py --json-io` as a separate process, so the app uses exactly the tested agent with
all of its guardrails (6-turn cap, token budgets, validation, logging).

Requires:  pip install streamlit pandas openpyxl python-docx
Uses the same environment variables as freight_agent.py (SQL_* and AZURE_OPENAI_* or LLM_PROVIDER=github).

Run (from the project folder, in the terminal where the variables are set):
    python3 -m streamlit run freight_chat_app.py

Features
    - conversation memory: follow-ups like "now only Ironhorse" or "summarize this" work
    - one-click report prompts (weekly risk, carrier scorecard, weather, switches, fuel)
    - "How this answer was produced": every tool the agent called, with its inputs
    - per answer: download the answer (.md) and the data behind it (.xlsx, one sheet per table)
    - whole session: download a Word report (.docx) or Markdown, and all data in one Excel file
"""

import io
import json
import os
import re
import subprocess
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

AGENT_SCRIPT = os.environ.get("FREIGHT_AGENT_SCRIPT", "freight_agent.py")
AGENT_TIMEOUT_S = 240
HISTORY_TURNS = 4            # user+assistant pairs sent back to the agent for follow-ups

QUICK_REPORTS = {
    "Weekly risk report": "Give me a weekly risk report: shipments that are already late, shipments predicted "
                          "to be delayed in the next 7 days, the total financial exposure, and the recommended actions.",
    "Carrier scorecard": "Create a carrier scorecard as a table: each carrier's overall on-time rate, its weakest "
                         "lane and that lane's on-time rate, average delay days and average cost overrun. "
                         "Which carriers should we review?",
    "Weather outlook": "Which lanes have moderate or severe weather in the next 7 days, and which booked "
                       "shipments on those lanes are most at risk?",
    "Carrier switches": "Which carrier switches do you recommend, with the risk and cost change for each? "
                        "Then tell me how that changes if a late delivery costs us $2,500.",
    "Fuel cost impact": "How has the diesel price moved recently, and what is the current fuel surcharge per mile?",
    "Summarize this session": "Summarize our conversation so far as an executive summary with key figures and "
                              "the top 3 actions.",
}
TOOL_LABELS = {"get_shipment_context": "Shipments", "get_route_conditions": "Route weather",
               "get_carrier_performance": "Carrier history", "calculate_risk_and_recommendation": "Risk model"}


# ---------------------------------------------------------------------------
# Talking to the agent
# ---------------------------------------------------------------------------
def ask_agent(question, history):
    payload = {"question": question, "history": history, "scenario": "CHAT"}
    try:
        proc = subprocess.run([sys.executable, AGENT_SCRIPT, "--json-io"], input=json.dumps(payload),
                              capture_output=True, text=True, timeout=AGENT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"answer": "The agent took too long to respond. Please try again.", "tool_calls": [], "error": True}
    for line in proc.stdout.splitlines():
        if line.startswith("JSONRESULT "):
            return json.loads(line[len("JSONRESULT "):])
    detail = (proc.stdout + "\n" + proc.stderr).strip()[-600:]
    return {"answer": "The agent could not run. Check that the environment variables are set in the terminal "
                      "where you started the app.\n\n```\n" + detail + "\n```", "tool_calls": [], "error": True}


def show_md(text):
    """Streamlit renders $...$ as LaTeX math, which garbles answers like "$8,992 and $40,085".
    Escape every dollar sign for display; downloads keep the original text."""
    return re.sub(r"(?<!\\)\$", r"\\$", str(text or ""))


def history_for_agent():
    msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if not m.get("error")]
    return msgs[-HISTORY_TURNS * 2:]


# ---------------------------------------------------------------------------
# Turning tool results into tables
# ---------------------------------------------------------------------------
def _flatten_scalars(obj, prefix=""):
    rows = []
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows += _flatten_scalars(v, key + ".")
        elif not isinstance(v, list):
            rows.append({"field": key, "value": v})
    return rows


def tables_from_tool_calls(tool_calls):
    """One DataFrame per list of records found in the tool results, plus a summary of scalar figures."""
    tables, summary = {}, []
    for i, call in enumerate(tool_calls, start=1):
        try:
            result = json.loads(call.get("result") or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict) or "error" in result:
            continue
        short = TOOL_LABELS.get(call.get("tool"), call.get("tool", "tool"))
        for row in _flatten_scalars(result):
            summary.append({"source": f"{i}. {short}", **row})

        def collect(obj, path):
            for k, v in obj.items():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    name = re.sub(r"[^A-Za-z0-9 ]", " ", f"{i} {short} {path}{k}").strip()
                    tables[name[:31]] = pd.DataFrame(v)
                elif isinstance(v, dict):
                    collect(v, f"{path}{k} ")
        collect(result, "")
    if summary:
        tables = {"Summary figures": pd.DataFrame(summary), **tables}
    return tables


def excel_bytes(tables):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not tables:
            pd.DataFrame({"note": ["No tabular data in this answer"]}).to_excel(writer, sheet_name="Data", index=False)
        used = set()
        for name, df in tables.items():
            sheet = name[:31]
            n = 2
            while sheet in used:
                sheet = f"{name[:28]}_{n}"
                n += 1
            used.add(sheet)
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(60, max(10, width + 2))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
DATA_NOTE = ("Shipments and carriers are synthetic (fictional carrier names); weather, routing, diesel and FX are "
             "live where loaded; port congestion is simulated. All figures come from the agent's tools.")


def session_pairs():
    pairs, pending = [], None
    for m in st.session_state.messages:
        if m["role"] == "user":
            pending = m
        elif m["role"] == "assistant" and pending is not None:
            pairs.append((pending, m))
            pending = None
    return pairs


def markdown_report():
    lines = [f"# Freight Risk Agent - session report", f"*Generated {datetime.now():%Y-%m-%d %H:%M}*", "",
             f"> {DATA_NOTE}", ""]
    for n, (q, a) in enumerate(session_pairs(), start=1):
        tools = ", ".join(TOOL_LABELS.get(c["tool"], c["tool"]) for c in a.get("tool_calls", [])) or "none"
        lines += [f"## {n}. {q['content']}", "", a["content"], "", f"*Tools used: {tools}*", ""]
    return "\n".join(lines)


def _add_markdown_to_docx(doc, text):
    """Minimal markdown: headings, bullets, numbered items, tables, **bold**."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows, i = [], i + 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Light Grid Accent 1"
            for j, h in enumerate(header):
                table.rows[0].cells[j].text = h.replace("**", "")
            for r in rows:
                cells = table.add_row().cells
                for j in range(min(len(header), len(r))):
                    cells[j].text = r[j].replace("**", "")
            continue
        if not line.strip():
            i += 1
            continue
        if line.startswith("#"):
            doc.add_heading(line.lstrip("#").strip().replace("**", ""), level=3)
        else:
            style = None
            content = line.strip()
            if re.match(r"^[-*•]\s+", content):
                style, content = "List Bullet", re.sub(r"^[-*•]\s+", "", content)
            elif re.match(r"^\d+[.)]\s+", content):
                style, content = "List Number", re.sub(r"^\d+[.)]\s+", "", content)
            p = doc.add_paragraph(style=style)
            for k, part in enumerate(re.split(r"\*\*", content)):
                run = p.add_run(part)
                run.bold = k % 2 == 1
        i += 1


def docx_report():
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading("Freight Risk Agent - Session Report", level=1)
    doc.add_paragraph(f"Generated {datetime.now():%B %d, %Y %H:%M}")
    note = doc.add_paragraph(DATA_NOTE)
    note.runs[0].italic = True
    for n, (q, a) in enumerate(session_pairs(), start=1):
        doc.add_heading(f"{n}. {q['content']}", level=2)
        _add_markdown_to_docx(doc, a["content"])
        tools = ", ".join(TOOL_LABELS.get(c["tool"], c["tool"]) for c in a.get("tool_calls", [])) or "none"
        p = doc.add_paragraph(f"Tools used: {tools}")
        p.runs[0].italic = True
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def session_tables():
    tables = {}
    for n, (_, a) in enumerate(session_pairs(), start=1):
        for name, df in tables_from_tool_calls(a.get("tool_calls", [])).items():
            tables[f"Q{n} {name}"[:31]] = df
    return tables


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Freight Risk Agent", page_icon="🚚", layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.header("Freight Risk Agent")
    provider = os.environ.get("LLM_PROVIDER", "azure")
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT") if provider == "azure" else os.environ.get("GITHUB_MODEL", "openai/gpt-4.1-mini")
    db_ok = all(os.environ.get(k) for k in ("SQL_SERVER", "SQL_DATABASE", "SQL_USER", "SQL_PASSWORD"))
    llm_ok = bool(os.environ.get("AZURE_OPENAI_API_KEY")) if provider == "azure" else bool(os.environ.get("GITHUB_TOKEN"))
    st.caption(f"Database: {'✅ configured' if db_ok else '❌ SQL_* variables missing'}")
    st.caption(f"Model: {'✅ ' + str(model) if llm_ok else '❌ model variables missing'}")

    st.subheader("Quick reports")
    for label, prompt in QUICK_REPORTS.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending = prompt

    st.subheader("Export this session")
    if session_pairs():
        try:
            st.download_button("Word report (.docx)", docx_report(), file_name="freight_agent_report.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               use_container_width=True)
        except ImportError:
            st.caption("Install python-docx for Word export.")
        st.download_button("Markdown report (.md)", markdown_report(), file_name="freight_agent_report.md",
                           use_container_width=True)
        st.download_button("All data (.xlsx)", excel_bytes(session_tables()), file_name="freight_agent_data.xlsx",
                           use_container_width=True)
    else:
        st.caption("Ask a question first.")
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(DATA_NOTE)

st.title("🚚 Freight Risk & Delay Agent")
st.caption("Ask about shipments, delay risk, weather, carriers, fuel and costs. Follow-up questions remember the "
           "conversation. Every figure comes from the agent's tools (XGBoost model + Azure SQL + live data).")

for idx, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(show_md(m["content"]))
        if m["role"] == "assistant" and not m.get("error"):
            calls = m.get("tool_calls", [])
            meta = f"{len(calls)} tool call(s) · {m.get('turns_used', '?')} turns · {m.get('total_tokens', 0):,} tokens · " \
                   f"{(m.get('latency_ms') or 0) / 1000:.1f}s"
            with st.expander(f"How this answer was produced ({meta})"):
                if not calls:
                    st.write("No tools were needed.")
                for c in calls:
                    st.markdown(f"**{TOOL_LABELS.get(c['tool'], c['tool'])}** (`{c['tool']}`) "
                                f"{'✅' if c.get('ok') else '❌ ' + str(c.get('error'))}")
                    st.code(c.get("arguments") or "{}", language="json")
            tables = tables_from_tool_calls(calls)
            col1, col2 = st.columns(2)
            col1.download_button("Download answer (.md)", m["content"], file_name=f"answer_{idx}.md", key=f"md{idx}")
            col2.download_button("Download data (.xlsx)", excel_bytes(tables), file_name=f"answer_{idx}_data.xlsx",
                                 key=f"xl{idx}", disabled=not tables)

typed = st.chat_input("e.g. Which booked shipments should switch carriers, and what does it cost?")
question = typed or st.session_state.pending
if question:
    st.session_state.pending = None
    history = history_for_agent()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(show_md(question))
    with st.chat_message("assistant"):
        with st.spinner("Checking shipments, weather, carriers and the risk model ..."):
            result = ask_agent(question, history)
    st.session_state.messages.append({"role": "assistant", "content": result.get("answer", ""),
                                      "tool_calls": result.get("tool_calls", []), "error": result.get("error", False),
                                      "turns_used": result.get("turns_used"), "total_tokens": result.get("total_tokens", 0),
                                      "latency_ms": result.get("latency_ms")})
    st.rerun()
