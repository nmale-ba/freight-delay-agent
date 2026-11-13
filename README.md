# Predictive Freight Cost & Delay Agent (Azure)

An agentic AI logistics assistant that predicts shipment delays and freight costs, quantifies financial exposure, and recommends corrective actions. It is built on Azure SQL, Python/XGBoost and Microsoft Foundry Agent Service, with Power BI for reporting.

> **Data disclosure:** All carriers, shipments and fuel prices in Phase 1 are **synthetic**, and the carrier names are fictional. Phase 3 replaces weather, routing, fuel and FX with live public APIs. Port congestion stays simulated and is labeled as such.

## Architecture decisions

| Decision | Choice | Reason |
|---|---|---|
| Database | New `freight` schema inside the existing Azure SQL database | No new Azure resource or cost, and the schema keeps this project cleanly separated |
| Execution environment | VS Code connected to Azure SQL (MSSQL extension) | Load data by running a .sql file, with no Python driver needed for loading |
| SQL driver | `pymssql` (not `pyodbc`) | Installs with pip alone, with no ODBC driver to install (the earlier blocker) |
| Tool execution | Agent tools run as Python in the same process as the agent loop | Foundry function-calling returns tool requests to the client, so Azure Functions are optional and can come later |
| Code structure | Every script is self-contained | Avoids cross-file `ImportError` issues |
| Data design | Delays and costs generated from known rules | Lets Phase 2 test whether the model recovers known effects, against a known ceiling |

## Phases

1. **Data foundation** *(this release)*: schema, synthetic data, load to Azure SQL
2. **Prediction engine**: XGBoost delay classifier and cost regressor, evaluated on held-out data; results written to `shipment_predictions`
3. **External data**: Open-Meteo weather, OSRM routing, EIA diesel and Frankfurter FX, written to `route_conditions` and `fuel_prices`
4. **Agent**: four tools on Foundry Agent Service, with a 6-turn cap, timeouts and logging to `agent_runs` / `agent_tool_calls`
5. **Evaluation**: Scenarios A/B/C, the LLM-only agent vs. the tool-augmented agent, tool-call accuracy, completion rate and latency
6. **Power BI dashboard and demo**

## Files

| File | Purpose |
|---|---|
| `generate_freight_data.py` | Writes `data/*.csv` (seeded, reproducible) |
| `freight_create_and_load.sql` | **Run in VS Code**: creates the `freight` schema (10 tables, 2 views) and loads all company data |
| `data/*.csv` | Company data: carriers, lanes, fuel_prices, shipments (`ground_truth.csv` is for evaluation only; do not load it) |
| `fetch_external_data.py` | Calls OSRM, Open-Meteo, Frankfurter and EIA (all free), then writes `external/*.csv` and `external/external_data_load.sql` |
| `load_to_azure_sql.py` | Optional Python alternative to the .sql loader (pymssql) |

## Data model (schema `freight`)

| Table / view | Grain | Populated in |
|---|---|---|
| `carriers` | One row per carrier (6; C06 quotes in CAD) | Phase 1 |
| `lanes` | One row per origin→destination lane (12), with coordinates for weather lookups | Phase 1 |
| `fuel_prices` | Weekly diesel price (simulated now, EIA later) | Phase 1 / 3 |
| `shipments` | 1,200 delivered + 100 active (Booked / In Transit) | Phase 1 |
| `route_conditions` | Lane × date route-wide weather severity (worst of 5 points along the OSRM route) | Phase 3 |
| `fx_rates` | Daily USD→CAD exchange rates (Frankfurter/ECB) | Phase 3 |
| `diesel_prices_eia` | Weekly US retail diesel price (EIA), the live market reference | Phase 3 |
| `shipment_predictions` | Model output per active shipment | Phase 2 |
| `agent_runs`, `agent_tool_calls` | One row per agent run / per tool call (evaluation evidence) | Phase 4–5 |
| `v_carrier_lane_performance` | Historical on-time rate, delay days and cost overrun by carrier × lane | Feeds `get_carrier_performance()` |
| `v_active_shipments` | Active shipments joined to lane and carrier details | Feeds `get_shipment_context()` |

## External APIs (all free)

| API | Key | Used for | Limits |
|---|---|---|---|
| OSRM public demo | None | Route distance, drive time, route points | Fair use, about 1 request/sec |
| Open-Meteo | None | Weather for the past 7 and next 16 days at 5 points per route | Free for non-commercial use |
| Frankfurter (ECB) | None | USD→CAD for the Canadian carrier | None stated |
| EIA Open Data v2 | Free registration | Weekly diesel price for the fuel surcharge | Optional; the script skips it if no key is set |

The simulated `fuel_prices` table stays, because it is the series the historical shipments were generated from. `diesel_prices_eia` is the live reference the agent uses for current fuel-surcharge estimates.

## How the synthetic data was generated

The delay probability is a logistic function of: carrier reliability, weather severity (0–3), port congestion (port-origin lanes only), winter exposure (Dec–Feb, northern/mountain lanes), distance, heavy loads (>40,000 lb), and two deliberately weak carrier–lane pairs (Ironhorse on LA→Chicago, Redline on LA→Seattle).

Costs are built as linehaul (rate per mile × equipment multiplier × market noise) plus a DOE-style fuel surcharge `(diesel − $1.25) / 6.5 mpg` per mile, a heavy-load adjustment and port drayage for intermodal moves. Delayed shipments pick up detention accessorials.

Phase 1 sanity checks (seed 42):
- Historical delay rate: 19.2%
- Carrier on-time rates range from 63% (Ironhorse) to 93% (Pacific Crest)
- A quick XGBoost check gives a test ROC-AUC of 0.77. An "oracle" using the true generating probability reaches 0.81, so 0.81 is the ceiling any model can reach on this data
- 11 in-transit shipments are already past their planned delivery date as of 2026-11-07, which gives the agent real exceptions to find

`data/ground_truth.csv` holds the true delay probability for every historical shipment. It is used only for evaluation and is never loaded into SQL or used as a model feature.
