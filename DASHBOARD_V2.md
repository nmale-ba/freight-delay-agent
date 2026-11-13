# Freight Risk Dashboard v2: report layout and setup

`tools/build_pbir.py` rebuilt the report pages on the existing semantic model; the data and the 44 measures are unchanged. It fixed the following problems in v1:
- The donut on page 1 counted ShipmentHistory rows (1,200 per slice).
- The combo chart on page 4 mixed ActiveShipments and SwitchSensitivity.
- Two visuals on page 2 were empty.
- Charts drilled through date hierarchies.
- The cards on page 5 averaged tokens and latency across agent *and* baseline rows.

Every visual now uses a single table, or measures from `_Measures`. Gray page background, white rounded tiles, a title and subtitle on every visual.

| Page | KPI cards | Visuals |
|---|---|---|
| 1. Shipment Risk Overview | Active, Already Late, Predicted At Risk, % Needing Attention, Total Exposure, At-Risk Exposure | Slicers for window, carrier and lane · donut by risk group · stacked column by window × risk group · exposure by lane · attention table (Low risk excluded, gradient on delay probability) |
| 2. Carrier Performance | Historical Shipments, On-Time %, Delay Rate, Avg Delay Days, Avg Cost Overrun, Weak Lanes | On-time % by carrier · delay rate by ship month · carrier × lane on-time heatmap · weak-lanes table |
| 3. Route Conditions & Fuel | Worst Severity, Lanes Moderate+, Diesel Price, Surcharge/mile, Diesel 12-wk % | Lane × day severity heatmap (Forecast only) · rain and snow by lane · diesel price line |
| 4. Recommended Actions | Switches, Risk Before/After, Freight Cost Change, Lateness Cost Avoided, Net Benefit | Switch table (`has_switch` = 1, sorted by net benefit) · actions by type · late-penalty sensitivity combo |
| 5. AI Agent & Model Evaluation | Agent and baseline correctness, lift, pass rate, grounding, tokens, seconds · ROC-AUC vs baseline | Correctness by scenario group (agent vs no tools) · XGBoost vs baseline metrics · per-scenario agent table |

**Required once after uploading:** set measure format strings. Percent (0–1 dp): % Shipments Needing Attention, Avg Delay Probability, Historical On-Time %, Historical Delay Rate, Diesel Change 12 Weeks %, Avg Risk Before/After Switch, Agent Correctness, Baseline Correctness, Agent Pass Rate, Agent Grounding Rate. Currency (0 dp): Total Exposure, Extra Carrier Charges, Business Cost of Lateness, At-Risk Exposure, Avg/Total Cost Overrun, Switch Freight Cost Change, Expected Lateness Cost Avoided, Net Benefit of Switches. Currency (2 dp): Latest Diesel Price, Latest Fuel Surcharge per Mile. Decimal: Prediction/Baseline ROC-AUC (3 dp), Avg Delay Days (1), Avg Response Time (1), Avg Risk Reduction (1), Correctness Lift (0), Avg Tokens (0). Also hide the `_Measures` column inside the `_Measures` table.
