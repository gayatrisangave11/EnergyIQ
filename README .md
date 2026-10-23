# EnergyIQ — Smart Energy Consumption Optimizer

**EnergyIQ** is an end-to-end, privacy-first smart energy analytics platform and decision-support system built with Python and Streamlit. It ingests interval consumption data from smart meters or building energy management systems (BEMS), cleans and validates the dataset, performs hourly and statistical analytics, models energy-shifting scenarios, identifies off-hours ghost loads, and produces executive mitigation plans using a Groq-powered large language model.

## 1. Problem Statement

Commercial facilities, industrial plants, and smart buildings generate vast amounts of interval meter data. However, facility managers face critical operational challenges:

1. **Dirty and Inconsistent Meter Data:** Missing timestamps, duplicate records, unannounced unit conversions, cumulative meter resets, and irregular sampling intervals make raw data unsuitable for immediate analysis.

2. **Hidden Off-Hours Waste (Ghost Load):** Equipment, HVAC systems, lighting, or IT infrastructure frequently consume high baseline power during unoccupied nights or weekends without detection.

3. **Expensive Peak-Demand Charges:** Shifting flexible loads away from peak tariff windows can yield significant cost savings, but managers lack quick, reliable tools to simulate "what-if" shifting scenarios under custom tariff models.

4. **Actionability Gap:** Standard analytics tools produce dense graphs without clear, evidence-grounded executive summaries or verifiable mitigation steps.

## 2. Proposed Solution

**EnergyIQ** bridges the gap between raw interval data and facility action through a structured 5-block processing pipeline:

```
[ Block 1: Foundation ] ➔ [ Block 2: Hourly Analytics ] ➔ [ Block 3: Visual Analytics ] ➔ [ Block 4: Simulator ] ➔ [ Block 5: LLM Executive Report ]
 Clean & Map Columns       Peaks & IsolationForest        Timeline, Heatmap & Schedule    What-If Shifting & ROI      Evidence-Grounded Mitigation

```

1. **Block 1 — Data Validation & Cleaning Foundation:** Handles upload errors, non-standard delimiters, timezone offsets, duplicate timestamps, cumulative meter resets, and missing-data flags without silent data destruction.

2. **Block 2 — Granular Hourly Analytics & Anomaly Engine:** Infers sampling rates, calculates time-weighted power averages or summed energy, extracts statistical peak percentiles, and fits an unsupervised IsolationForest model to flag behavioral anomalies.

3. **Block 3 — Interactive Visualizations & Ghost Load Heatmap:** Features an interactive Plotly timeline with range sliders, a 7×24 day/hour heatmap, and an operating schedule engine that contrasts in-schedule versus off-hours load profiles.

4. **Block 4 — What-If Energy Shifting & ROI Simulator:** Models load-shifting from peak tariff hours to off-peak periods while guaranteeing energy conservation, with sensitivity breakdowns and an illustrative Green Sync renewable-share scenario.

5. **Block 5 — Evidence-Grounded Executive Mitigation Plan:** Aggregates facts, indicators, and simulation outputs into a strict JSON evidence bundle (`<evidence>`), then prompts the Groq API (`openai/gpt-oss-120b`) to draft three concrete, verifiable, and non-hallucinatory mitigation steps.

## 3. Key Features & Design Rules

* **Strict Single-Source of Truth (`session_state["analysis"]`):** Every downstream block reads the cleaned dataset through a centralized bundle. Modifying the mapping or re-uploading automatically invalidates stale downstream block states (`block2_` through `block5_`).

* **Zero Silent Data Inventions:** Missing values are never treated as zero, and long gaps are never silently bridged. Negative meter deltas (resets) are set to missing, while negative power/energy values are explicitly flagged (e.g., net metering/generation).

* **Strict Unit Separation:** Costs, carbon emissions, and shifting calculations require verified energy ($kWh$) or power ($kW$) inputs. Unknown units or non-standard measurements are flagged and excluded from financial and carbon equations.

* **API Key Security:** API keys (e.g., `GROQ_API_KEY`) are read strictly from environment variables or Streamlit secrets (`.env` or `.streamlit/secrets.toml`). Keys are never displayed, stored in session state, logged, or transmitted to the client browser.

* **Strict Prompt Guarding against Hallucinations:** The LLM receives only structured facts and indicators inside `<evidence>` tags. The system prompt forces clear taxonomy tags: `[Observed]`, `[Statistical indicator]`, `[Simulated scenario]`, and `[Hypothesis]`.

## 4. Project Architecture & Repository Structure

```
EnergyIQ/
├── app.py              # Main Streamlit application containing Blocks 1–5
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules for virtualenvs, keys, and caches
└── README.md           # Project documentation

```

## 5. Installation & Setup

### Prerequisites

* Python **3.9+** or higher

* A Groq API Key 

### 1. Clone the Repository & Setup Environment

```
git clone https://github.com/your-username/EnergyIQ.git
cd EnergyIQ

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

```

### 2. Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt

```

### 3. Configure API Credentials

Create a `.env` file in the project root directory (or use Streamlit secrets at `.streamlit/secrets.toml`):

```
cp .env.example .env

```

Edit `.env` and insert your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

```

### 4. Run the Streamlit Application

```
streamlit run app.py

```

Open your browser at `http://localhost:8501`.

## 6. Data Requirements & Supported Formats

EnergyIQ accepts `.csv` files containing time-series energy or power data.

### Supported Input Structures:

* **Timestamps:** Single combined column (`YYYY-MM-DD HH:MM:SS`, ISO 8601, or UTC string with offsets) or split `Date` and `Time` columns. Supports Unix Epoch timestamps (seconds or milliseconds).

* **Measurement Types:**

  * **Power:** Instantaneous demand ($W$, $kW$, $MW$).

  * **Energy per interval:** Interval readings ($Wh$, $kWh$, $MWh$).

  * **Cumulative Meter Reading:** Running meter totals ($Wh$, $kWh$, $MWh$) — automatically handles meter resets.

  * **Unknown / Unspecified:** Allows exploratory analysis while safely disabling financial and carbon models.

* **Meters / Zones:** Optional column identifying building zones, circuits, equipment IDs, or individual meters. Supports per-meter filtering or explicit timestamp aggregation.

## 7. Detailed Processing Blocks

### Block 1: Validation & Cleaning

* Deduplicates timestamps via user policy (*Keep first*, *Keep last*, *Average*, *Flag only*).

* Detects irregular interval spacing and labels gaps.

* Formats decimal separators (`,` or `.`) and handles optional forward-fill or linear interpolation up to user limits.

### Block 2: Hourly Analytics & Anomaly Engine

* Aggregates fine or irregular intervals to hourly standard frames using coverage thresholds ($\ge 80\%$).

* Computes time-weighted average power ($kW$) and total interval energy ($kWh$).

* Extracts percentile-based statistical peaks (e.g., $95^{\text{th}}$ percentile).

* Fits an `IsolationForest` model on trigonometric cyclical features (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`) and load values to isolate behavioral anomalies.

### Block 3: Visual Analytics & Ghost Load

* **Plotly Timeline:** Multi-trace line plot with interactive range slider, break handling over real gaps, and optional min-max display decimation for fast rendering.

* **Ghost Load Heatmap:** 7-day $\times$ 24-hour load profile matrix pinpointing off-hours baseload patterns.

* **Operating Schedule Engine:** User-defined active hours contrast in-schedule vs. off-hours load profiles, quantifying potential vampire draw.

### Block 4: Energy-Shifting & ROI Simulator

* Models moving a user-defined fraction ($0\% - 100\%$) of expensive-window consumption to off-peak periods ($1 - 6$ hours later).

* Verifies **energy conservation** ($\text{Energy}_{\text{baseline}} == \text{Energy}_{\text{simulated}}$).

* Calculates gross financial savings against peak vs. baseline tariffs.

* Evaluates emissions timing deltas and provides an interactive **Green Sync** renewable supply share dial.

### Block 5: Groq AI Executive Report

* Generates a JSON bundle (<12 KB) containing dataset statistics, flags, schedule ratios, and simulation outputs.

* Transmits the evidence bundle to Groq (`openai/gpt-oss-120b`).

* Renders 3 structured, actionable mitigation steps containing:

  * **Action:** Concrete step for the facility manager.

  * **Evidence:** Categorized evidence tag (`[Observed]`, `[Statistical indicator]`, `[Simulated scenario]`, or `[Hypothesis]`).

  * **Expected Benefit:** Qualified, realistic savings estimate.

  * **Verification:** Concrete test method to confirm resolution.

## 8. Security & Privacy

* **Local Data Ingestion:** Meter data processed during the session remains local to the Streamlit runtime.

* **No Raw Data Uploaded to LLM:** Raw CSV rows are **never** sent to the Groq API. Only a compact, anonymized statistical evidence dictionary is transmitted.

* **Zero Secret Leakage:** Environment keys and Streamlit secrets are encapsulated on the server side.

## 9. License

This project is released under the [MIT License](LICENSE).