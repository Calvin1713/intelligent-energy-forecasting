# Intelligent Energy Consumption Forecasting and Anomaly Detection System Using Machine Learning

![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-179C52.svg)
![License](https://img.shields.io/badge/Project-M.Tech%20Data%20Science-7C3AED.svg)

---

## 🏢 Project Overview

The **Intelligent Energy Consumption Forecasting and Anomaly Detection System** is an enterprise-grade energy intelligence SaaS platform designed for facility directors, campus energy managers, building engineers, and energy analytics researchers. 

Built as an **M.Tech Data Science Capstone Project**, the platform shifts away from static academic charting into an operational, decision-support control center. It integrates multi-model empirical benchmarks, 24-hour forward predictive forecasting, multi-criteria anomaly surveillance with triage workflows, peak demand threshold tracking, and an interactive prediction lab for what-if scenario testing.

---

## 📊 Dataset & Scientific Methodology

### 1. Data Foundation
- **Source:** Building Data Genome Project 2 (BDG2), an open-source dataset curated by the Building and Urban Data Science (BUDS) Lab at the National University of Singapore.
- **Analyzed Portfolio:** 20 verified commercial office facilities across 5 campus sites (`Robin`, `Lamb`, `Shrew`, `Hog`, `Eagle`).
- **Temporal Horizon:** 2 full calendar years of continuous hourly electricity meter readings (Jan 1, 2016 00:00 to Dec 31, 2017 23:00).
- **Total Valid Records:** 350,880 hourly observations (17,544 readings per facility).
- **Measurement Unit:** Hourly electricity meter reading ($\text{kWh}$).

### 2. Feature Engineering Pipeline
The predictive models leverage a 15-dimensional feature space engineered to capture short-term persistence, diurnal cycles, and weekly seasonality:
- **Autoregressive Lags:**
  - `lag_1`: Demand in the immediately preceding hour (captures thermal persistence).
  - `lag_24`: Demand at the exact same hour on the previous day (captures daily periodicity).
  - `lag_168`: Demand at the exact same hour one week prior (captures weekly periodicity).
- **Rolling Statistical Aggregations:**
  - `rolling_mean_24`: Moving average over the preceding 24 hours.
  - `rolling_mean_168`: Moving average over the preceding 168 hours (7 days).
- **Trigonometric Cyclical Encodings:**
  - Hour of day: $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$
  - Day of week: $\sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$
- **Calendar & Categorical Attributes:**
  - `hour` (0–23), `day_of_week` (0–6), `day_of_month` (1–31), `month` (1–12), `is_weekend` (binary flag).
  - `building_id` (one-hot encoded categorical identifier).

### 3. Model Benchmark Results (Testing Set: 69,520 Observations)
Models were evaluated on the final 20% temporal split for each facility (Aug 9, 2017 to Dec 31, 2017):

| Rank | Model Architecture | MAE ($\text{kWh}$) | RMSE ($\text{kWh}$) | $R^2$ Score | Status |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **XGBoost Regressor** | **3.5734** | **7.6601** | **0.998321** | **Champion Model** |
| 🥈 | **Decision Tree** | 3.8771 | 9.1489 | 0.997605 | Benchmark |
| 🥉 | **Random Forest** | 5.3142 | 9.2131 | 0.997571 | Benchmark |
| 4 | **Linear Regression** | 6.4200 | 14.7178 | 0.993802 | Baseline Linear |
| 5 | **Seasonal Naïve (24h)** | 11.9045 | 40.4488 | 0.953183 | Persistence Baseline |

### 4. Anomaly Detection Methodology
Anomalies are detected using a dual-criteria strategy:
1. **Statistical Residual Multiplier:** Hourly observations where the absolute residual $|\text{Actual} - \text{Predicted}| \ge 3 \times \text{MAE}$ are flagged as high-confidence statistical anomalies.
2. **Isolation Forest:** Unsupervised tree isolation isolating multivariate density outliers in load space.
3. **Investigation Workflow:** An operational review system allowing operators to triage events into `Unreviewed`, `Under investigation`, `Reviewed`, and `Resolved`.

---

## 🏛️ Application Architecture

The platform follows a modular, production-ready architecture:

```
├── app.py                     # Primary Streamlit application entrypoint with st.navigation
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── pages/
│   ├── overview.py            # Page 1: Executive Overview & Portfolio KPIs
│   ├── consumption.py         # Page 2: Energy Consumption Analytics Workspace
│   ├── forecasting.py         # Page 3: 24-Hour Predictive Forecasting Center
│   ├── anomalies.py           # Page 4: Anomaly Detection & Investigation Centre
│   ├── peak_demand.py         # Page 5: Peak Demand Management & Load Shifting
│   ├── model_performance.py   # Page 6: ML Evaluation, Residuals & Feature Importance
│   ├── recommendations.py     # Page 7: Facility Efficiency Guidance & Tariff Calculator
│   ├── prediction_lab.py      # Page 8: Interactive Prediction Lab (Live XGBoost Inference)
│   └── reports.py             # Page 9: Data Explorer & Printable HTML Executive Report
├── components/
│   ├── sidebar.py             # Enterprise dark sidebar, brand logo & global filters
│   ├── kpi_cards.py           # Custom styled KPI metric cards & header banners
│   ├── charts.py              # Theme-aligned Plotly time-series, diurnal curves & heatmaps
│   └── alerts.py              # Callout banners, scientific notices & empty states
├── utils/
│   ├── data_loader.py         # Caching and validation layer for CSVs and Parquet
│   ├── preprocessing.py       # Temporal, cyclical, and lag feature computation
│   ├── prediction.py          # Dual-engine inference manager (precomputed + live)
│   ├── recommendations.py     # Recommendation categorization & tariff scenario calculator
│   └── report_generator.py    # Automated standalone HTML printable executive report
├── assets/
│   ├── logo/logo.svg          # Corporate SVG vector brand glyph
│   └── styles/main.css        # Enterprise design system stylesheet
├── dashboard_exports/         # Verified outputs generated during Google Colab training
├── saved_models/              # Trained models and pipeline artifacts
└── data/                      # Extracted BDG2 historical records (Parquet) & building metadata
```

---

## 🚀 Installation & Execution

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/Calvin/Intelligent_Energy_Forecasting.git
cd Intelligent_Energy_Forecasting

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Launch the Platform
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🔒 Scientific & Operational Disclaimers

1. **No Live Meter Connection:** This platform analyzes verified historical records from the Building Data Genome Project 2 (BDG2). It is not connected to live IoT energy meters or SCADA building automation systems.
2. **Statistical Anomaly vs. Mechanical Fault:** An anomaly denotes an unusual statistical deviation from model expectation; it serves as an investigative flag and does not confirm equipment failure or energy theft.
3. **Hypothetical Tariff Calculations:** Cost avoidance scenarios are illustrative modeling estimates based on user-entered utility rates ($\$/\text{kWh}$ and demand charges). Actual savings require on-site measurement and verification (M&V) according to IPMVP guidelines.

---

## 🎓 Academic Attribution
Developed for the Master of Technology (**M.Tech**) Data Science Program.  
Dataset: **Building Data Genome Project 2 (BDG2)**, BUDS Lab, National University of Singapore.
