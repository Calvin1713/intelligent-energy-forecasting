"""
Intelligent Energy Consumption Forecasting and Anomaly Detection System
Primary Streamlit Application Entrypoint
"""

import os
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Energy Intelligence Platform | ML Forecasting & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom Enterprise Design System
css_path = os.path.join(os.path.dirname(__file__), 'assets', 'styles', 'main.css')
if os.path.exists(css_path):
    with open(css_path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Import Components & Pages
from components.sidebar import render_sidebar
from pages.overview import render_page as render_overview
from pages.consumption import render_page as render_consumption
from pages.forecasting import render_page as render_forecasting
from pages.anomalies import render_page as render_anomalies
from pages.peak_demand import render_page as render_peak_demand
from pages.model_performance import render_page as render_model_performance
from pages.recommendations import render_page as render_recommendations
from pages.prediction_lab import render_page as render_prediction_lab
from pages.reports import render_page as render_reports

# 4. Render Global Sidebar Controls
filters = render_sidebar()

# 5. Define Page Routing Wrappers
def page_overview():
    render_overview(filters)

def page_consumption():
    render_consumption(filters)

def page_forecasting():
    render_forecasting(filters)

def page_anomalies():
    render_anomalies(filters)

def page_peak_demand():
    render_peak_demand(filters)

def page_model_performance():
    render_model_performance(filters)

def page_recommendations():
    render_recommendations(filters)

def page_prediction_lab():
    render_prediction_lab(filters)

def page_reports():
    render_reports(filters)

# 6. Streamlit Navigation Structure
nav_sections = {
    "Executive & Strategy": [
        st.Page(page_overview, title="Executive Overview", icon="🏢", default=True),
        st.Page(page_recommendations, title="Energy Recommendations", icon="💡"),
        st.Page(page_reports, title="Reports & Data Explorer", icon="📄")
    ],
    "Analytics & Operations": [
        st.Page(page_consumption, title="Consumption Analytics", icon="📈"),
        st.Page(page_forecasting, title="Energy Forecasting", icon="🔮"),
        st.Page(page_anomalies, title="Anomaly Detection Centre", icon="🚨"),
        st.Page(page_peak_demand, title="Peak Demand Management", icon="⚡")
    ],
    "Machine Learning Studio": [
        st.Page(page_model_performance, title="Model Evaluation & Benchmarks", icon="🤖"),
        st.Page(page_prediction_lab, title="Interactive Prediction Lab", icon="🧪")
    ]
}

nav = st.navigation(nav_sections)
nav.run()
