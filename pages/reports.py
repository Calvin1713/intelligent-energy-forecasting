"""
Page 9 — Reports & Data Explorer
Comprehensive data exploration portal providing searchable multi-dataset tables,
one-click CSV audit exports, and a publication-grade HTML executive report generator.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import (
    load_metadata, load_building_summary, load_historical_consumption,
    load_model_metrics, load_feature_importance, load_forecast,
    load_predictions, load_anomalies, load_peak_periods, load_recommendations
)
from utils.report_generator import generate_html_executive_report
from components.kpi_cards import render_kpi_card, render_header_banner
from components.alerts import render_alert

def render_page(filters: dict):
    render_header_banner(
        title="Reports & Data Explorer Portal",
        subtitle="Searchable repository of all verified project datasets, multi-format export engines, and executive briefing generator.",
        badge_text="Data Governance",
        badge_type="teal"
    )

    df_meta = load_metadata()
    df_summary = load_building_summary()
    df_hist = load_historical_consumption()
    df_metrics = load_model_metrics()
    df_fi = load_feature_importance()
    df_fc = load_forecast()
    df_pred = load_predictions()
    df_anom = load_anomalies()
    df_peak = load_peak_periods()
    df_rec = load_recommendations()

    selected_bldg = filters.get('selected_building')
    start_date = filters.get('start_date')
    end_date = filters.get('end_date')

    # Executive Briefing Generator Card
    st.markdown(
        """
        <div class="analytics-card" style="background: linear-gradient(to right, #FFFFFF, #F8FAFC); border-left: 5px solid #2563EB;">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">📄 Automated Executive Briefing Generator</h4>
                    <div class="analytics-card-desc">Compile complete portfolio analytics into a printable executive report ready for management review</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    rcol1, rcol2 = st.columns([8, 4])
    with rcol1:
        st.markdown(
            """
            The executive report aggregates executive KPIs, model benchmarking tables, high-consumption building rankings, 
            and targeted facility recommendations into a standalone, styled document designed for browser printing and PDF archiving.
            """
        )
    with rcol2:
        html_report = generate_html_executive_report(
            metadata=df_meta,
            df_metrics=df_metrics,
            df_summary=df_summary,
            df_forecast=df_fc,
            df_anomalies=df_anom,
            df_recommendations=df_rec
        )
        st.download_button(
            label="📄 Download HTML Executive Report (Printable)",
            data=html_report,
            file_name=f"energy_executive_report_{pd.Timestamp.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            type="primary"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Multi-Dataset Explorer Tabs
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🗄️ Multi-Dataset Explorer & Export Repository</h4>
                    <div class="analytics-card-desc">Search, filter, and extract verified data tables</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    tabs = st.tabs([
        "🏢 Building Summary",
        "📈 24h Forecast Table",
        "🚨 Anomaly Events",
        "⚡ Peak Demand Records",
        "🤖 Model Metrics & Benchmarks",
        "💡 Facility Recommendations",
        "🎯 Feature Importance"
    ])

    with tabs[0]:
        st.markdown("##### 20 Commercial Buildings Summary Table")
        st.dataframe(df_summary, use_container_width=True, height=280)
        csv_sum = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Building Summary CSV", csv_sum, "building_summary.csv", "text/csv")

    with tabs[1]:
        st.markdown("##### 24-Hour Forward Horizon Energy Forecast")
        fc_display = df_fc if not selected_bldg else df_fc[df_fc['building_id'] == selected_bldg]
        st.dataframe(fc_display, use_container_width=True, height=280)
        csv_fc = fc_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Forecast CSV", csv_fc, "forecast_24h.csv", "text/csv")

    with tabs[2]:
        st.markdown("##### Detected Energy Consumption Anomalies")
        anom_display = df_anom if not selected_bldg else df_anom[df_anom['building_id'] == selected_bldg]
        anom_display = anom_display[anom_display['anomaly_flag'] == True]
        st.dataframe(anom_display.head(500), use_container_width=True, height=280)
        csv_anom = anom_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Anomalies CSV", csv_anom, "anomalies_flagged.csv", "text/csv")

    with tabs[3]:
        st.markdown("##### Peak Demand (p95 Threshold Excursion) Events")
        peak_display = df_peak if not selected_bldg else df_peak[df_peak['building_id'] == selected_bldg]
        st.dataframe(peak_display.head(500), use_container_width=True, height=280)
        csv_peak = peak_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Peak Periods CSV", csv_peak, "peak_periods.csv", "text/csv")

    with tabs[4]:
        st.markdown("##### Machine Learning Model Benchmarks")
        st.dataframe(df_metrics, use_container_width=True, height=220)
        csv_met = df_metrics.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Model Metrics CSV", csv_met, "model_metrics.csv", "text/csv")

    with tabs[5]:
        st.markdown("##### Operational Facility Recommendations")
        rec_display = df_rec if not selected_bldg else df_rec[df_rec['building_id'] == selected_bldg]
        st.dataframe(rec_display, use_container_width=True, height=280)
        csv_rec = rec_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Recommendations CSV", csv_rec, "recommendations.csv", "text/csv")

    with tabs[6]:
        st.markdown("##### Permutation Feature Importance Scores")
        st.dataframe(df_fi, use_container_width=True, height=280)
        csv_fi = df_fi.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Feature Importance CSV", csv_fi, "feature_importance.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)
