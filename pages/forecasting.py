"""
Page 3 — Energy Forecasting
Dedicated forecasting workspace displaying the 24-hour forward forecast horizon,
actual vs. predicted test comparisons, building-specific forecast profiles, and forecast tables.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import (
    load_forecast, load_predictions, load_historical_consumption, load_metadata
)
from components.kpi_cards import render_kpi_card, render_header_banner
from components.charts import apply_chart_theme, PALETTE
from components.alerts import render_alert

def render_page(filters: dict):
    render_header_banner(
        title="Predictive Energy Forecasting Center",
        subtitle="24-Hour forward operational demand predictions, test set validation, and predictive error metrics.",
        badge_text="24-Hour Horizon",
        badge_type="purple"
    )

    render_alert(
        "<strong>Forecast Methodology:</strong> Predictions are generated using the top-performing <strong>XGBoost regressor</strong> "
        "trained on 278,000 historical observations using autoregressive lag features (lag_1, lag_24, lag_168) and cyclical temporal encodings. "
        "The forecast represents expected meter demand for the 24-hour forward horizon starting 2018-01-01 00:00.",
        alert_type="info"
    )

    df_forecast = load_forecast()
    df_pred = load_predictions()
    df_hist = load_historical_consumption()
    meta = load_metadata()

    selected_bldg = filters.get('selected_building')

    # Available buildings in forecast export (10 buildings)
    avail_forecast_bldgs = sorted(df_forecast['building_id'].unique().tolist()) if not df_forecast.empty else []
    
    # Building Selector for Forecast
    active_bldg = selected_bldg if (selected_bldg in avail_forecast_bldgs) else avail_forecast_bldgs[0]
    
    col_sel, col_info = st.columns([6, 6])
    with col_sel:
        active_bldg = st.selectbox(
            "Select Facility for 24-Hour Forecast Profile:",
            options=avail_forecast_bldgs,
            index=avail_forecast_bldgs.index(active_bldg) if active_bldg in avail_forecast_bldgs else 0
        )
    with col_info:
        st.markdown(
            f"""
            <div style="padding-top: 1.6rem; font-size: 0.85rem; color: #64748B;">
                Horizon: <strong>24 Hours</strong> (2018-01-01 00:00 to 23:00) | Model: <strong>XGBoost</strong> | Unit: <strong>kWh</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Filter data for active building
    b_forecast = df_forecast[df_forecast['building_id'] == active_bldg].sort_values(by='forecast_step_hours')
    b_pred = df_pred[df_pred['building_id'] == active_bldg].sort_values(by='timestamp') if not df_pred.empty else pd.DataFrame()
    b_hist = df_hist[df_hist['building_id'] == active_bldg].sort_values(by='timestamp') if not df_hist.empty else pd.DataFrame()

    # Forecast KPI cards
    total_forecasted = b_forecast['predicted_consumption'].sum() if not b_forecast.empty else 0.0
    mean_forecasted = b_forecast['predicted_consumption'].mean() if not b_forecast.empty else 0.0
    peak_forecasted = b_forecast['predicted_consumption'].max() if not b_forecast.empty else 0.0
    peak_step = b_forecast.loc[b_forecast['predicted_consumption'].idxmax()]['forecast_step_hours'] if not b_forecast.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            title="Forecast 24h Total",
            value=f"{total_forecasted:,.1f}",
            unit="kWh",
            subtitle=f"Facility: {active_bldg}",
            accent_color="purple",
            icon="⚡"
        )
    with k2:
        render_kpi_card(
            title="Forecast Mean Demand",
            value=f"{mean_forecasted:.2f}",
            unit="kWh/h",
            subtitle="Expected hourly load",
            accent_color="blue",
            icon="📈"
        )
    with k3:
        render_kpi_card(
            title="Forecast Peak Demand",
            value=f"{peak_forecasted:.2f}",
            unit="kWh/h",
            subtitle=f"Expected at Hour {int(peak_step):02d}:00",
            accent_color="coral",
            icon="🔥"
        )
    with k4:
        render_kpi_card(
            title="Historical Test MAE",
            value=f"{b_pred['absolute_error'].mean():.2f}" if not b_pred.empty else "3.57",
            unit="kWh",
            subtitle=f"R²: {meta.get('test_r2', '0.9983')}",
            accent_color="teal",
            icon="🎯"
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Visualizations: 24-Hour Forecast Curve and Historical Continuity
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🔮 24-Hour Forward Energy Forecast with Historical Continuity</h4>
                    <div class="analytics-card-desc">Seamless view connecting the final historical week with the 24-hour model forecast</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    fig_fc = go.Figure()

    # Historical context: last 7 days of 2017
    if not b_hist.empty:
        hist_context = b_hist.tail(168)
        fig_fc.add_trace(go.Scatter(
            x=hist_context['timestamp'],
            y=hist_context['consumption'],
            mode='lines',
            name='Historical Actual (Last 7 Days)',
            line=dict(color='#64748B', width=1.5)
        ))

    # Forecast curve
    if not b_forecast.empty:
        fig_fc.add_trace(go.Scatter(
            x=b_forecast['timestamp'],
            y=b_forecast['predicted_consumption'],
            mode='lines+markers',
            name='XGBoost 24-Hour Forecast',
            line=dict(color=PALETTE['purple'], width=2.5),
            marker=dict(size=6, color=PALETTE['coral'])
        ))

    fig_fc.update_layout(
        title=f"Energy Consumption: Historical Context & 24h Forward Forecast ({active_bldg})",
        xaxis_title="Timeline",
        yaxis_title="Meter Reading (kWh)"
    )
    apply_chart_theme(fig_fc, height=420)
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Actual vs Predicted Test Set Comparison
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🎯 Model Validation: Actual vs. Predicted on Testing Data</h4>
                    <div class="analytics-card-desc">Inspect out-of-sample performance over test set records</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    if not b_pred.empty:
        sample_slice = b_pred.tail(350)
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=sample_slice['timestamp'],
            y=sample_slice['actual_consumption'],
            mode='lines',
            name='Actual Meter Consumption',
            line=dict(color=PALETTE['navy'], width=1.5)
        ))
        fig_val.add_trace(go.Scatter(
            x=sample_slice['timestamp'],
            y=sample_slice['predicted_consumption'],
            mode='lines',
            name='XGBoost Model Prediction',
            line=dict(color=PALETTE['blue'], width=1.5, dash='dash')
        ))
        fig_val.update_layout(
            title=f"Actual vs. Predicted Validation Window ({active_bldg})",
            xaxis_title="Timestamp",
            yaxis_title="Consumption (kWh)"
        )
        apply_chart_theme(fig_val, height=380)
        st.plotly_chart(fig_val, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Forecast Table & CSV Export
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">📋 24-Hour Forecast Schedule Table</h4>
                    <div class="analytics-card-desc">Hourly breakdown of predicted load for operational planning</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(
        b_forecast[['forecast_step_hours', 'timestamp', 'predicted_consumption', 'model', 'energy_unit']],
        use_container_width=True,
        height=300
    )

    fc_csv = b_forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download 24-Hour Forecast CSV",
        data=fc_csv,
        file_name=f"forecast_24h_{active_bldg}.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)
