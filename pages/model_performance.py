"""
Page 6 — Machine Learning Performance
Evaluation and benchmarking centre comparing the 5 ML models,
permutation feature importances, residual error distributions, and metric educational guides.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    load_model_metrics, load_feature_importance, load_predictions, load_metadata
)
from components.kpi_cards import render_kpi_card, render_header_banner
from components.charts import (
    apply_chart_theme, plot_actual_vs_predicted, plot_residual_analysis,
    plot_feature_importance, PALETTE
)
from components.alerts import render_alert

def render_page(filters: dict):
    render_header_banner(
        title="Machine Learning Model Evaluation & Benchmark Centre",
        subtitle="Empirical performance comparison across 5 algorithmic architectures, residual diagnostics, and feature attributions.",
        badge_text="Colab Model Benchmark",
        badge_type="purple"
    )

    df_metrics = load_model_metrics()
    df_fi = load_feature_importance()
    df_pred = load_predictions()
    meta = load_metadata()

    if df_metrics.empty:
        st.error("Model metrics dataset is unavailable.")
        return

    best_model = df_metrics.iloc[0]

    # Top KPI Metrics for Best Model
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Champion Model",
            value=f"{best_model['model']}",
            subtitle="Lowest MAE & Highest R²",
            accent_color="purple",
            icon="🏆"
        )
    with col2:
        render_kpi_card(
            title="Mean Absolute Error (MAE)",
            value=f"{best_model['MAE']:.4f}",
            unit="kWh",
            subtitle="Average absolute prediction divergence",
            accent_color="blue",
            icon="🎯"
        )
    with col3:
        render_kpi_card(
            title="Root Mean Squared Error (RMSE)",
            value=f"{best_model['RMSE']:.4f}",
            unit="kWh",
            subtitle="Penalizes large error outliers",
            accent_color="teal",
            icon="📐"
        )
    with col4:
        render_kpi_card(
            title="Coefficient of Determination (R²)",
            value=f"{best_model['R2']:.6f}",
            subtitle="Explains 99.83% of variance",
            accent_color="coral",
            icon="✨"
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Model Comparison Benchmark Table & Bar Chart
    mcol1, mcol2 = st.columns([6, 6])
    with mcol1:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📊 Multi-Model Performance Benchmark</h4>
                        <div class="analytics-card-desc">Ranked comparison on 69,520 out-of-sample test observations</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(
            df_metrics.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.6f}'}),
            use_container_width=True,
            height=240
        )

        with st.expander("📖 Educational Guide: Evaluation Metrics Explained", expanded=False):
            st.markdown(
                """
                - **MAE (Mean Absolute Error):** Measures the average magnitude of prediction errors in the original measurement units ($kWh$). Unlike RMSE, all individual errors are weighted equally.
                - **RMSE (Root Mean Squared Error):** Takes the square root of the average of squared differences. It assigns higher penalties to large errors, making it particularly useful when large forecasting spikes are costly.
                - **R² Score (Coefficient of Determination):** Proportion of the variance in electricity consumption that is predictable from the input features. $R^2 = 1.0$ represents perfect predictive accuracy.
                """
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with mcol2:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📉 Comparative Error Bar Chart</h4>
                        <div class="analytics-card-desc">MAE comparison highlighting superiority of gradient boosted trees</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=df_metrics['model'],
            y=df_metrics['MAE'],
            name='MAE (kWh)',
            marker=dict(color=PALETTE['blue'])
        ))
        fig_comp.add_trace(go.Bar(
            x=df_metrics['model'],
            y=df_metrics['RMSE'],
            name='RMSE (kWh)',
            marker=dict(color=PALETTE['purple'])
        ))
        fig_comp.update_layout(
            barmode='group',
            xaxis_title="",
            yaxis_title="Error Magnitude (kWh)"
        )
        apply_chart_theme(fig_comp, height=330)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance Plot
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🔍 Model Interpretability: Permutation Feature Importance</h4>
                    <div class="analytics-card-desc">Quantifies the contribution of autoregressive lag features and calendar encodings</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    if not df_fi.empty:
        fig_fi = plot_feature_importance(df_fi, top_n=15)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown(
            f"""
            <div style="background: #F8FAFC; border-left: 3px solid #2563EB; padding: 0.8rem 1rem; border-radius: 6px; font-size: 0.8rem; color: #475569;">
                <strong>Key Finding:</strong> <code>lag_1</code> (previous hour demand) dominates with an importance score of <strong>{df_fi.iloc[0]['importance_mean']:.1f}</strong>, 
                followed by <code>rolling_mean_24</code> (<strong>{df_fi.iloc[1]['importance_mean']:.1f}</strong>) and <code>lag_24</code> (<strong>{df_fi.iloc[2]['importance_mean']:.1f}</strong>). 
                This demonstrates that immediate thermal persistence and diurnal periodicity are the primary drivers of building electrical load.
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Actual vs Predicted & Residual Error Analysis
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🎯 Out-of-Sample Predictions vs. Actuals</h4>
                    <div class="analytics-card-desc">Validation performance across selected facility test sequence</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    selected_bldg = filters.get('selected_building')
    bldg_pred = df_pred.copy()
    if selected_bldg:
        bldg_pred = bldg_pred[bldg_pred['building_id'] == selected_bldg]

    if not bldg_pred.empty:
        fig_avp = plot_actual_vs_predicted(
            bldg_pred,
            sample_size=350,
            title=f"Actual vs. Predicted Consumption Window {'(' + selected_bldg + ')' if selected_bldg else '(Portfolio Sample)'}"
        )
        st.plotly_chart(fig_avp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Residuals Analysis
    rcol1, rcol2 = st.columns(2)
    fig_scat, fig_hist = plot_residual_analysis(bldg_pred)
    with rcol1:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📉 Residuals vs. Fitted Values</h4>
                        <div class="analytics-card-desc">Tests for homoscedasticity and zero-mean error assumption</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_scat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with rcol2:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📊 Residual Error Distribution</h4>
                        <div class="analytics-card-desc">Inspects symmetry and heavy tails in prediction error</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
