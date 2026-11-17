"""
Page 2 — Energy Consumption Analytics
Interactive deep-dive workspace for analyzing multi-scale load profiles, diurnal curves,
heatmaps, distributions, and filtered data exports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    load_historical_consumption, load_building_summary, load_building_metadata
)
from components.kpi_cards import render_kpi_card, render_header_banner
from components.charts import (
    apply_chart_theme, plot_consumption_trend, plot_hourly_profile,
    plot_hour_day_heatmap, PALETTE
)
from components.alerts import render_empty_state

def render_page(filters: dict):
    render_header_banner(
        title="Energy Consumption Analytics Workspace",
        subtitle="Multi-scale temporal analysis, diurnal distributions, load heatmaps, and building benchmarking.",
        badge_text="Operational Diagnostics",
        badge_type="teal"
    )

    df_hist = load_historical_consumption()
    df_summary = load_building_summary()
    df_meta = load_building_metadata()

    selected_bldg = filters.get('selected_building')
    start_date = filters.get('start_date')
    end_date = filters.get('end_date')

    # Apply global filtering
    df_filtered = df_hist.copy()
    if selected_bldg:
        df_filtered = df_filtered[df_filtered['building_id'] == selected_bldg]
    if start_date and end_date and not df_filtered.empty:
        df_filtered = df_filtered[
            (df_filtered['timestamp'].dt.date >= start_date) &
            (df_filtered['timestamp'].dt.date <= end_date)
        ]

    if df_filtered.empty:
        render_empty_state(
            title="No Records Found",
            description="The selected date interval or facility contains no records. Please adjust filters in the sidebar."
        )
        return

    # Add temporal columns
    df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    df_filtered['day_of_week'] = df_filtered['timestamp'].dt.dayofweek
    df_filtered['month'] = df_filtered['timestamp'].dt.month
    df_filtered['is_weekend'] = (df_filtered['day_of_week'] >= 5).astype(int)

    # Compute key stats for selected view
    total_consumed = df_filtered['consumption'].sum()
    mean_demand = df_filtered['consumption'].mean()
    peak_demand = df_filtered['consumption'].max()
    min_demand = df_filtered['consumption'].min()
    record_count = len(df_filtered)

    # Display 4 Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Interval Consumption",
            value=f"{total_consumed / 1e6:.2f}M" if total_consumed > 1e6 else f"{total_consumed:,.0f}",
            unit="kWh",
            subtitle=f"{record_count:,} hourly meter readings",
            accent_color="teal",
            icon="⚡"
        )
    with col2:
        render_kpi_card(
            title="Mean Hourly Demand",
            value=f"{mean_demand:.2f}",
            unit="kWh/h",
            subtitle=f"Facility: {selected_bldg or 'Portfolio Avg'}",
            accent_color="blue",
            icon="📈"
        )
    with col3:
        render_kpi_card(
            title="Peak Observed Demand",
            value=f"{peak_demand:.2f}",
            unit="kWh/h",
            subtitle="Maximum load event",
            accent_color="coral",
            icon="🔥"
        )
    with col4:
        render_kpi_card(
            title="Baseload (Minimum)",
            value=f"{min_demand:.2f}",
            unit="kWh/h",
            subtitle="Minimum standby draw",
            accent_color="purple",
            icon="🌙"
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Granularity Toggle for Time-Series
    tabs = st.tabs(["Hourly & Daily Trends", "Hour-Day Heatmap", "Diurnal Load Profile", "Statistical Distribution"])

    with tabs[0]:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📈 Electricity Consumption Over Time</h4>
                        <div class="analytics-card-desc">Interactive time series with resolution selector</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        agg_choice = st.radio(
            "Temporal Resolution:",
            options=["Daily Average", "Weekly Sum", "Raw Hourly Readings"],
            horizontal=True
        )

        if agg_choice == "Daily Average":
            df_plot = df_filtered.set_index('timestamp').resample('D')['consumption'].mean().reset_index()
            fig_t = plot_consumption_trend(df_plot, title="Daily Average Energy Consumption (kWh)")
        elif agg_choice == "Weekly Sum":
            df_plot = df_filtered.set_index('timestamp').resample('W')['consumption'].sum().reset_index()
            fig_t = plot_consumption_trend(df_plot, title="Weekly Total Energy Consumption (kWh)")
        else:
            # Limit hourly points to prevent browser freeze
            df_plot = df_filtered.tail(3000)
            fig_t = plot_consumption_trend(df_plot, title="Hourly Electricity Consumption (Latest 3,000 Points)")

        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">🔥 Hour of Day vs. Day of Week Intensity Heatmap</h4>
                        <div class="analytics-card-desc">Reveals recurrent operational peaks and after-hours baseline patterns</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        fig_heat = plot_hour_day_heatmap(
            df_filtered,
            title=f"Energy Consumption Heatmap {'(' + selected_bldg + ')' if selected_bldg else '(Portfolio Average)'}"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">⏱️ Diurnal Load Curves (24-Hour Profile)</h4>
                        <div class="analytics-card-desc">Comparison between regular workdays and weekend off-peak operations</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        fig_prof = plot_hourly_profile(
            df_filtered,
            title=f"Average Hourly Load Profile {'(' + selected_bldg + ')' if selected_bldg else '(Portfolio Average)'}"
        )
        st.plotly_chart(fig_prof, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📊 Consumption Frequency Distribution & Boxplots</h4>
                        <div class="analytics-card-desc">Evaluates skewness, median demand, and statistical outlier thresholds</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            fig_hist = px.histogram(
                df_filtered.sample(min(5000, len(df_filtered)), random_state=42),
                x='consumption',
                nbins=40,
                color_discrete_sequence=[PALETTE['blue']],
                title="Consumption Histogram"
            )
            fig_hist.update_xaxes(title="Meter Reading (kWh)")
            apply_chart_theme(fig_hist, height=340)
            st.plotly_chart(fig_hist, use_container_width=True)

        with dcol2:
            fig_box = px.box(
                df_filtered.sample(min(5000, len(df_filtered)), random_state=42),
                x='is_weekend',
                y='consumption',
                color='is_weekend',
                color_discrete_map={0: PALETTE['blue'], 1: PALETTE['amber']},
                labels={'is_weekend': 'Day Type', 'consumption': 'kWh'},
                title="Weekday (0) vs. Weekend (1) Distribution"
            )
            apply_chart_theme(fig_box, height=340)
            st.plotly_chart(fig_box, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Filtered Data Table & CSV Download
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">📋 Filtered Energy Records Table</h4>
                    <div class="analytics-card-desc">Inspect exact meter readings and export for external audit</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(
        df_filtered[['timestamp', 'building_id', 'consumption']].head(500),
        use_container_width=True,
        height=280
    )

    csv_data = df_filtered[['timestamp', 'building_id', 'consumption']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Consumption CSV",
        data=csv_data,
        file_name=f"consumption_{selected_bldg or 'portfolio'}_{start_date}_{end_date}.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)
