"""
Page 5 — Peak Demand Management
Operational module for identifying extreme demand periods, recurring peak hours,
facility load profiles, and load-shifting / peak-shaving opportunities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_peak_periods, load_building_summary
from components.kpi_cards import render_kpi_card, render_header_banner
from components.charts import apply_chart_theme, PALETTE
from components.alerts import render_alert, render_empty_state

def render_page(filters: dict):
    render_header_banner(
        title="Peak Demand Surveillance & Load Management",
        subtitle="Identification of high-draw intervals, threshold excursions (p95), peak hours, and load-shifting opportunities.",
        badge_text="Demand Response",
        badge_type="amber"
    )

    render_alert(
        "<strong>Demand Management Insight:</strong> Commercial facility electricity bills often incur heavy monthly "
        "peak-demand tariffs based on the highest 15-to-60 minute kilowatt surge. "
        "Identifying when peaks recur enables facility managers to evaluate <strong>operational schedule staggering</strong>, "
        "pre-cooling strategies, and selective equipment staging.",
        alert_type="info"
    )

    df_peak = load_peak_periods()
    df_summary = load_building_summary()

    if df_peak.empty:
        render_empty_state("No Peak Period Data", "The peak periods dataset is unavailable.")
        return

    selected_bldg = filters.get('selected_building')
    start_date = filters.get('start_date')
    end_date = filters.get('end_date')

    # Apply global filters
    df_filtered = df_peak.copy()
    if selected_bldg:
        df_filtered = df_filtered[df_filtered['building_id'] == selected_bldg]
    if start_date and end_date and not df_filtered.empty:
        df_filtered = df_filtered[
            (df_filtered['timestamp'].dt.date >= start_date) &
            (df_filtered['timestamp'].dt.date <= end_date)
        ]

    if df_filtered.empty:
        render_empty_state("No Data in Selected Interval", "Please widen your date range or choose another building.")
        return

    # Add temporal columns
    df_filtered['hour'] = df_filtered['timestamp'].dt.hour
    df_filtered['day_of_week'] = df_filtered['timestamp'].dt.dayofweek
    df_filtered['is_weekend'] = (df_filtered['day_of_week'] >= 5).astype(int)

    total_peak_events = len(df_filtered)
    max_demand_recorded = df_filtered['consumption'].max()
    mean_peak_consumption = df_filtered['consumption'].mean()
    mean_threshold = df_filtered['p95_threshold'].mean()

    # Most frequent peak hour
    hour_counts = df_filtered['hour'].value_counts()
    peak_hour = hour_counts.index[0] if not hour_counts.empty else 14

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Peak Demand Events",
            value=f"{total_peak_events:,}",
            subtitle=f"{selected_bldg or 'Portfolio-wide'} interval count",
            accent_color="amber",
            icon="⚡"
        )
    with col2:
        render_kpi_card(
            title="Max Recorded Demand",
            value=f"{max_demand_recorded:.1f}",
            unit="kWh/h",
            subtitle="Absolute highest load surge",
            accent_color="coral",
            icon="🔥"
        )
    with col3:
        render_kpi_card(
            title="Predominant Peak Hour",
            value=f"{peak_hour:02d}:00",
            subtitle=f"Hour {peak_hour:02d}:00 to {(peak_hour+1)%24:02d}:00",
            accent_color="purple",
            icon="⏱️"
        )
    with col4:
        render_kpi_card(
            title="Mean p95 Threshold",
            value=f"{mean_threshold:.1f}",
            unit="kWh/h",
            subtitle="95th percentile baseline cut-off",
            accent_color="blue",
            icon="📊"
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Visualizations: Peak Hours Distribution & Daily Distribution
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">⏱️ Peak Demand Occurrence by Hour of Day</h4>
                        <div class="analytics-card-desc">Reveals concentrated operating periods triggering peak threshold excursions</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        hourly_peaks = df_filtered.groupby('hour').size().reset_index(name='peak_count')
        fig_h = px.bar(
            hourly_peaks,
            x='hour',
            y='peak_count',
            color='peak_count',
            color_continuous_scale='Oranges',
            title="Peak Demand Episodes by Hour"
        )
        fig_h.update_xaxes(title="Hour of Day (0-23)", dtick=2)
        fig_h.update_yaxes(title="Count of Peak Events")
        fig_h.update_layout(coloraxis_showscale=False)
        apply_chart_theme(fig_h, height=360)
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with pcol2:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📅 Peak Demand by Day of Week</h4>
                        <div class="analytics-card-desc">Distribution across workdays versus weekends</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_peaks = df_filtered.groupby('day_of_week').size().reset_index(name='peak_count')
        dow_peaks['day_name'] = dow_peaks['day_of_week'].map(lambda x: days[x] if x < len(days) else str(x))
        
        fig_d = px.bar(
            dow_peaks,
            x='day_name',
            y='peak_count',
            color='day_name',
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Peak Frequency by Day"
        )
        fig_d.update_xaxes(title="")
        fig_d.update_yaxes(title="Count of Peak Events")
        fig_d.update_layout(showlegend=False)
        apply_chart_theme(fig_d, height=360)
        st.plotly_chart(fig_d, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Building Comparison Bar Chart
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🏢 Peak Events Distribution Across Facilities</h4>
                    <div class="analytics-card-desc">Identifies buildings generating the highest frequency of high-demand periods</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    b_peaks = df_filtered.groupby('building_id').size().reset_index(name='count').sort_values(by='count', ascending=True)
    fig_bp = go.Figure(go.Bar(
        x=b_peaks['count'],
        y=b_peaks['building_id'],
        orientation='h',
        marker=dict(color=PALETTE['amber'])
    ))
    fig_bp.update_layout(xaxis_title="Peak Period Count", yaxis_title="")
    apply_chart_theme(fig_bp, height=380)
    st.plotly_chart(fig_bp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Actionable Facility Management Opportunities
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">💡 Operational Load Management & Peak Mitigation Strategies</h4>
                    <div class="analytics-card-desc">Practical engineering options for facility operators to investigate</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                <div style="background: #FFFBEB; border-left: 3px solid #D97706; padding: 1rem; border-radius: 8px;">
                    <div style="font-weight: 700; color: #92400E; margin-bottom: 0.25rem;">Staggered Chiller & Air Handler Startup</div>
                    <div style="font-size: 0.8rem; color: #78350F;">
                        Morning ramp-up between 07:00 and 09:00 frequently exhibits simultaneous start spikes. 
                        Staggering HVAC plant staging in 15-minute increments avoids coincidence peaks.
                    </div>
                </div>
                <div style="background: #F0FDF4; border-left: 3px solid #16A34A; padding: 1rem; border-radius: 8px;">
                    <div style="font-weight: 700; color: #166534; margin-bottom: 0.25rem;">Thermal Pre-Cooling Opportunities</div>
                    <div style="font-size: 0.8rem; color: #14532D;">
                        During high-temperature periods, pre-cooling building thermal mass between 05:00 and 07:00 
                        allows setting back chillers during afternoon peak tariff windows (13:00 - 16:00).
                    </div>
                </div>
                <div style="background: #EFF6FF; border-left: 3px solid #2563EB; padding: 1rem; border-radius: 8px;">
                    <div style="font-weight: 700; color: #1E40AF; margin-bottom: 0.25rem;">Flexible Load Dispatch</div>
                    <div style="font-size: 0.8rem; color: #1E3A8A;">
                        Non-critical discretionary loads (water heating, secondary booster pumps, EV chargers) 
                        can be interlocked with sub-meter threshold triggers to shed loads automatically.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Peak Periods Data Table & Download
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">📋 Peak Demand Event Records</h4>
                    <div class="analytics-card-desc">Inspect exact peak timestamps and threshold values</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(
        df_filtered[['timestamp', 'building_id', 'consumption', 'p95_threshold', 'peak_demand_flag']].head(300),
        use_container_width=True,
        height=280
    )

    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Peak Demand Events CSV",
        data=csv_data,
        file_name=f"peak_demand_{selected_bldg or 'portfolio'}_{start_date}_{end_date}.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)
