"""
Page 4 — Anomaly Detection Centre
Operational surveillance module for detecting, triaging, and investigating unusual energy consumption events.
Includes multi-criteria severity categorization and interactive investigation workflow state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_anomalies, load_building_summary
from components.kpi_cards import render_kpi_card, render_header_banner
from components.charts import apply_chart_theme, plot_anomaly_timeline, PALETTE
from components.alerts import render_alert, render_empty_state

def init_anomaly_session_state(df_anom: pd.DataFrame):
    """Initialize investigation workflow status in session state if not set."""
    if 'anomaly_review_status' not in st.session_state:
        # Default all to Unreviewed
        st.session_state.anomaly_review_status = {}

def get_anomaly_severity(residual: float, mae: float = 3.57) -> str:
    """Classify anomaly severity based on absolute residual multiplier."""
    abs_res = abs(residual)
    if abs_res >= 3.0 * mae:
        return "High"
    elif abs_res >= 1.5 * mae:
        return "Medium"
    else:
        return "Low"

def render_page(filters: dict):
    render_header_banner(
        title="Energy Anomaly Detection & Investigation Centre",
        subtitle="Surveillance of statistical consumption deviations using dual-criteria ML residual and Isolation Forest flags.",
        badge_text="Operational Triage",
        badge_type="coral"
    )

    render_alert(
        "<strong>Operational Clarification:</strong> An 'anomaly' in this platform denotes a significant statistical divergence "
        "from the model-predicted baseline. It serves as an <strong>investigative alert</strong> for facility operators "
        "(e.g., unintended schedule overrides, sensor recalibration, or unusual load events), "
        "and does <strong>not</strong> constitute confirmed mechanical failure or unauthorized use.",
        alert_type="warning"
    )

    df_anom = load_anomalies()
    df_summary = load_building_summary()

    if df_anom.empty:
        render_empty_state("No Anomaly Data", "The anomalies dataset is currently unavailable.")
        return

    init_anomaly_session_state(df_anom)

    selected_bldg = filters.get('selected_building')
    start_date = filters.get('start_date')
    end_date = filters.get('end_date')

    # Apply global filters
    df_filtered = df_anom.copy()
    if selected_bldg:
        df_filtered = df_filtered[df_filtered['building_id'] == selected_bldg]
    if start_date and end_date and not df_filtered.empty:
        df_filtered = df_filtered[
            (df_filtered['timestamp'].dt.date >= start_date) &
            (df_filtered['timestamp'].dt.date <= end_date)
        ]

    # Calculate severity and add investigation status
    df_filtered['severity'] = df_filtered['residual'].apply(get_anomaly_severity)
    
    # Map review status from session state
    df_filtered['event_id'] = df_filtered['building_id'] + "_" + df_filtered['timestamp'].astype(str)
    df_filtered['investigation_status'] = df_filtered['event_id'].apply(
        lambda eid: st.session_state.anomaly_review_status.get(eid, "Unreviewed")
    )

    # Anomaly subsets
    anom_only = df_filtered[df_filtered['anomaly_flag'] == True].copy()

    total_records = len(df_filtered)
    total_anomalies = len(anom_only)
    high_sev = len(anom_only[anom_only['severity'] == 'High'])
    med_sev = len(anom_only[anom_only['severity'] == 'Medium'])
    low_sev = len(anom_only[anom_only['severity'] == 'Low'])

    # KPI Summary Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Total Flagged Events",
            value=f"{total_anomalies:,}",
            subtitle=f"{total_anomalies / max(1, total_records) * 100:.2f}% of monitored interval",
            accent_color="coral",
            icon="🚨"
        )
    with col2:
        render_kpi_card(
            title="High Severity",
            value=f"{high_sev:,}",
            subtitle="Residual ≥ 3× MAE threshold",
            accent_color="coral",
            icon="🔴"
        )
    with col3:
        render_kpi_card(
            title="Medium Severity",
            value=f"{med_sev:,}",
            subtitle="Residual 1.5× - 3× MAE",
            accent_color="amber",
            icon="🟡"
        )
    with col4:
        render_kpi_card(
            title="Dual-Detection Agreement",
            value=f"{len(anom_only[anom_only['residual_anomaly'] & anom_only['isolation_anomaly']]):,}",
            subtitle="Flagged by both Residual & Isolation Forest",
            accent_color="purple",
            icon="🛡️"
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Anomaly Timeline Chart
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">📈 Anomaly Detection Timeline</h4>
                    <div class="analytics-card-desc">Red markers highlight consumption spikes and drops exceeding statistical bounds</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    if not df_filtered.empty:
        # Sample for smooth browser rendering if very large
        chart_slice = df_filtered.tail(3500)
        fig_timeline = plot_anomaly_timeline(chart_slice, height=380)
        st.plotly_chart(fig_timeline, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Anomaly Breakdown: Buildings and Severity
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">🏢 Anomalies by Building</h4>
                        <div class="analytics-card-desc">Ranked frequency of unusual energy episodes</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        if not anom_only.empty:
            b_counts = anom_only.groupby('building_id').size().reset_index(name='count').sort_values(by='count', ascending=True)
            fig_b = go.Figure(go.Bar(
                x=b_counts['count'],
                y=b_counts['building_id'],
                orientation='h',
                marker=dict(color=PALETTE['coral'])
            ))
            fig_b.update_layout(xaxis_title="Anomaly Count", yaxis_title="")
            apply_chart_theme(fig_b, height=340)
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("No anomalies detected for the active filter range.")
        st.markdown("</div>", unsafe_allow_html=True)

    with bcol2:
        st.markdown(
            """
            <div class="analytics-card">
                <div class="analytics-card-header">
                    <div>
                        <h4 class="analytics-card-title">📊 Anomaly Residual Magnitude Distribution</h4>
                        <div class="analytics-card-desc">Deviation magnitude between observed meter readings and expected baseline</div>
                    </div>
                </div>
            """,
            unsafe_allow_html=True
        )
        if not anom_only.empty:
            fig_res = px.histogram(
                anom_only,
                x='residual',
                color='severity',
                color_discrete_map={'High': PALETTE['coral'], 'Medium': PALETTE['amber'], 'Low': PALETTE['teal']},
                nbins=40,
                title="Residual Deviation Histogram"
            )
            fig_res.update_xaxes(title="Residual (Actual - Predicted) [kWh]")
            apply_chart_theme(fig_res, height=340)
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("No anomalies to display.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Investigation Workflow & Event Log Table
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🔍 Operational Investigation Log & Triage Workflow</h4>
                    <div class="analytics-card-desc">Review individual anomalies, assign investigation status, and export audit trails</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # Sub-filter by severity and status
    scol1, scol2, scol3 = st.columns([4, 4, 4])
    with scol1:
        sev_filter = st.multiselect(
            "Filter by Severity:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with scol2:
        status_filter = st.multiselect(
            "Filter by Status:",
            options=["Unreviewed", "Under investigation", "Reviewed", "Resolved"],
            default=["Unreviewed", "Under investigation", "Reviewed", "Resolved"]
        )
    with scol3:
        sort_by = st.selectbox(
            "Sort Log By:",
            options=["Timestamp (Newest First)", "Residual Magnitude (Highest First)"],
            index=0
        )

    # Filter anomaly table
    table_df = anom_only[
        (anom_only['severity'].isin(sev_filter)) &
        (anom_only['investigation_status'].isin(status_filter))
    ].copy()

    if sort_by == "Timestamp (Newest First)":
        table_df = table_df.sort_values(by='timestamp', ascending=False)
    else:
        table_df['abs_residual'] = table_df['residual'].abs()
        table_df = table_df.sort_values(by='abs_residual', ascending=False)

    # Status updater control
    st.markdown("##### Quick Triage Action for Top Events:")
    with st.expander("Update Event Status / Facility Review", expanded=False):
        if not table_df.empty:
            event_options = table_df['event_id'].head(20).tolist()
            target_event = st.selectbox("Select Anomaly Event ID to Update:", options=event_options)
            col_stat, col_btn = st.columns([6, 6])
            with col_stat:
                new_status = st.selectbox(
                    "Set New Investigation Status:",
                    options=["Unreviewed", "Under investigation", "Reviewed", "Resolved"]
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("Apply Status Update", type="primary"):
                    st.session_state.anomaly_review_status[target_event] = new_status
                    st.success(f"Updated {target_event} to '{new_status}'!")
                    st.rerun()

    # Display clean table
    display_cols = [
        'timestamp', 'building_id', 'actual_consumption', 'predicted_consumption',
        'residual', 'severity', 'residual_anomaly', 'isolation_anomaly', 'investigation_status'
    ]
    st.dataframe(
        table_df[display_cols].head(300),
        use_container_width=True,
        height=320
    )

    # Export anomalies CSV
    anom_csv = table_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Anomalies CSV",
        data=anom_csv,
        file_name=f"anomalies_{selected_bldg or 'portfolio'}_{start_date}_{end_date}.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)
