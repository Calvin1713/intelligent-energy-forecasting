"""
Page 7 — Energy Optimization Recommendations
Actionable facility recommendations categorized by operational opportunity,
backed by empirical evidence and paired with a transparent tariff scenario modeling calculator.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_recommendations, load_building_summary
from utils.recommendations import enrich_recommendations, calculate_tariff_scenarios
from components.kpi_cards import render_kpi_card, render_header_banner
from components.alerts import render_alert

def render_page(filters: dict):
    render_header_banner(
        title="Energy Efficiency & Facility Optimization Recommendations",
        subtitle="Empirical operational guidance derived from consumption patterns, baseload audits, and weekend load tracking.",
        badge_text="Actionable Insights",
        badge_type="teal"
    )

    render_alert(
        "<strong>Integrity & Tariff Disclaimer:</strong> All recommendations are generated directly from patterns identified in "
        "historical meter data. Cost avoidance calculations below are <strong>hypothetical scenarios</strong> based on user-entered "
        "tariff rates. Savings are not guaranteed and require on-site engineering verification before capital investment.",
        alert_type="info"
    )

    df_rec = load_recommendations()
    df_summary = load_building_summary()

    if df_rec.empty:
        st.info("No recommendations found.")
        return

    enriched_rec = enrich_recommendations(df_rec, df_summary)

    selected_bldg = filters.get('selected_building')
    if selected_bldg:
        filtered_rec = enriched_rec[enriched_rec['building_id'] == selected_bldg]
    else:
        filtered_rec = enriched_rec

    # Top KPI Metrics
    total_recs = len(filtered_rec)
    high_prio = len(filtered_rec[filtered_rec['priority'] == 'High'])
    overnight_cases = len(filtered_rec[filtered_rec['pattern'] == 'Elevated overnight baseline'])
    weekend_cases = len(filtered_rec[filtered_rec['pattern'] == 'Weekend consumption present'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            title="Total Opportunities",
            value=f"{total_recs}",
            subtitle=f"{selected_bldg or 'Portfolio-wide'} audit findings",
            accent_color="teal",
            icon="💡"
        )
    with col2:
        render_kpi_card(
            title="High Priority Actions",
            value=f"{high_prio}",
            subtitle="Immediate baseload audit required",
            accent_color="coral",
            icon="🔴"
        )
    with col3:
        render_kpi_card(
            title="Overnight Baseload Alerts",
            value=f"{overnight_cases}",
            subtitle="Off-hours draw > 40% daytime",
            accent_color="purple",
            icon="🌙"
        )
    with col4:
        render_kpi_card(
            title="Weekend Load Anomalies",
            value=f"{weekend_cases}",
            subtitle="Significant unoccupied demand",
            accent_color="amber",
            icon="📅"
        )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Interactive Recommendations Feed
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">📋 Facility Energy Optimization Matrix</h4>
                    <div class="analytics-card-desc">Detailed engineering actions with traceable evidence and rationale</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    # Category and Priority filters
    fcol1, fcol2 = st.columns([6, 6])
    with fcol1:
        cat_choices = ["All Categories"] + sorted(enriched_rec['category'].unique().tolist())
        selected_cat = st.selectbox("Filter by Action Category:", options=cat_choices)
    with fcol2:
        prio_choices = ["All Priorities"] + sorted(enriched_rec['priority'].unique().tolist())
        selected_prio = st.selectbox("Filter by Priority Level:", options=prio_choices)

    display_rec = filtered_rec.copy()
    if selected_cat != "All Categories":
        display_rec = display_rec[display_rec['category'] == selected_cat]
    if selected_prio != "All Priorities":
        display_rec = display_rec[display_rec['priority'] == selected_prio]

    if display_rec.empty:
        st.info("No recommendations match the chosen criteria.")
    else:
        for idx, row in display_rec.iterrows():
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid var(--accent-{row['badge_color']}); border-radius: 10px; padding: 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="font-weight: 700; font-size: 1.05rem; color: #0F172A;">
                            {row['icon']} {row['building_id']} &mdash; <span style="font-size: 0.95rem; color: #334155;">{row['category']}</span>
                        </div>
                        <span class="platform-badge badge-{row['badge_color']}">{row['priority']} Priority</span>
                    </div>
                    <div style="margin-bottom: 0.6rem; font-size: 0.85rem; color: #1E293B; line-height: 1.5;">
                        <strong>Action Recommendation:</strong> {row['recommendation']}
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.78rem; color: #475569;">
                        <div>🔍 <strong>Observed Metric:</strong> {row['observed_value']}</div>
                        <div>📊 <strong>Benchmark:</strong> {row['comparison_value']}</div>
                        <div>🛠️ <strong>Investigation Target:</strong> {row['action_summary']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    rec_csv = display_rec[['building_id', 'pattern', 'observed_value', 'comparison_value', 'recommendation', 'category', 'priority']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Recommendations CSV",
        data=rec_csv,
        file_name=f"energy_recommendations_{selected_bldg or 'portfolio'}.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Tariff & Cost Scenario Calculator
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🧮 Interactive Energy Tariff & Efficiency Scenario Calculator</h4>
                    <div class="analytics-card-desc">Simulate potential cost avoidance by adjusting contractual electricity rate assumptions</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 1rem;">
            Enter your local utility tariff structure below to project annualized electricity expenditure and explore 
            potential cost reductions from implementing the operational recommendations.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Calculate baseline consumption from summary
    if selected_bldg and not df_summary.empty:
        b_sum = df_summary[df_summary['building_id'] == selected_bldg]
        calc_mean = float(b_sum['mean_consumption'].iloc[0]) if not b_sum.empty else 50.0
        calc_peak = float(b_sum['maximum_consumption'].iloc[0]) if not b_sum.empty else 120.0
    else:
        calc_mean = float(df_summary['mean_consumption'].mean()) if not df_summary.empty else 100.0
        calc_peak = float(df_summary['maximum_consumption'].max()) if not df_summary.empty else 500.0

    ccol1, ccol2, ccol3 = st.columns(3)
    with ccol1:
        rate_kwh = st.number_input(
            "Energy Rate ($ / kWh):",
            min_value=0.01,
            max_value=1.50,
            value=0.14,
            step=0.01,
            help="Blended volumetric rate per kilowatt-hour."
        )
    with ccol2:
        demand_rate = st.number_input(
            "Peak Demand Charge ($ / kW / month):",
            min_value=0.0,
            max_value=50.0,
            value=12.0,
            step=1.0,
            help="Monthly demand charge billed against peak monthly kW demand."
        )
    with ccol3:
        target_red = st.slider(
            "Target Load Reduction (%):",
            min_value=1,
            max_value=25,
            value=5,
            step=1,
            help="Estimated percentage energy reduction achievable through schedule and baseload tuning."
        )

    # Run scenario calculation
    scen = calculate_tariff_scenarios(
        mean_hourly_consumption=calc_mean,
        peak_demand=calc_peak,
        rate_per_kwh=rate_kwh,
        peak_demand_charge_per_kw=demand_rate,
        target_reduction_pct=target_red
    )

    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 600;">Annual Energy Baseline</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-top: 0.2rem;">
                    ${scen['annual_energy_cost_baseline']:,.0f}
                </div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">
                    Based on {scen['annual_kwh_baseline']:,.0f} kWh/year
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with rcol2:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 600;">Peak Demand Charges</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin-top: 0.2rem;">
                    ${scen['annual_demand_cost_baseline']:,.0f}
                </div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.2rem;">
                    Estimated annual capacity fees
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with rcol3:
        st.markdown(
            f"""
            <div style="background: #ECFDF5; border: 1px solid #A7F3D0; padding: 1rem; border-radius: 8px;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #047857; font-weight: 600;">Modeled Cost Avoidance ({target_red}%)</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: #065F46; margin-top: 0.2rem;">
                    ${scen['potential_cost_avoided']:,.0f} / yr
                </div>
                <div style="font-size: 0.75rem; color: #047857; margin-top: 0.2rem;">
                    ~{scen['potential_kwh_avoided']:,.0f} kWh annual reduction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
