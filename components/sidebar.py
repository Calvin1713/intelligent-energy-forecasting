"""
Enterprise Sidebar Navigation and Global Filtering
Provides global building selection, date filters, dataset diagnostics, and brand identity.
"""

import os
import streamlit as st
import pandas as pd
from utils.data_loader import get_building_list, get_date_bounds, load_metadata

def render_sidebar():
    """Render unified sidebar controls and return active filter state."""
    with st.sidebar:
        # Platform Brand Header
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'logo', 'logo.svg')
        if os.path.exists(logo_path):
            with open(logo_path, 'r', encoding='utf-8') as f:
                svg_data = f.read()
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid #334155;">
                    <div style="width: 48px; height: 48px;">{svg_data}</div>
                    <div>
                        <div style="font-weight: 800; font-size: 1.05rem; letter-spacing: -0.01em; color: #FFFFFF; line-height: 1.15;">
                            ENERGY INTEL
                        </div>
                        <div style="font-size: 0.72rem; color: #38BDF8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                            Analytics Platform
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("### ⚡ ENERGY INTEL")

        st.markdown(
            """
            <div style="margin-bottom: 1.2rem;">
                <span class="platform-badge badge-teal" style="font-size: 0.7rem;">● LIVE M.TECH PLATFORM</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### ⚙️ GLOBAL CONTROLS")
        
        # 1. Building Filter
        buildings = get_building_list()
        building_options = ["All Buildings (Aggregated)"] + buildings
        selected_bldg_option = st.selectbox(
            "Selected Facility / Building",
            options=building_options,
            index=0,
            help="Filter all analytics and diagnostics by a specific building or view portfolio aggregate."
        )
        selected_building = None if selected_bldg_option == "All Buildings (Aggregated)" else selected_bldg_option

        # 2. Date Range Filter
        min_date, max_date = get_date_bounds()
        selected_date_range = st.date_input(
            "Analysis Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter historical energy records within the verified BDG2 dataset interval."
        )

        # Handle single date selection during picker interaction
        if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
        elif isinstance(selected_date_range, tuple) and len(selected_date_range) == 1:
            start_date = selected_date_range[0]
            end_date = max_date
        else:
            start_date, end_date = min_date, max_date

        st.markdown("---")
        
        # 3. Dataset & Model Telemetry
        meta = load_metadata()
        st.markdown("#### 📊 DATASET TELEMETRY")
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; line-height: 1.7; color: #94A3B8;">
                <div>🏢 <strong>Portfolio:</strong> 20 Commercial Offices</div>
                <div>📈 <strong>Records:</strong> 350,880 Hourly Readings</div>
                <div>📅 <strong>Range:</strong> 2016-01-01 to 2017-12-31</div>
                <div>⚡ <strong>Best Model:</strong> XGBoost (R² 0.9983)</div>
                <div>🎯 <strong>Horizon:</strong> 24-Hour Forecast</div>
                <div>🚨 <strong>Anomalies:</strong> 1,677 Statistical Flags</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.72rem; color: #64748B; text-align: center; margin-top: 1rem;">
                BDG2 Energy Intelligence Project<br>
                M.Tech Data Science Project Platform
            </div>
            """,
            unsafe_allow_html=True
        )

    return {
        'selected_building': selected_building,
        'start_date': start_date,
        'end_date': end_date
    }
