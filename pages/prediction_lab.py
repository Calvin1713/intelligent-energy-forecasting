"""
Page 8 — Interactive Prediction Lab
Real-time machine learning inference workspace utilizing the fitted preprocessing pipeline
and trained XGBoost model. Enables custom operational scenario testing, historical test replay,
statistical input validation, and baseline benchmark comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, time

from utils.data_loader import (
    load_metadata, load_building_summary, load_historical_consumption,
    load_predictions, get_building_list
)
from utils.preprocessing import (
    build_feature_dataframe, validate_prediction_inputs, FEATURE_COLS
)
from utils.prediction import (
    predict_energy, get_building_historical_stats, get_prediction_pipeline
)
from components.kpi_cards import render_kpi_card, render_header_banner
from components.charts import apply_chart_theme, PALETTE
from components.alerts import render_alert

def render_page(filters: dict):
    render_header_banner(
        title="Interactive Energy Prediction Lab",
        subtitle="Live machine learning inference engine evaluating future operational scenarios and feature sensitivities.",
        badge_text="Native Inference Engine",
        badge_type="blue"
    )

    df_meta = load_metadata()
    df_summary = load_building_summary()
    df_hist = load_historical_consumption()
    df_pred = load_predictions()
    building_list = get_building_list()

    selected_bldg = filters.get('selected_building') or (building_list[0] if building_list else "Eagle_office_Mable")

    render_alert(
        "<strong>Pipeline Telemetry:</strong> Inference is executed through the fitted scikit-learn preprocessing pipeline "
        "(StandardScaler + OneHotEncoder) into the <strong>XGBoost Regressor</strong> (depth: 7, n_estimators: 400). "
        "Predictions are returned in <strong>kWh (kilowatt-hours)</strong> representing next-hour meter draw. "
        "Inputs are strictly validated against training domain bounds to avoid unsupported extrapolation.",
        alert_type="info"
    )

    # 1. Building Selection & Historical Stats
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">🏢 Facility Selection & Baseline Context</h4>
                    <div class="analytics-card-desc">Select target building to inspect its historical statistical envelope</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )
    
    active_bldg = st.selectbox(
        "Select Target Facility for Prediction:",
        options=building_list,
        index=building_list.index(selected_bldg) if selected_bldg in building_list else 0
    )

    # Fetch statistical bounds for selected building
    b_stats = get_building_historical_stats(df_hist, active_bldg)
    
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        render_kpi_card(
            title="Historical Median",
            value=f"{b_stats['median']:.1f}",
            unit="kWh",
            subtitle="50th percentile demand",
            accent_color="blue",
            icon="📊"
        )
    with kcol2:
        render_kpi_card(
            title="Historical Mean",
            value=f"{b_stats['mean']:.1f}",
            unit="kWh",
            subtitle=f"Std Dev: ±{b_stats.get('std', 0):.1f} kWh",
            accent_color="teal",
            icon="📈"
        )
    with kcol3:
        render_kpi_card(
            title="Historical Max (Peak)",
            value=f"{b_stats['max']:.1f}",
            unit="kWh",
            subtitle="Recorded ceiling",
            accent_color="coral",
            icon="🔥"
        )
    with kcol4:
        render_kpi_card(
            title="Typical Base Load (p10)",
            value=f"{b_stats['p10']:.1f}",
            unit="kWh",
            subtitle="Overnight standby baseline",
            accent_color="purple",
            icon="🌙"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Prediction Mode Selection
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">⚙️ Prediction Configuration & Scenario Inputs</h4>
                    <div class="analytics-card-desc">Choose between replaying verified test instances or designing custom operational scenarios</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    pred_mode = st.radio(
        "Select Prediction Mode:",
        options=["Custom Operational Scenario (Live What-If)", "Historical Test Set Replay (Validation)"],
        horizontal=True
    )

    if pred_mode == "Historical Test Set Replay (Validation)":
        b_preds = df_pred[df_pred['building_id'] == active_bldg].sort_values(by='timestamp')
        if b_preds.empty:
            st.warning(f"No test set records found for {active_bldg}.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 0.8rem;">
                Select a specific test observation timestamp to compare the pipeline's prediction directly against the actual meter reading:
            </div>
            """,
            unsafe_allow_html=True
        )

        sample_timestamps = b_preds['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        # Default to a mid-point test record
        selected_ts_str = st.selectbox("Select Test Timestamp:", options=sample_timestamps, index=len(sample_timestamps)//2)
        
        match_row = b_preds[b_preds['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S") == selected_ts_str].iloc[0]
        actual_val = float(match_row['actual_consumption'])
        model_pred = float(match_row['predicted_consumption'])
        abs_err = float(match_row['absolute_error'])
        pct_err = (abs_err / max(1.0, actual_val)) * 100.0

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            render_kpi_card(
                title="Actual Meter Reading",
                value=f"{actual_val:.2f}",
                unit="kWh",
                subtitle="Verified BDG2 ground truth",
                accent_color="navy" if "navy" in ["blue"] else "blue",
                icon="⚡"
            )
        with rcol2:
            render_kpi_card(
                title="Model Predicted Demand",
                value=f"{model_pred:.2f}",
                unit="kWh",
                subtitle="XGBoost prediction",
                accent_color="purple",
                icon="🤖"
            )
        with rcol3:
            render_kpi_card(
                title="Prediction Error",
                value=f"{abs_err:.2f}",
                unit="kWh",
                subtitle=f"{pct_err:.1f}% relative error",
                accent_color="teal" if pct_err < 10 else "amber",
                icon="🎯"
            )

    else:
        # Custom Scenario Mode
        st.markdown(
            """
            <div style="font-size: 0.825rem; color: #64748B; margin-bottom: 1rem;">
                Define temporal conditions and autoregressive lag inputs. Values are initialized to the selected facility's historical medians:
            </div>
            """,
            unsafe_allow_html=True
        )

        # Date & Time controls
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            target_date = st.date_input("Target Date:", value=date(2018, 1, 15))
        with tcol2:
            target_hour = st.slider("Target Hour of Day (00:00 - 23:00):", min_value=0, max_value=23, value=14, step=1)

        sim_timestamp = pd.Timestamp(datetime.combine(target_date, time(target_hour, 0)))

        st.markdown("##### Autoregressive Demand History (Lags & Rolling Means):")
        
        # Populate realistic defaults based on historical stats
        default_lag1 = round(b_stats['median'], 1)
        default_lag24 = round(b_stats['mean'], 1)
        default_lag168 = round(b_stats['p75'], 1)
        default_rm24 = round(b_stats['mean'], 1)
        default_rm168 = round(b_stats['median'], 1)

        lcol1, lcol2, lcol3 = st.columns(3)
        with lcol1:
            in_lag1 = st.number_input(
                "Immediate Prior Hour (lag_1) [kWh]:",
                min_value=0.0,
                max_value=float(b_stats['max'] * 3.0),
                value=float(default_lag1),
                step=1.0,
                help="Electricity consumed in the immediately preceding hour."
            )
            in_rm24 = st.number_input(
                "24-Hour Rolling Mean [kWh]:",
                min_value=0.0,
                max_value=float(b_stats['max'] * 3.0),
                value=float(default_rm24),
                step=1.0,
                help="Mean consumption across preceding 24 hours."
            )
        with lcol2:
            in_lag24 = st.number_input(
                "Same Hour Yesterday (lag_24) [kWh]:",
                min_value=0.0,
                max_value=float(b_stats['max'] * 3.0),
                value=float(default_lag24),
                step=1.0,
                help="Electricity consumed exactly 24 hours prior."
            )
            in_rm168 = st.number_input(
                "7-Day Rolling Mean [kWh]:",
                min_value=0.0,
                max_value=float(b_stats['max'] * 3.0),
                value=float(default_rm168),
                step=1.0,
                help="Mean consumption across preceding 168 hours."
            )
        with lcol3:
            in_lag168 = st.number_input(
                "Same Hour Last Week (lag_168) [kWh]:",
                min_value=0.0,
                max_value=float(b_stats['max'] * 3.0),
                value=float(default_lag168),
                step=1.0,
                help="Electricity consumed at this exact time 7 days ago."
            )

        # Build feature DataFrame
        feature_df = build_feature_dataframe(
            building_id=active_bldg,
            timestamp=sim_timestamp,
            lag_1=in_lag1,
            lag_24=in_lag24,
            lag_168=in_lag168,
            rolling_mean_24=in_rm24,
            rolling_mean_168=in_rm168
        )

        # Validate inputs
        inputs_dict = {
            'lag_1': in_lag1, 'lag_24': in_lag24, 'lag_168': in_lag168,
            'rolling_mean_24': in_rm24, 'rolling_mean_168': in_rm168
        }
        is_valid, validation_issues = validate_prediction_inputs(inputs_dict, b_stats)

        if not is_valid:
            for issue in validation_issues:
                st.warning(f"⚠️ {issue}")

        # Run Prediction
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Generate Energy Consumption Prediction", type="primary", disabled=not is_valid):
            pred_value = predict_energy(feature_df)
            
            if pred_value is not None:
                st.markdown(
                    f"""
                    <div style="background: #EFF6FF; border: 2px solid #3B82F6; border-radius: 12px; padding: 1.5rem; margin-top: 1.25rem; text-align: center;">
                        <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #1E40AF; letter-spacing: 0.05em;">
                            Predicted Next-Hour Electricity Demand
                        </div>
                        <div style="font-size: 2.75rem; font-weight: 800; color: #0F172A; margin: 0.4rem 0;">
                            {pred_value:.2f} <span style="font-size: 1.25rem; font-weight: 600; color: #64748B;">kWh</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #475569;">
                            Facility: <strong>{active_bldg}</strong> | Target Timestamp: <strong>{sim_timestamp.strftime('%Y-%m-%d %H:%M')}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Contextual Gauge: Where does this prediction fall relative to history?
                st.markdown("##### Historical Context Gauge:")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Demand vs. Historical Baseline (kWh)", 'font': {'size': 14}},
                    delta={'reference': b_stats['median'], 'increasing': {'color': PALETTE['coral']}, 'decreasing': {'color': PALETTE['teal']}},
                    gauge={
                        'axis': {'range': [0, max(pred_value * 1.2, b_stats['max'] * 1.1)]},
                        'bar': {'color': PALETTE['blue']},
                        'steps': [
                            {'range': [0, b_stats['p25']], 'color': "#F1F5F9"},
                            {'range': [b_stats['p25'], b_stats['p75']], 'color': "#E2E8F0"},
                            {'range': [b_stats['p75'], b_stats['max']], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': PALETTE['coral'], 'width': 3},
                            'thickness': 0.75,
                            'value': b_stats['p95']
                        }
                    }
                ))
                apply_chart_theme(fig_gauge, height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Download Prediction Result
                res_df = feature_df.copy()
                res_df['predicted_consumption'] = pred_value
                res_df['prediction_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                res_csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Prediction Result CSV",
                    data=res_csv,
                    file_name=f"prediction_{active_bldg}_{sim_timestamp.strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.error("Prediction could not be computed. Please check input parameters.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Technical Limitations Explanation
    st.markdown(
        """
        <div class="analytics-card">
            <div class="analytics-card-header">
                <div>
                    <h4 class="analytics-card-title">⚠️ Model Specifications & Prediction Limitations</h4>
                    <div class="analytics-card-desc">Academic and engineering boundaries governing model inference</div>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.6;">
                <ul>
                    <li><strong>Feature Alignment:</strong> The model expects 15 exact features: 1 categorical (<code>building_id</code>) and 14 numeric variables including cyclical sine/cosine encodings of hour and day of week.</li>
                    <li><strong>Autoregressive Lag Dependency:</strong> The accuracy of next-hour forecasts relies on the availability of <code>lag_1</code>, <code>lag_24</code>, and <code>lag_168</code>. If preceding sensor readings are missing, iterative multi-step forecasts accumulate uncertainty.</li>
                    <li><strong>Weather & Occupancy Exclusions:</strong> This model configuration evaluates temporal and autoregressive features. Exterior ambient temperature, solar irradiance, and live turnstile counts were not included in this pipeline and represent potential avenues for model enhancement.</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
