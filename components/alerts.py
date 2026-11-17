"""
Alert and Notification Components
Standardized callout banners for data freshness, scientific disclaimers, and operational warnings.
"""

import streamlit as st

def render_alert(message: str, alert_type: str = "info", icon: str = None):
    """
    Render custom enterprise alert box.
    alert_type: 'info', 'success', 'warning', 'danger'
    """
    default_icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'danger': '🚨'
    }
    use_icon = icon or default_icons.get(alert_type, 'ℹ️')
    
    st.markdown(
        f"""
        <div class="custom-alert alert-{alert_type}">
            <div style="font-size: 1.15rem; line-height: 1;">{use_icon}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_scientific_notice():
    """Render mandatory BDG2 scientific integrity notice."""
    render_alert(
        "<strong>Scientific & Operational Disclosure:</strong> This platform utilizes historical data from the "
        "<strong>Building Data Genome Project 2 (BDG2)</strong>. It is not connected to live IoT meters. "
        "Anomalies represent statistical deviations rather than confirmed equipment faults. "
        "Calculated savings and scenarios represent modeling opportunities for facility teams.",
        alert_type="info"
    )

def render_empty_state(title: str, description: str):
    """Render polished empty state when filtered data yields 0 records."""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem 1.5rem; background: #FFFFFF; border-radius: 12px; border: 1px dashed #CBD5E1; margin: 1.5rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍</div>
            <h4 style="font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">{title}</h4>
            <p style="color: #64748B; font-size: 0.875rem; max-width: 480px; margin: 0 auto;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
