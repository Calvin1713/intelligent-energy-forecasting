"""
Enterprise KPI Metric Cards
Standardized card components with accent borders, icons, and trend indicators.
"""

import streamlit as st


def render_kpi_card(
    title: str,
    value: str,
    unit: str = "",
    subtitle: str = "",
    accent_color: str = "blue",
    icon: str = ""
):
    """
    Render a single custom KPI card using st.markdown with unsafe_allow_html.
    accent_color: 'blue', 'teal', 'purple', 'amber', 'coral'
    """
    # Build conditional HTML snippets as plain variables first to avoid
    # nested f-string evaluation issues inside triple-quoted strings.
    icon_html = f'<div style="font-size: 1.1rem; line-height: 1;">{icon}</div>' if icon else ""
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    footer_html = f'<div class="kpi-footer">{subtitle}</div>' if subtitle else ""

    html = (
        f'<div class="kpi-card kpi-card-{accent_color}">'
        f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
        f'    <div class="kpi-title">{title}</div>'
        f'    {icon_html}'
        f'  </div>'
        f'  <div style="display:flex;align-items:baseline;margin-top:0.2rem;">'
        f'    <div class="kpi-value">{value}</div>'
        f'    {unit_html}'
        f'  </div>'
        f'  {footer_html}'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_header_banner(
    title: str,
    subtitle: str,
    badge_text: str = "Verified BDG2 Model",
    badge_type: str = "blue"
):
    """Render top page header banner with title and status pill."""
    html = (
        f'<div class="platform-header">'
        f'  <div>'
        f'    <h2 class="platform-title">&#9889; {title}</h2>'
        f'    <div class="platform-subtitle">{subtitle}</div>'
        f'  </div>'
        f'  <div>'
        f'    <span class="platform-badge badge-{badge_type}">{badge_text}</span>'
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
