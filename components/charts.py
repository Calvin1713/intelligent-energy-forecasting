"""
Reusable Enterprise Plotly Charts
Optimized visualization presets for energy time-series, load profiles, heatmaps, and ML diagnostics.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Tuple

# Enterprise Energy Color Palette
PALETTE = {
    'blue': '#2563EB',
    'teal': '#0D9488',
    'purple': '#7C3AED',
    'amber': '#D97706',
    'coral': '#E11D48',
    'slate': '#475569',
    'navy': '#0F172A',
    'grid': '#F1F5F9',
    'light_blue': '#93C5FD',
    'light_coral': '#FECDD3'
}

COLORWAY = ['#2563EB', '#0D9488', '#7C3AED', '#D97706', '#E11D48', '#0284C7', '#059669', '#4F46E5']

def apply_chart_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply consistent clean enterprise styling to Plotly figures."""
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#334155"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        colorway=COLORWAY,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=PALETTE['grid'],
            zeroline=False,
            linecolor="#E2E8F0"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=PALETTE['grid'],
            zeroline=False,
            linecolor="#E2E8F0"
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_color="#FFFFFF"
        )
    )
    return fig

def plot_consumption_trend(
    df: pd.DataFrame,
    x_col: str = 'timestamp',
    y_col: str = 'consumption',
    color_col: str = None,
    title: str = "Energy Consumption Trend (kWh)",
    height: int = 380
) -> go.Figure:
    """Create interactive line chart for energy consumption."""
    if color_col and color_col in df.columns:
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='lines',
            name='Consumption',
            line=dict(color=PALETTE['blue'], width=1.75)
        ))
        fig.update_layout(title=title)
        
    fig.update_yaxes(title="Meter Reading (kWh)")
    fig.update_xaxes(title="Timestamp")
    return apply_chart_theme(fig, height=height)

def plot_hourly_profile(
    df: pd.DataFrame,
    group_col: str = 'is_weekend',
    title: str = "Average Hourly Load Profile (24 Hours)"
) -> go.Figure:
    """Create average 24-hour diurnal profile split by weekday vs weekend."""
    temp = df.copy()
    if 'hour' not in temp.columns:
        temp['hour'] = temp['timestamp'].dt.hour
    if 'is_weekend' not in temp.columns:
        temp['is_weekend'] = (temp['timestamp'].dt.dayofweek >= 5).astype(int)
        
    temp['Day Type'] = temp['is_weekend'].map({0: 'Weekday (Mon-Fri)', 1: 'Weekend (Sat-Sun)'})
    agg = temp.groupby(['hour', 'Day Type'])['consumption'].mean().reset_index()
    
    fig = px.line(
        agg,
        x='hour',
        y='consumption',
        color='Day Type',
        color_discrete_map={'Weekday (Mon-Fri)': PALETTE['blue'], 'Weekend (Sat-Sun)': PALETTE['amber']},
        markers=True,
        title=title
    )
    fig.update_xaxes(title="Hour of Day (00:00 - 23:00)", tickmode='linear', tick0=0, dtick=2)
    fig.update_yaxes(title="Mean Consumption (kWh)")
    return apply_chart_theme(fig, height=360)

def plot_hour_day_heatmap(df: pd.DataFrame, title: str = "Hourly Consumption Heatmap") -> go.Figure:
    """Create Day-of-Week vs Hour-of-Day heatmap."""
    temp = df.copy()
    if 'hour' not in temp.columns:
        temp['hour'] = temp['timestamp'].dt.hour
    if 'day_of_week' not in temp.columns:
        temp['day_of_week'] = temp['timestamp'].dt.dayofweek
        
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    temp['day_name'] = temp['day_of_week'].map(lambda x: days[x] if x < len(days) else str(x))
    
    pivot = temp.pivot_table(index='day_name', columns='hour', values='consumption', aggfunc='mean')
    # Reindex to ensure Monday to Sunday order
    pivot = pivot.reindex([d for d in days if d in pivot.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Viridis',
        colorbar=dict(title="kWh"),
        hoverongaps=False
    ))
    fig.update_layout(title=title)
    fig.update_xaxes(title="Hour of Day", dtick=2)
    fig.update_yaxes(title="Day of Week")
    return apply_chart_theme(fig, height=340)

def plot_actual_vs_predicted(
    df: pd.DataFrame,
    sample_size: int = 400,
    title: str = "Actual vs. Predicted Consumption"
) -> go.Figure:
    """Plot actual vs predicted line overlay for test predictions."""
    sample = df.sort_values(by='timestamp').tail(sample_size)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample['timestamp'],
        y=sample['actual_consumption'],
        mode='lines',
        name='Actual Consumption',
        line=dict(color=PALETTE['navy'], width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=sample['timestamp'],
        y=sample['predicted_consumption'],
        mode='lines',
        name='Model Prediction',
        line=dict(color=PALETTE['blue'], width=1.5, dash='dash')
    ))
    fig.update_layout(title=title)
    fig.update_yaxes(title="Consumption (kWh)")
    fig.update_xaxes(title="Timestamp")
    return apply_chart_theme(fig, height=380)

def plot_residual_analysis(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Plot residual scatter and residual histogram."""
    temp = df.copy()
    if 'residual' not in temp.columns:
        temp['residual'] = temp['actual_consumption'] - temp['predicted_consumption']
        
    # 1. Residuals vs Predicted
    fig_scatter = px.scatter(
        temp.sample(min(3000, len(temp)), random_state=42),
        x='predicted_consumption',
        y='residual',
        opacity=0.4,
        color_discrete_sequence=[PALETTE['teal']],
        title="Residuals vs. Fitted Values"
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color=PALETTE['coral'])
    fig_scatter.update_xaxes(title="Predicted Consumption (kWh)")
    fig_scatter.update_yaxes(title="Residual (Actual - Predicted)")
    
    # 2. Residual Distribution Histogram
    fig_hist = px.histogram(
        temp,
        x='residual',
        nbins=60,
        marginal='box',
        color_discrete_sequence=[PALETTE['purple']],
        title="Residual Error Distribution"
    )
    fig_hist.update_xaxes(title="Residual (kWh)")
    fig_hist.update_yaxes(title="Frequency")
    
    return apply_chart_theme(fig_scatter, height=360), apply_chart_theme(fig_hist, height=360)

def plot_feature_importance(df_fi: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plot horizontal bar chart of feature importances."""
    top = df_fi.head(top_n).sort_values(by='importance_mean', ascending=True)
    fig = go.Figure(go.Bar(
        x=top['importance_mean'],
        y=top['feature'],
        orientation='h',
        error_x=dict(type='data', array=top['importance_std'], visible=True),
        marker=dict(color=PALETTE['blue'])
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importance (Permutation Mean)",
        xaxis_title="Importance Score",
        yaxis_title="Feature"
    )
    return apply_chart_theme(fig, height=420)

def plot_anomaly_timeline(df_anom: pd.DataFrame, height: int = 380) -> go.Figure:
    """Plot anomaly scatter on actual consumption time series."""
    fig = go.Figure()
    # Baseline line
    fig.add_trace(go.Scatter(
        x=df_anom['timestamp'],
        y=df_anom['actual_consumption'],
        mode='lines',
        name='Normal Baseline',
        line=dict(color='#CBD5E1', width=1)
    ))
    # Anomaly points
    anoms = df_anom[df_anom['anomaly_flag'] == True]
    if not anoms.empty:
        fig.add_trace(go.Scatter(
            x=anoms['timestamp'],
            y=anoms['actual_consumption'],
            mode='markers',
            name=f'Detected Anomaly ({len(anoms)})',
            marker=dict(color=PALETTE['coral'], size=6, symbol='circle')
        ))
    fig.update_layout(title="Observed Energy Consumption with Flagged Anomalies")
    fig.update_yaxes(title="Consumption (kWh)")
    fig.update_xaxes(title="Timestamp")
    return apply_chart_theme(fig, height=height)
