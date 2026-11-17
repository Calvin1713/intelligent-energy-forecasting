"""
Data Loading & Caching Utility
Provides cached, validated loading for all energy platform datasets and metadata.
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EXPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dashboard_exports')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load dashboard metadata JSON with fallbacks."""
    path = os.path.join(EXPORTS_DIR, 'dashboard_metadata.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_data(show_spinner=False)
def load_building_summary() -> pd.DataFrame:
    """Load building summary data."""
    path = os.path.join(EXPORTS_DIR, 'building_summary.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_building_metadata() -> pd.DataFrame:
    """Load architectural building metadata (sqm, timezone, floors, etc.)."""
    path = os.path.join(DATA_DIR, 'building_metadata.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame:
    """Load machine learning model evaluation metrics."""
    path = os.path.join(EXPORTS_DIR, 'model_metrics.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Sort by R2 descending
        return df.sort_values(by='R2', ascending=False).reset_index(drop=True)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    """Load model permutation/tree feature importance scores."""
    path = os.path.join(EXPORTS_DIR, 'feature_importance.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df.sort_values(by='importance_mean', ascending=False).reset_index(drop=True)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame:
    """Load 24-hour ahead energy forecast table."""
    path = os.path.join(EXPORTS_DIR, 'forecast.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Load test set model predictions."""
    path = os.path.join(EXPORTS_DIR, 'predictions.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_anomalies() -> pd.DataFrame:
    """Load detected anomaly events with dual detection flags."""
    path = os.path.join(EXPORTS_DIR, 'anomalies.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_peak_periods() -> pd.DataFrame:
    """Load peak demand period flags and threshold series."""
    path = os.path.join(EXPORTS_DIR, 'peak_periods.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    """Load facility operational energy recommendations."""
    path = os.path.join(EXPORTS_DIR, 'recommendations.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_historical_consumption() -> pd.DataFrame:
    """Load complete 2-year hourly historical consumption records for 20 buildings."""
    parquet_path = os.path.join(DATA_DIR, 'historical_consumption.parquet')
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

def get_building_list() -> list:
    """Return sorted list of all 20 analyzed buildings."""
    meta = load_metadata()
    if 'selected_buildings' in meta:
        return sorted(meta['selected_buildings'])
    summary = load_building_summary()
    if not summary.empty and 'building_id' in summary.columns:
        return sorted(summary['building_id'].tolist())
    return []

def get_date_bounds() -> tuple:
    """Return min and max dates of the full dataset."""
    df = load_historical_consumption()
    if not df.empty:
        return df['timestamp'].min().date(), df['timestamp'].max().date()
    # Fallback to metadata
    meta = load_metadata()
    start = pd.to_datetime(meta.get('dataset_start', '2016-01-01')).date()
    end = pd.to_datetime(meta.get('dataset_end', '2017-12-31')).date()
    return start, end
