"""
Model Prediction & Inference Manager
Handles runtime pipeline loading, live feature prediction, and historical context comparisons.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models')

_CACHED_PIPELINE = None

def get_prediction_pipeline():
    """
    Retrieve loaded prediction pipeline with caching.
    Uses native runtime pipeline trained with exact identical parameters and features.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None:
        return _CACHED_PIPELINE

    native_path = os.path.join(MODELS_DIR, 'native_runtime_pipeline.joblib')
    if os.path.exists(native_path):
        try:
            _CACHED_PIPELINE = joblib.load(native_path)
            return _CACHED_PIPELINE
        except Exception as e:
            print(f"Warning: Failed to load native pipeline: {e}")

    # Fallback to runtime joblib if present
    runtime_path = os.path.join(MODELS_DIR, 'model_runtime.joblib')
    if os.path.exists(runtime_path):
        try:
            _CACHED_PIPELINE = joblib.load(runtime_path)
            return _CACHED_PIPELINE
        except Exception as e:
            print(f"Warning: Failed to load model_runtime: {e}")

    return None

def predict_energy(feature_df: pd.DataFrame) -> Optional[float]:
    """
    Execute real-time model inference for a feature DataFrame.
    Returns predicted consumption value in original meter units (kWh).
    """
    pipeline = get_prediction_pipeline()
    if pipeline is None:
        return None
    
    try:
        preds = pipeline.predict(feature_df)
        # Meter readings cannot be physically negative
        return max(0.0, float(preds[0]))
    except Exception as e:
        print(f"Prediction inference error: {e}")
        return None

def get_building_historical_stats(df_hist: pd.DataFrame, building_id: str) -> Dict[str, float]:
    """
    Calculate statistical benchmarks for a building from historical actuals.
    """
    if df_hist.empty or 'building_id' not in df_hist.columns:
        return {
            'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
            'p10': 0.0, 'p25': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0
        }
    
    b_data = df_hist[df_hist['building_id'] == building_id]['consumption'].dropna()
    if b_data.empty:
        return {
            'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0,
            'p10': 0.0, 'p25': 0.0, 'p75': 0.0, 'p90': 0.0, 'p95': 0.0
        }

    return {
        'mean': float(b_data.mean()),
        'median': float(b_data.median()),
        'min': float(b_data.min()),
        'max': float(b_data.max()),
        'std': float(b_data.std()),
        'p10': float(b_data.quantile(0.10)),
        'p25': float(b_data.quantile(0.25)),
        'p75': float(b_data.quantile(0.75)),
        'p90': float(b_data.quantile(0.90)),
        'p95': float(b_data.quantile(0.95))
    }
