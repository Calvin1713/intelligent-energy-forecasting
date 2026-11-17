"""
Feature Engineering and Preprocessing Utilities
Handles temporal features, cyclical sine/cosine encodings, lag feature calculation, and input validation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

FEATURE_COLS = [
    'building_id', 'hour', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168'
]

NUMERIC_COLS = [
    'hour', 'day_of_week', 'day_of_month', 'month',
    'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168'
]

def compute_cyclical_features(hour: int, day_of_week: int) -> Tuple[float, float, float, float]:
    """
    Compute trigonometric representations of hour and day of week.
    hour: 0 to 23
    day_of_week: 0 (Monday) to 6 (Sunday)
    """
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    dow_sin = float(np.sin(2 * np.pi * day_of_week / 7.0))
    dow_cos = float(np.cos(2 * np.pi * day_of_week / 7.0))
    return hour_sin, hour_cos, dow_sin, dow_cos

def build_feature_dataframe(
    building_id: str,
    timestamp: pd.Timestamp,
    lag_1: float,
    lag_24: float,
    lag_168: float,
    rolling_mean_24: float,
    rolling_mean_168: float
) -> pd.DataFrame:
    """
    Construct a properly structured single-row feature DataFrame matching training schema.
    """
    hour = int(timestamp.hour)
    day_of_week = int(timestamp.dayofweek)
    day_of_month = int(timestamp.day)
    month = int(timestamp.month)
    is_weekend = int(day_of_week >= 5)

    hour_sin, hour_cos, dow_sin, dow_cos = compute_cyclical_features(hour, day_of_week)

    row = {
        'building_id': building_id,
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'month': month,
        'is_weekend': is_weekend,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'dow_sin': dow_sin,
        'dow_cos': dow_cos,
        'lag_1': float(lag_1),
        'lag_24': float(lag_24),
        'lag_168': float(lag_168),
        'rolling_mean_24': float(rolling_mean_24),
        'rolling_mean_168': float(rolling_mean_168)
    }

    return pd.DataFrame([row])[FEATURE_COLS]

def validate_prediction_inputs(inputs: Dict[str, Any], building_stats: Dict[str, float]) -> Tuple[bool, list]:
    """
    Validate user inputs against physical and historical statistical bounds.
    Prevents unsupported extrapolation without arbitrary silent changes.
    """
    issues = []
    
    # Check numeric lags are non-negative
    for lag_name in ['lag_1', 'lag_24', 'lag_168', 'rolling_mean_24', 'rolling_mean_168']:
        val = inputs.get(lag_name)
        if val is None or np.isnan(val):
            issues.append(f"Input '{lag_name}' cannot be empty or NaN.")
        elif val < 0:
            issues.append(f"Electricity consumption '{lag_name}' cannot be negative ({val:.2f}).")
        
        # Check excessive extrapolation (> 2.5x historical maximum)
        max_hist = building_stats.get('maximum_consumption', 1000.0)
        if val is not None and val > max_hist * 2.5:
            issues.append(
                f"'{lag_name}' value ({val:.1f}) exceeds 250% of historical maximum ({max_hist:.1f}). "
                "Prediction accuracy degrades significantly outside trained feature domain."
            )
            
    is_valid = len(issues) == 0
    return is_valid, issues
