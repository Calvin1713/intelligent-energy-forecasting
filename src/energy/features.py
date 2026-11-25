"""Calendar features and horizon-safe lag features.

Rule: when forecasting `horizon` hours ahead, every feature of a target hour t may only use readings at
t - horizon or earlier (the forecast is made at t - horizon). tests/test_features.py proves this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
LAG_CANDIDATES = [1, 2, 3, 24, 48, 168, 336]
ROLL_WINDOWS = [24, 168]
ROLL_MIN_VALID = 0.75   # a rolling mean needs 75 % of its window valid (masked hours are NaN)


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=ts.index)
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.day_of_week
    out["month"] = ts.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def lag_hours(horizon: int) -> list[int]:
    """Lags available when forecasting `horizon` hours ahead (never shorter than the horizon)."""
    return [lag for lag in LAG_CANDIDATES if lag >= horizon]


def make_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """df: building_id, timestamp, consumption, sorted by building then time, one row per hour.

    Returns one row per input row: calendar features, horizon-safe lags and rolling means, building_id,
    timestamp and target. Missing readings give missing features; no row is dropped here.
    """
    if not df[["building_id", "timestamp"]].equals(
            df[["building_id", "timestamp"]].sort_values(["building_id", "timestamp"])):
        raise ValueError("df must be sorted by building_id, timestamp")
    g = df.groupby("building_id", observed=True)["consumption"]
    feats = calendar_features(df["timestamp"])
    for lag in lag_hours(horizon):
        feats[f"lag_{lag}"] = g.shift(lag)
    latest_known = g.shift(horizon)                       # the newest reading available at forecast time
    by_b = latest_known.groupby(df["building_id"], observed=True)
    for w in ROLL_WINDOWS:
        min_p = int(w * ROLL_MIN_VALID)
        feats[f"roll_mean_{w}_h{horizon}"] = by_b.transform(lambda s: s.rolling(w, min_periods=min_p).mean())
    feats["building_id"] = df["building_id"].astype("category")
    feats["timestamp"] = df["timestamp"]
    feats["target"] = df["consumption"]
    return feats


def feature_columns(f: pd.DataFrame) -> list[str]:
    """Model inputs: everything except the target and the timestamp."""
    return [c for c in f.columns if c not in ("target", "timestamp")]
