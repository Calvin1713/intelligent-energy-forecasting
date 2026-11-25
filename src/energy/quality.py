"""Data-quality checks for hourly meter data with columns building_id, timestamp, consumption.

Rules (decision D-001 in docs/decisions.md):
- A zero electricity reading in an office building is treated as missing, not as low consumption.
  BDG2's own cleaned meter file applies the same rule ("zero readings in electricity meters are removed").
- A non-zero value repeated for 24 or more consecutive hours is a stuck meter or a filled gap, also missing.
- A building is modelled only if at least half of its hours in the evaluation year are valid.
"""
from __future__ import annotations

import pandas as pd

STUCK_HOURS = 24        # an office meter never repeats a 4-decimal reading for a whole day
MIN_VALID_SHARE = 0.5   # share of valid hours a building needs in the evaluation year


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Expected vs present hourly rows per building."""
    span = df.groupby("building_id")["timestamp"].agg(first="min", last="max", rows="count")
    span["expected"] = ((span["last"] - span["first"]) / pd.Timedelta("1h")).astype(int) + 1
    span["missing_hours"] = span["expected"] - span["rows"]
    span["null_readings"] = df.groupby("building_id")["consumption"].apply(lambda s: int(s.isna().sum()))
    return span


def zero_share_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Share of zero readings per building (rows) and calendar month (columns)."""
    m = df.assign(month=df["timestamp"].dt.to_period("M"), zero=df["consumption"].eq(0).astype(float))
    return m.pivot_table(index="building_id", columns="month", values="zero", aggfunc="mean")


def dead_meters(df: pd.DataFrame, threshold: float = 0.9, min_months: int = 1) -> list[str]:
    """Buildings with at least `min_months` months where zero readings reach `threshold`."""
    z = zero_share_by_month(df)
    flagged = (z >= threshold).sum(axis=1) >= min_months
    return sorted(flagged[flagged].index)


def _run_lengths(df: pd.DataFrame) -> pd.Series:
    """Length of the run of identical consecutive readings that each row belongs to (per building).

    Expects df sorted by building_id, timestamp. Missing readings (NaN) never join a run.
    """
    c = df["consumption"]
    same_as_prev = c.eq(c.groupby(df["building_id"]).shift()) & c.notna()
    run_id = (~same_as_prev).cumsum()
    return c.groupby(run_id).transform("size")


def flat_runs(df: pd.DataFrame, min_run: int = STUCK_HOURS) -> pd.DataFrame:
    """Count runs of >= min_run identical non-zero readings per building (stuck meters or filled gaps)."""
    out = []
    for b, g in df.sort_values("timestamp").groupby("building_id"):
        s = g["consumption"]
        change = s.ne(s.shift()).cumsum()
        lengths = s.groupby(change).agg(["size", "first"])
        stuck = lengths[(lengths["size"] >= min_run) & (lengths["first"] != 0)]
        out.append({"building_id": b, "flat_runs": len(stuck), "flat_hours": int(stuck["size"].sum()),
                    "longest_run": int(stuck["size"].max()) if len(stuck) else 0})
    return pd.DataFrame(out).set_index("building_id")


def invalid_mask(df: pd.DataFrame, min_stuck: int = STUCK_HOURS) -> pd.DataFrame:
    """Per-row flags: zero, stuck (non-zero repeated >= min_stuck hours), null, and invalid (any of them).

    Returns a frame aligned with df's index. Rows are flagged, never dropped.
    """
    order = df.sort_values(["building_id", "timestamp"])
    run_len = _run_lengths(order).reindex(df.index)
    c = df["consumption"]
    flags = pd.DataFrame(index=df.index)
    flags["zero"] = c.eq(0)
    flags["stuck"] = c.ne(0) & c.notna() & (run_len >= min_stuck)
    flags["null"] = c.isna()
    flags["invalid"] = flags["zero"] | flags["stuck"] | flags["null"]
    return flags


def building_scope(df: pd.DataFrame, eval_year: int = 2017, min_valid: float = MIN_VALID_SHARE) -> pd.DataFrame:
    """Valid-hour share per building overall and in `eval_year`, and whether the building is modelled."""
    flags = invalid_mask(df)
    valid = ~flags["invalid"]
    in_year = df["timestamp"].dt.year.eq(eval_year)
    out = pd.DataFrame({
        "hours": df.groupby("building_id").size(),
        "zero_hours": flags["zero"].groupby(df["building_id"]).sum(),
        "stuck_hours": flags["stuck"].groupby(df["building_id"]).sum(),
        "valid_share": valid.groupby(df["building_id"]).mean(),
        f"valid_share_{eval_year}": valid[in_year].groupby(df.loc[in_year, "building_id"]).mean(),
    })
    out["modelled"] = out[f"valid_share_{eval_year}"].fillna(0) >= min_valid
    return out
