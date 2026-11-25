"""Load the hourly consumption data and apply the building scope and invalid-hour rule from D-001."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from energy.quality import invalid_mask

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "historical_consumption.parquet"

EXCLUDED = frozenset({  # D-001: valid in less than 50 % of 2017 hours, see docs/decisions.md
    "Shrew_office_Doris", "Shrew_office_Doug", "Shrew_office_Kenneth", "Shrew_office_Lin",
    "Shrew_office_Nora", "Shrew_office_Rose", "Shrew_office_Sherill", "Lamb_office_Callie",
})


def prepare(df: pd.DataFrame, exclude: frozenset[str] | set[str] = EXCLUDED, mask: bool = True) -> pd.DataFrame:
    """Drop excluded buildings, sort, and set invalid hours (zero or stuck) to NaN. Never drops hours."""
    out = df[~df["building_id"].isin(exclude)].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["consumption"] = out["consumption"].astype(float)
    out = out.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    if mask:
        out.loc[invalid_mask(out)["invalid"], "consumption"] = np.nan
    return out


def load_consumption(path: Path | None = None, exclude: frozenset[str] | set[str] = EXCLUDED,
                     mask: bool = True) -> pd.DataFrame:
    """The modelling table: building_id, timestamp, consumption (kWh, NaN where invalid)."""
    return prepare(pd.read_parquet(path or DATA), exclude=exclude, mask=mask)
