import numpy as np
import pandas as pd

from energy.quality import (
    building_scope,
    completeness,
    dead_meters,
    flat_runs,
    invalid_mask,
    zero_share_by_month,
)


def make(values, building="B1", start="2017-01-01"):
    ts = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"building_id": building, "timestamp": ts, "consumption": np.asarray(values, dtype=float)})


def daily_profile(days, building, start="2017-01-01"):
    """A realistic office-like load: changes every hour, never zero."""
    hours = np.arange(24 * days)
    return make(10 + 5 * np.sin(2 * np.pi * hours / 24) + 0.01 * hours, building, start)


def test_completeness_counts_missing_hours():
    df = make([1.0] * 10).drop(index=[3, 4])          # two hours missing
    row = completeness(df).loc["B1"]
    assert row["expected"] == 10 and row["missing_hours"] == 2


def test_dead_meter_found_and_live_meter_not():
    dead = make([0.0] * 744, "DEAD")                   # all of January is zero
    live = make([5.0 + (i % 7) for i in range(744)], "LIVE")
    assert dead_meters(pd.concat([dead, live])) == ["DEAD"]
    assert zero_share_by_month(dead).iloc[0, 0] == 1.0


def test_zeros_on_a_holiday_only_are_not_a_dead_meter():
    df = daily_profile(31, "B1")
    df.loc[df["timestamp"].dt.day == 26, "consumption"] = 0.0   # one day of zeros: 24/744 = 3 %
    assert dead_meters(df) == []


def test_flat_run_ignores_zeros_and_short_runs():
    df = make([1, 2, 7, 7, 7, 7, 7, 7, 3, 0, 0, 0, 0, 0, 0, 0])
    r = flat_runs(df, min_run=6).loc["B1"]
    assert r["flat_runs"] == 1 and r["flat_hours"] == 6


def test_invalid_mask_flags_zeros_and_long_stuck_runs_only():
    values = [5.0, 6.0] + [4.2] * 24 + [7.0, 0.0, 8.0] + [3.3] * 23 + [9.0]
    f = invalid_mask(make(values))
    assert f["zero"].sum() == 1
    assert f["stuck"].sum() == 24                       # the 23-hour run is not stuck
    assert f["invalid"].sum() == 25
    assert len(f) == len(values)                        # rows are flagged, never dropped


def test_stuck_runs_do_not_cross_buildings():
    a = make([1.0, 2.0] + [4.0] * 12, "A")
    b = make([4.0] * 12 + [1.0, 2.0], "B")              # 12 + 12 identical values across the boundary
    f = invalid_mask(pd.concat([a, b], ignore_index=True))
    assert f["stuck"].sum() == 0


def test_invalid_mask_keeps_the_input_index_order():
    a, b = daily_profile(2, "A"), make([0.0] * 48, "B")
    df = pd.concat([b, a], ignore_index=True).sample(frac=1, random_state=0)   # shuffled rows
    f = invalid_mask(df)
    assert f.index.equals(df.index)
    assert f.loc[df["building_id"].eq("B"), "zero"].all()
    assert not f.loc[df["building_id"].eq("A"), "invalid"].any()


def test_building_scope_excludes_a_meter_dead_in_the_evaluation_year():
    live = daily_profile(60, "LIVE", start="2016-12-01")
    dead = daily_profile(60, "DEAD", start="2016-12-01")
    dead.loc[dead["timestamp"] >= "2017-01-01", "consumption"] = 0.0
    s = building_scope(pd.concat([live, dead], ignore_index=True), eval_year=2017)
    assert s.loc["LIVE", "modelled"] and not s.loc["DEAD", "modelled"]
    assert s.loc["DEAD", "valid_share_2017"] == 0.0
