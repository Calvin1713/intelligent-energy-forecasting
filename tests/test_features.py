import numpy as np
import pandas as pd
import pytest

from energy.data import prepare
from energy.features import feature_columns, lag_hours, make_features


def frame(n=24 * 30, buildings=("A", "B")):
    parts = []
    for i, b in enumerate(buildings):
        ts = pd.date_range("2017-01-01", periods=n, freq="h")
        parts.append(pd.DataFrame({"building_id": b, "timestamp": ts,
                                   "consumption": np.arange(n, dtype=float) + 1000 * i}))
    return pd.concat(parts, ignore_index=True)


def test_day_ahead_has_no_short_lags():
    assert min(lag_hours(24)) == 24 and 1 in lag_hours(1)


def test_lags_stay_inside_each_building():
    f = make_features(frame(), horizon=1)
    first_b = f[f["building_id"] == "B"].iloc[0]
    assert np.isnan(first_b["lag_1"])                 # no value leaked from building A


@pytest.mark.parametrize("h", [1, 24])
def test_no_feature_changes_when_the_future_changes(h):
    """Leakage test: alter every value after an origin; features up to origin + horizon must not move."""
    df = frame()
    origin = pd.Timestamp("2017-01-20 00:00")
    before = make_features(df, horizon=h)
    df.loc[df["timestamp"] > origin, "consumption"] = -999.0
    after = make_features(df, horizon=h)
    window = (before["timestamp"] > origin) & (before["timestamp"] <= origin + pd.Timedelta(hours=h))
    cols = [c for c in feature_columns(before) if c != "building_id"]
    pd.testing.assert_frame_equal(before.loc[window, cols], after.loc[window, cols])


def test_one_output_row_per_input_row():
    df = frame()
    f = make_features(df, horizon=24)
    assert len(f) == len(df) and f.index.equals(df.index)


def test_a_missing_hour_makes_its_lags_missing_but_keeps_the_row():
    df = frame(buildings=("A",))
    df.loc[100, "consumption"] = np.nan                # a masked (invalid) hour
    f = make_features(df, horizon=1)
    assert len(f) == len(df)
    assert np.isnan(f.loc[101, "lag_1"]) and np.isnan(f.loc[124, "lag_24"])
    assert np.isnan(f.loc[100, "target"])              # training drops it by its target, not here


def test_rolling_mean_tolerates_a_few_missing_hours():
    df = frame(buildings=("A",))
    df.loc[200:205, "consumption"] = np.nan            # 6 of 24 hours missing: 75 % still valid
    f = make_features(df, horizon=1)
    assert not np.isnan(f.loc[206, "roll_mean_24_h1"])
    df.loc[200:210, "consumption"] = np.nan            # 11 of 24 missing: not enough
    assert np.isnan(make_features(df, horizon=1).loc[212, "roll_mean_24_h1"])


def test_unsorted_input_is_rejected():
    with pytest.raises(ValueError):
        make_features(frame().sample(frac=1, random_state=0), horizon=1)


def test_prepare_excludes_buildings_and_masks_zero_and_stuck_hours():
    df = frame(buildings=("KEEP", "DROP"))
    df["consumption"] += 1.0                           # no accidental zeros in the synthetic series
    df.loc[df["building_id"].eq("KEEP") & df["timestamp"].eq(pd.Timestamp("2017-01-02 03:00")), "consumption"] = 0.0
    stuck = df["building_id"].eq("KEEP") & df["timestamp"].between("2017-01-10 00:00", "2017-01-11 05:00")
    df.loc[stuck, "consumption"] = 42.0                # 30 identical hours
    out = prepare(df, exclude={"DROP"})
    assert set(out["building_id"]) == {"KEEP"} and len(out) == 24 * 30
    assert out["consumption"].isna().sum() == 1 + 30
