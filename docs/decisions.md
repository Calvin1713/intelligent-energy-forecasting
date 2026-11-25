# Decision log

One entry per decision that changes what the code does or what the results mean. Newest last.

---

## D-001 · Building scope and invalid hours

**Date:** 2026-11-25 · **Status:** accepted · **Evidence:** `notebooks/01_data_audit.ipynb`,
`reports/data_quality_summary.csv`, `reports/figures/zero_share_heatmap.png`

### Context

The prototype modelled all 20 office buildings and treated every reading as real consumption. The file has no
missing hours at all, which means gaps were filled. The audit found two kinds of fill:

- **Zeros.** 7 Shrew buildings read exactly 0 kWh from late January 2017 (Shrew_office_Rose from late March) to the
  end of the data: 6,677 to 8,193 zero hours each. They make up 24,332 of the 69,520 prototype test rows (35 %).
  Hog_office_Alisha reads 0 for its first 5,003 hours (until 27 July 2016) and is live afterwards.
- **Stuck values.** Runs of one non-zero value for 24 hours or more: 27,107 hours across 11 buildings. The Robin
  buildings share the same flat dates (site-wide outages). Lamb_office_Callie sits at 0.0007 kWh from 24 August 2017
  to the end, which covers about 90 % of its test period.

Neither is consumption. Predicting a zero or a constant is easy, so these hours flatter the metric: on the prototype's
own predictions, MAE goes from 3.57 kWh (all 20 buildings) to 5.01 kWh (valid hours of the buildings kept).

### Decision

1. **Invalid hour:** a reading of exactly 0, or part of a run of 24+ identical non-zero readings. Invalid hours are
   masked as missing. They are never a training target or a test actual, and lag features that would use them
   become missing. Nothing is imputed.
2. **Building scope:** a building is modelled if at least 50 % of its 2017 hours are valid (2017 is the evaluation
   year for every backtest).

| Result | Buildings |
|---|---|
| **Modelled (12)** | Eagle_office_Mable, Hog_office_Alisha, Hog_office_Bill, Hog_office_Gustavo, Lamb_office_Vasiliki, Lamb_office_Velma, Robin_office_Gayle, Robin_office_Lindsay, Robin_office_Sammie, Robin_office_Victor, Robin_office_Wai, Robin_office_Zelma |
| **Excluded (8)** | Shrew_office_Doris, Doug, Kenneth, Lin, Nora, Sherill (6.6–6.8 % of 2017 valid), Shrew_office_Rose (23.8 %), Lamb_office_Callie (39.8 %) |

Lowest kept: Lamb_office_Vasiliki and Lamb_office_Velma at 62.5 % (a stuck stretch from 10 May to 8 Aug 2017 is
masked). Inside the 12 modelled buildings, 26,954 of 210,528 hours (12.8 %) are masked, most of them in 2016.

The excluded buildings stay in the audit notebook as the data-quality case study.

### Why these thresholds

- **Zero = invalid:** an occupied office never draws exactly 0 kWh in an hour. BDG2's own cleaned file applies the
  same rule ("zero readings in electricity meters are removed").
- **24 hours for stuck:** meters report to 2–4 decimals, so a real load does not repeat for a whole day. Shorter
  repeats exist on Eagle's 2-decimal meter at night and are kept. The 2017 valid shares split cleanly, with nothing
  between 39.8 % and 62.5 %, so any threshold from 0.40 to 0.62 gives the same scope.
- **The plan's original rule** (`zero share ≥ 0.9 in any month`) flags 8 buildings, including Alisha, whose zeros
  are only at the start, and misses Callie, whose fault is a near-zero constant rather than a zero. A valid-hours
  rule handles both cases.

### Alternatives considered

- **Impute the dead periods** (for example from similar buildings): rejected. Months of imputed load would invent
  the very values being forecast, and the errors on them would measure the imputation, not the model.
- **Drop invalid rows:** rejected. Dropping rows silently breaks the hourly spacing that lag features rely on.
  Masking keeps the timeline intact and makes the affected lags visibly missing.
- **Keep Shrew's 2016 data for training only:** possible, but these buildings could never be evaluated in 2017.
  Revisit if the model needs more training buildings.

### Consequences

- The plan's "13 live buildings" becomes **12** everywhere (README, posts, week-5 comparison).
- Headline MAE will rise; that is the true error on buildings with working meters. WAPE becomes the headline metric
  from week 5 because it reads as a percentage and is not shrunk by free zeros.
- Week 4 (implemented): `energy.data.load_consumption()` drops the 8 buildings and sets invalid hours to NaN
  (26,954 of 210,528 hours). Rows are kept, so lags of a masked hour are missing. Rolling means need 75 % of their
  window valid (with 100 %, 10.6 % of usable rows would lose the weekly mean; with 75 %, 3.9 %). Training drops rows
  by a missing target only; XGBoost handles missing features.

### Open question

Are the zeros and stuck values present in BDG2's raw electricity file, or were they created by the prototype's
preprocessing (for example missing values filled with 0)? The answer changes the explanation, not the decision.
Check: section 6 of the audit notebook, after downloading the raw file (`data/README.md`).

### With more time

Replace the excluded buildings with other office buildings from BDG2's cleaned meter file that pass the same rule,
to bring the scope back toward 20.
