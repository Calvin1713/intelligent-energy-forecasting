# Data

## Files

| File | What it is | Rows |
|---|---|---|
| `historical_consumption.parquet` | Hourly electricity for 20 BDG2 office buildings, 2016-01-01 00:00 to 2017-12-31 23:00. Columns: `building_id`, `timestamp`, `consumption` (kWh) | 350,880 (17,544 per building) |
| `building_metadata.csv` | BDG2 metadata for the same 20 buildings: site, floor area, time zone, year built and more | 20 |
| `raw/` (not committed) | Optional downloads from BDG2, see below | |

The parquet was extracted from BDG2 in the Colab prototype (`v0.1-colab`). The extraction code was not kept; week 4
replaces it with `src/energy/data.py`.

## Source and licence

- **Building Data Genome Project 2 (BDG2)**, BUDS Lab, National University of Singapore:
  <https://github.com/buds-lab/building-data-genome-project-2>
- Paper: Miller, C. et al. (2020). *The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy
  Predictor III competition.* Scientific Data 7, 368. <https://doi.org/10.1038/s41597-020-00712-x>
- **Licence:** the BDG2 repository is licensed under Creative Commons Attribution-ShareAlike 4.0 (its `LICENSE`
  file). This repository redistributes a derived subset, so the files in `data/` stay under CC BY-SA 4.0 with
  attribution to the BDG2 authors.

## Units

**kWh: the electricity used during each hour.** The BDG2 paper states that every site's raw readings were converted
to *kWh sum* for energy meters, each row being the energy summed over that hour (paper, Tables 4 and 5 and the
data-records section; checked 25 Nov 2026). Because each reading covers exactly one hour, the value in kWh is also
the building's average demand in kW over that hour, which is how peak demand is read in this project.

The dashboard exports from the prototype still label the unit "meter reading (check dataset documentation)". That
label is replaced when the exports are regenerated from code (week 10).

## Known quality issues (audit: `notebooks/01_data_audit.ipynb`, decision D-001)

- No missing rows, nulls, duplicates or negative values. Gaps were therefore **filled** before this file was made.
- **Zero readings:** 7 Shrew buildings read 0 from late January 2017 (Rose from late March) to the end of the data;
  Hog_office_Alisha reads 0 until 27 July 2016 (late meter start). A few live buildings have zero stretches of up to
  28 hours.
- **Stuck readings:** runs of one non-zero value for 24+ hours, mostly site-wide at Robin (shared dates) and Lamb.
  Longest: Robin_office_Lindsay at 3.94 kWh for 5,195 hours (Mar–Oct 2016). Lamb_office_Callie is stuck at 0.0007 kWh
  from 24 Aug 2017 to the end.
- Rule used by the code: zero readings and stuck runs are **invalid hours**, masked as missing, never imputed.
  BDG2's own cleaned file also removes zero electricity readings. Per-building counts:
  `reports/data_quality_summary.csv`.
- Open question: whether these zeros and constants are in BDG2's raw file or were introduced by the prototype's
  preprocessing. Section 6 of the audit notebook answers it once the raw file is downloaded.

## Downloads (optional)

Large BDG2 files are stored with Git LFS, so download them from the GitHub web page rather than with `git clone`:

1. Open <https://github.com/buds-lab/building-data-genome-project-2/tree/master/data>.
2. For the raw electricity check: `meters/raw/electricity.csv` (about 175 MB) → click the file → **Download raw file**
   → save as `data/raw/electricity.csv`.
3. For week 9: `weather/weather.csv` → save as `data/raw/weather.csv`.

`data/raw/` is listed in `.gitignore`, so these files are never committed.
