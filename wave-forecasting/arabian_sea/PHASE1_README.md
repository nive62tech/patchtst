# Phase 1 — Data Loading & Merging (Arabian Sea)

## What was built
`src/data_pipeline_arabian_sea.py` — loads the two raw ERA5 files
(wind vars, wave vars) for buoy AD07 (14.904N, 69.050E), inner-joins them
on `valid_time`, sorts chronologically, asserts no duplicate timestamps
and no missing expected columns, saves `arabian_sea_merged.csv`.

## How to run

python .\src\data_pipeline_arabian_sea.py


## Key technical decisions
- Same inner-join approach as Bay of Bengal.
- Wave file's own `latitude`/`longitude` dropped before merge; wind file's
  coordinates (15.0N, 69.0E) kept as the record of location.
- `blh` carried through the merge; dropped in Phase 3.

## Result — this run
File 1: 232,344 rows. File 2: 232,344 rows. Merged: 232,344 rows
(zero row loss). Date range: 2000-01-01 to 2026-07-03.

## Files created/updated
- `src/data_pipeline_arabian_sea.py`
- `arabian_sea/data/arabian_sea_merged.csv` (gitignored)
