# Phase 2 — Data Cleaning (Arabian Sea)

## What was built
`src/data_cleaning_arabian_sea.py` — reindexes to a full hourly timeline,
counts NaNs, linear-interpolates gaps <=3h, drops unfillable gaps, removes
duplicate timestamps, saves `arabian_sea_clean.csv`.

## How to run

python .\src\data_cleaning_arabian_sea.py


## Key technical decisions
- Same policy as Atlantic/Pacific/Bay of Bengal: interpolate small gaps, drop the rest.

## Result — this run
232,344 rows in, 232,344 rows out. 0 duplicate timestamps, 0 missing
timestamps, 0 NaNs, 0 rows dropped. Fully complete hourly series,
2000-01-01 to 2026-07-03.

## Files created/updated
- `src/data_cleaning_arabian_sea.py`
- `arabian_sea/data/arabian_sea_clean.csv` (gitignored)
