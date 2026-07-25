# Phase 2 — Data Cleaning (Bay of Bengal)

## What was built
`src/data_cleaning_bay_of_bengal.py` — reindexes to a full hourly timeline,
counts NaNs, linear-interpolates gaps <=3h, drops unfillable gaps, removes
duplicate timestamps, saves `bay_of_bengal_clean.csv`.

## How to run

python .\src\data_cleaning_bay_of_bengal.py


## Key technical decisions
- Same policy as Atlantic/Pacific: interpolate small gaps, drop the rest.
- `blh` still present at this stage; dropped in Phase 3 output selection, not here.

## Result — this run
232,344 rows in, 232,344 rows out. 0 duplicate timestamps, 0 missing
timestamps, 0 NaNs, 0 rows dropped. Fully complete hourly series,
2000-01-01 to 2026-07-03.

## Files created/updated
- `src/data_cleaning_bay_of_bengal.py`
- `bay_of_bengal/data/bay_of_bengal_clean.csv` (gitignored)
