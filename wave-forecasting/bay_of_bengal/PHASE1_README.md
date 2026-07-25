# Phase 1 — Data Loading & Merging (Bay of Bengal)

## What was built
`src/data_pipeline_bay_of_bengal.py` — loads the two raw ERA5 files
(wind vars, wave vars) for buoy BD08 (17.833N, 89.216E), inner-joins them
on `valid_time`, sorts chronologically, asserts no duplicate timestamps
and no missing expected columns, saves `bay_of_bengal_merged.csv`.

## How to run
python .\src\data_pipeline_bay_of_bengal.py


## Key technical decisions
- Inner join on `valid_time` (not outer) — any timestamp missing from either
  file is dropped rather than filled, since both files came from the same
  ERA5 pull and should already align.
- Wave file's own `latitude`/`longitude` dropped before merge (its ERA5
  grid snap is 18.0N/89.0E vs. wind file's 17.75N/89.25E) — wind file's
  coordinates kept as the record of location.
- `blh` carried through the merge; will be dropped in Phase 3 (feature
  engineering), not used as a model input.

## Result — this run
File 1: 232,344 rows. File 2: 232,344 rows. Merged: 232,344 rows
(zero row loss). Date range: 2000-01-01 to 2026-07-03.

## Files created/updated
- `src/data_pipeline_bay_of_bengal.py`
- `bay_of_bengal/data/bay_of_bengal_merged.csv` (gitignored)
