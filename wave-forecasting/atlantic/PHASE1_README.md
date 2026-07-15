# Phase 1 — Data Loading & Merging

## What was built
- `src/data_pipeline.py`: loads the two raw ERA5 CSVs, merges them, cleans up columns, saves the merged dataset
- Loads `atlantic_raw_file1.csv` (u10, v10, u100, v100, fg10) and `atlantic_raw_file2.csv` (mwd, mwp, swh)
- Merges on `valid_time` via inner join, sorted ascending
- Drops `u100, v100, fg10` (and any lat/long duplicates from the merge) — not part of the model's input set
- Saves final 6-column dataset: `valid_time, u10, v10, swh, mwp, mwd`

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\data_pipeline.py
```
Output: `atlantic\data\atlantic_merged.csv`

## Key technical decisions
- **6 input channels, not the originally planned 10** — `mdts`/`mdww` aren't available in the ERA5 hourly time-series product this project uses, so they were dropped from the plan entirely (no re-download attempted)
- **Inner join on `valid_time`** rather than outer — any timestamp missing from either raw file is simply excluded at this stage; gap-filling logic belongs in Phase 2, not here
- Column-name collisions (`latitude`/`longitude` present in both raw files) are handled safely — the final column selection step only keeps the explicitly defined 6 columns, so merge-suffixed duplicates never reach the saved file

## Result
- 232,344 rows, 2000-01-01 00:00 to 2026-07-03 23:00, hourly, no row loss from the merge (row count matches both raw files)

## Files created/updated
- `src/data_pipeline.py`
- `atlantic/data/atlantic_merged.csv` (gitignored, not committed)