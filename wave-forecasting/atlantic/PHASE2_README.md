# Phase 2 — Data Cleaning

## What was built
- `src/data_cleaning.py`: loads `atlantic_merged.csv`, checks for duplicate timestamps, builds the expected full hourly range and detects missing timestamps, interpolates short gaps, drops rows for unfillable gaps, and saves `atlantic_clean.csv`

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\data_cleaning.py
```
Output: `atlantic\data\atlantic_clean.csv`

## Key technical decisions
- Duplicate timestamps removed via `drop_duplicates(subset="valid_time", keep="first")` before any gap analysis
- Missing timestamps found by reindexing onto a full expected hourly `date_range` — any timestamp absent from the merged data becomes a NaN row rather than silently vanishing
- Gaps of ≤3 consecutive hours are linear-interpolated (`limit=3, limit_area="inside"`); nothing is extrapolated at the start/end edges of the series
- Any row still NaN after interpolation (a gap >3 hours) is dropped entirely, not imputed

## Result — this run
- **0 duplicate timestamps** found
- **0 missing timestamps** — the full expected hourly range (232,344 timestamps, 2000-01-01 00:00 to 2026-07-03 23:00) was already completely present in the merged data
- **0 NaNs** in any column, before or after interpolation
- **0 rows dropped**
- Final dataset: **232,344 rows**, identical row count to `atlantic_merged.csv` — the raw ERA5 time-series export had no gaps or duplicates for this point/date range, so cleaning was a verification pass rather than an active fix

## Files created/updated
- `src/data_cleaning.py`
- `atlantic/data/atlantic_clean.csv` (gitignored, not committed)