# Pacific Phase 2 — Data Cleaning

## What was built
- `src/data_cleaning_pacific.py`: loads `pacific_merged.csv`, checks duplicate timestamps, detects missing timestamps against the full expected hourly range, interpolates short gaps, drops rows for unfillable gaps, saves `pacific_clean.csv` — mirrors `src/data_cleaning.py` (Atlantic) exactly, renamed throughout

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\data_cleaning_pacific.py
```
Output: `pacific\data\pacific_clean.csv`

## Key technical decisions
- Identical logic to Atlantic's Phase 2 — same duplicate-removal, reindex-based gap detection, ≤3-hour interpolation, and unfillable-gap drop rules

## Result — this run
- **0 duplicate timestamps**, **0 missing timestamps**, **0 NaNs**, **0 rows dropped**
- Final dataset: **232,344 rows**, identical to `pacific_merged.csv` — Pacific's raw download is just as complete/gap-free as Atlantic's was

## Files created/updated
- `src/data_cleaning_pacific.py`
- `pacific/data/pacific_clean.csv` (gitignored, not committed)