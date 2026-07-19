# Pacific Phase 1 — Data Loading & Merging

## What was built
- `src/data_pipeline_pacific.py`: loads the two raw Pacific ERA5 CSVs, merges them, cleans up columns, saves the merged dataset — mirrors `src/data_pipeline.py` (Atlantic) exactly, renamed throughout

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\data_pipeline_pacific.py
```
Output: `pacific\data\pacific_merged.csv`

## Key technical decisions
- Same logic as Atlantic's Phase 1 throughout — inner join on `valid_time`, sorted ascending, drops `u100/v100/fg10` plus merge-suffixed lat/long duplicates via the final column selection
- Separate, dedicated script (`data_pipeline_pacific.py`) rather than a shared parameterized script — simpler to reason about and matches the exact per-ocean workflow already used for Atlantic

## Result
- 232,344 rows — same row count and same date range (2000-01-01 to 2026-07-03) as Atlantic's merge
- Final 6 columns: `valid_time, u10, v10, swh, mwp, mwd`

## Files created/updated
- `src/data_pipeline_pacific.py`
- `pacific/data/pacific_merged.csv` (gitignored, not committed)