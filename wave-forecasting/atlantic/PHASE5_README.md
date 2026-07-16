# Phase 5 — Forecasting Targets

## What was built
- `src/targets.py`: defines and verifies the forecasting target construction logic — given a window start index `i`, returns the 20 future values (t+6h through t+120h, 6-hour steps) for `swh`, `mwp`, `mwd`. No output data file is produced in this phase; this is a reusable module Phase 8 imports to build the actual `y_forecast` arrays separately for train and test.

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\targets.py
```
No file output — this run verifies the logic against `atlantic_features.csv`.

## Key technical decisions
- Forecast targets are **swh, mwp, raw mwd** — not `sin_mwd`/`cos_mwd`, since forecasting error on an angle-in-degrees is more directly interpretable than on its circular encoding; `mdts`/`mdww` were never targets even in the original 10-channel design and don't exist in this project's dataset at all
- Target indices are relative to the same `start_idx` as Phase 8's 72-hour input window: `[i+6, i+12, ..., i+120]` — this is defined now so Phase 8 has a single, tested source of truth rather than reimplementing the indexing inline
- Included an explicit boundary check (`build_single_target` returns `None` past the last valid start index) so Phase 8's windowing loop can safely skip incomplete samples at the tail of the dataset without silently producing malformed arrays

## Result — this run
- Target index structure confirmed correct: `[6, 12, 18, ..., 120]` for `start_idx=0`
- Last start index with a complete 20-step target: **232,223** — the final **120 rows** of the dataset can't form a full forecast target and will be excluded once windowing runs in Phase 8
- Demo target shapes both `(20, 3)` as expected, boundary check correctly returns `None` one step past the last valid index

## Files created/updated
- `src/targets.py`