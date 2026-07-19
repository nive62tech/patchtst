# Pacific Phase 5 — Forecasting Targets

## What was built
- `src/targets_pacific.py`: defines and verifies the forecasting target construction logic for Pacific — mirrors `src/targets.py` (Atlantic) exactly. No output data file; this is logic verification consumed by Phase 8's windowing script.

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\targets_pacific.py
```

## Key technical decisions
- Identical target definition to Atlantic — `swh`, `mwp`, raw `mwd`, t+6h to t+120h in 6-hour steps, 20 total steps, same index formula relative to the window start

## Result — this run
- Target index structure confirmed correct: `[6, 12, 18, ..., 120]`
- Last start index with a complete 20-step target: **232,223** — identical to Atlantic's, since Pacific has the same row count (232,344). The final **120 rows** of the dataset will be excluded once windowing runs
- Demo target shapes both `(20, 3)`, boundary check correctly returns `None`

## Files created/updated
- `src/targets_pacific.py`