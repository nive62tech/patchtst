# Phase 3 — Feature Engineering (Direction Encoding)

## What was built
- `src/feature_engineering.py`: loads `atlantic_clean.csv`, applies sin/cos circular encoding to `mwd` per Formula 3, keeps raw `mwd` in the file for later use as a Phase 5 forecasting target, and saves `atlantic_features.csv`

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\feature_engineering.py
```
Output: `atlantic\data\atlantic_features.csv`

## Key technical decisions
- Only `mwd` gets sin/cos encoded — `mdts`/`mdww` were dropped earlier in the project (not available in the ERA5 time-series product), so this project only has one circular direction variable instead of the originally planned three
- Raw `mwd` (in degrees) is kept in the output file alongside the encoded columns — it's not a model input, but Phase 5 needs it as a raw forecasting target
- Included a unit-circle sanity check (`sin² + cos² ≈ 1`) in the script output to catch encoding bugs immediately rather than downstream

## Result — this run
- `sin_mwd` range: `[-1.0000, 1.0000]`, `cos_mwd` range: `[-1.0000, 1.0000]` — both fully within expected bounds
- Unit-circle check: max deviation `2.22e-16` — floating-point noise only, encoding is correct
- Final dataset: **232,344 rows, 8 columns** (`valid_time, u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd`) — same row count as `atlantic_clean.csv`, confirming no rows were lost
- **Final 6 model input channels confirmed:** `u10, v10, swh, mwp, sin_mwd, cos_mwd`

## Files created/updated
- `src/feature_engineering.py`
- `atlantic/data/atlantic_features.csv` (gitignored, not committed)