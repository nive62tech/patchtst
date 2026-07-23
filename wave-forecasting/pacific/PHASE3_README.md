# Pacific Phase 3 — Feature Engineering (Direction Encoding)

## What was built
- `src/feature_engineering_pacific.py`: loads `pacific_clean.csv`, applies sin/cos circular encoding to `mwd`, keeps raw `mwd` for later use as a Phase 5 forecasting target, saves `pacific_features.csv` — mirrors `src/feature_engineering.py` (Atlantic) exactly, renamed throughout

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\feature_engineering_pacific.py
```
Output: `pacific\data\pacific_features.csv`

## Key technical decisions
- Identical logic to Atlantic's Phase 3 — same Formula 3 encoding, same unit-circle sanity check, same column layout (6 model inputs + raw `mwd` carried through)

## Result — this run
- `sin_mwd`/`cos_mwd` both fully within `[-1.0000, 1.0000]`
- Unit-circle check: max deviation `2.22e-16` — same floating-point-noise precision as Atlantic's run
- Final dataset: **232,344 rows, 8 columns** (`valid_time, u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd`) — no rows lost
- **Final 6 model input channels confirmed:** `u10, v10, swh, mwp, sin_mwd, cos_mwd`

## Files created/updated
- `src/feature_engineering_pacific.py`
- `pacific/data/pacific_features.csv` (gitignored, not committed)s