# Pacific Phase 7 — Normalization

## What was built
- `src/normalize_pacific.py`: fits a `StandardScaler` on `pacific_train.csv` only, applies it to both splits, saves `pacific_train_scaled.csv`, `pacific_test_scaled.csv`, and the fitted scaler as `scaler_pacific.pkl` — mirrors `src/normalize.py` (Atlantic) exactly

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\normalize_pacific.py
```
Output: `pacific\data\pacific_train_scaled.csv`, `pacific\data\pacific_test_scaled.csv`, `pacific\scaler_pacific.pkl`

## Key technical decisions
- Same 7-column scaling decision as Atlantic — the 6 model inputs plus raw `mwd` (since `mwd` doubles as a forecasting target alongside `swh`/`mwp`)
- Own independent scaler — never shares fit statistics with Atlantic's scaler

## Result — this run
- Train post-scaling: **mean -0.000000, std 1.000003** across all 7 columns
- Test post-scaling: close but not exact (e.g. `swh` mean -0.110, `mwd` mean -0.039) — confirms the scaler wasn't re-fit on test
- Reloaded scaler's `mean_`/`scale_` match the fit-time values exactly
- Fitted stats (real units, from train): `mwp` mean ≈ 8.79s, std ≈ 1.64s — notably higher and more variable than Atlantic's (7.85s/1.25s), consistent with Pacific's generally rougher wave climate seen since Phase 4

## Files created/updated
- `src/normalize_pacific.py`
- `pacific/data/pacific_train_scaled.csv`, `pacific/data/pacific_test_scaled.csv` (gitignored)
- `pacific/scaler_pacific.pkl` (gitignored)