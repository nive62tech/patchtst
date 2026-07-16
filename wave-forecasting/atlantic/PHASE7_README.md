# Phase 7 — Normalization

## What was built
- `src/normalize.py`: fits a `StandardScaler` on `atlantic_train.csv` only, applies the same fitted scaler to both `atlantic_train.csv` and `atlantic_test.csv`, verifies the post-scaling stats, and saves both scaled CSVs plus the fitted scaler as `scaler_atlantic.pkl`

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\normalize.py
```
Output: `atlantic\data\atlantic_train_scaled.csv`, `atlantic\data\atlantic_test_scaled.csv`, `atlantic\scaler_atlantic.pkl`

## Key technical decisions
- Scaler is **fit on train only**, then applied (never re-fit) to test — the standard rule to prevent test-set statistics leaking into training
- **7 columns scaled**: the 6 model input channels (`u10, v10, swh, mwp, sin_mwd, cos_mwd`) plus raw `mwd` — `mwd` was included because it doubles as a Phase 5 forecasting target alongside `swh`/`mwp`, so keeping all three targets in the same scaled space keeps the forecasting loss consistent; the saved scaler lets Phase 12 inverse-transform predictions back to real units
- `mwp_class` (the classification label) and `valid_time` are explicitly excluded from scaling
- Scaler persisted with `joblib` per the tech stack spec, and the script reload-verifies it loads back correctly before finishing

## Result — this run
- Train post-scaling: **mean 0.000000, std 1.000003** across all 7 columns — correct StandardScaler behavior
- Test post-scaling: means/stds close to but not exactly 0/1 (e.g. `swh` mean -0.085, `mwd` mean -0.021) — expected and correct, confirms the scaler wasn't re-fit on test
- Reloaded scaler's `mean_`/`scale_` arrays match the fit-time values exactly, confirming the `.pkl` file is valid
- Example fitted stats (real units, from train): `mwp` mean ≈ 7.85s, std ≈ 1.25s; `swh` mean ≈ 1.82m, std ≈ 0.77m

## Files created/updated
- `src/normalize.py`
- `atlantic/data/atlantic_train_scaled.csv`, `atlantic/data/atlantic_test_scaled.csv` (gitignored)
- `atlantic/scaler_atlantic.pkl` (gitignored — generated artifact)