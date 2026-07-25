# Phase 7 — Normalization (Bay of Bengal)

## What was built
`src/normalize_bay_of_bengal.py` — fits StandardScaler on train only (7
columns: u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd), applies to both splits,
saves scaler_bay_of_bengal.pkl.

## How to run

python .\src\normalize_bay_of_bengal.py


## Key technical decisions
- mwd included in scaling (doubles as forecast target).
- mwp_class and valid_time never scaled.
- Scaler fit strictly on train (162,640 rows) - test transform uses train statistics.

## Result — this run
Train post-scaling: mean ~0.000000, std ~1.000003 across all 7 columns
(numerically exact). Test stats close but not identical (e.g. swh mean
-0.059, std 0.997) - expected, confirms no leakage.

## Files created/updated
- `src/normalize_bay_of_bengal.py`
- `bay_of_bengal/data/bay_of_bengal_train_scaled.csv`, `bay_of_bengal_test_scaled.csv` (gitignored)
- `bay_of_bengal/scaler_bay_of_bengal.pkl` (gitignored)
