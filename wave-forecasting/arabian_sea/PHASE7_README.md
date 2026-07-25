# Phase 7 — Normalization (Arabian Sea)

## What was built
`src/normalize_arabian_sea.py` — fits StandardScaler on train only (7
columns: u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd), applies to both splits,
saves scaler_arabian_sea.pkl.

## How to run

python .\src\normalize_arabian_sea.py


## Key technical decisions
- Same as Bay of Bengal - mwd scaled, mwp_class/valid_time untouched, scaler fit on train only.

## Result — this run
Train post-scaling: mean ~0.000000, std ~1.000003 across all 7 columns.
Test stats close (e.g. swh mean -0.026, std 0.950) - no leakage confirmed.

## Files created/updated
- `src/normalize_arabian_sea.py`
- `arabian_sea/data/arabian_sea_train_scaled.csv`, `arabian_sea_test_scaled.csv` (gitignored)
- `arabian_sea/scaler_arabian_sea.pkl` (gitignored)
