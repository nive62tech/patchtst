# Phase 3 — Feature Engineering (Arabian Sea)

## What was built
`src/feature_engineering_arabian_sea.py` — applies Formula 3 circular
direction encoding to `mwd` (sin_mwd, cos_mwd), keeps raw `mwd` for later
use as a Phase 5 forecasting target, saves `arabian_sea_features.csv`
with final 8 columns: valid_time, u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd.

## How to run

python .\src\feature_engineering_arabian_sea.py


## Key technical decisions
- Same as Bay of Bengal: 6 fixed model input channels, mwd retained separately.

## Result — this run
232,344 rows, 8 columns. sin_mwd/cos_mwd both range [-1.0, 1.0].
sin^2+cos^2 unit-circle check: max deviation 2.22e-16 (numerically exact).

## Files created/updated
- `src/feature_engineering_arabian_sea.py`
- `arabian_sea/data/arabian_sea_features.csv` (gitignored)
