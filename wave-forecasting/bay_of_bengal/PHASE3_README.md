# Phase 3 — Feature Engineering (Bay of Bengal)

## What was built
`src/feature_engineering_bay_of_bengal.py` — applies Formula 3 circular
direction encoding to `mwd` (sin_mwd, cos_mwd), keeps raw `mwd` for later
use as a Phase 5 forecasting target, saves `bay_of_bengal_features.csv`
with final 8 columns: valid_time, u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd.

## How to run

python .\src\feature_engineering_bay_of_bengal.py


## Key technical decisions
- Model input channels fixed at 6: u10, v10, swh, mwp, sin_mwd, cos_mwd.
- mwd retained unscaled/unencoded alongside sin/cos, purely as forecast-target material.

## Result — this run
232,344 rows, 8 columns. sin_mwd/cos_mwd both range [-1.0, 1.0].
sin^2+cos^2 unit-circle check: max deviation 2.22e-16 (numerically exact).

## Files created/updated
- `src/feature_engineering_bay_of_bengal.py`
- `bay_of_bengal/data/bay_of_bengal_features.csv` (gitignored)
