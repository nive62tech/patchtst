# Phase 8 — Sequence Windowing (Bay of Bengal)

## What was built
`src/windowing_bay_of_bengal.py` — builds 72h input windows and matching
class/forecast targets separately per split (never spanning the boundary),
saves 6 .npy files to bay_of_bengal/data/windows/.

## How to run

python .\src\windowing_bay_of_bengal.py


## Key technical decisions
- X[i] = features[i:i+72] -> [72,6]; y_class[i] = mwp_class[i+72];
  y_forecast[i] = [swh,mwp,mwd][i+6:i+126:6] -> [20,3].
- Train and test windowed independently, no cross-boundary leakage.

## Result — this run
X_train: (162520, 72, 6), y_forecast_train: (162520, 20, 3).
X_test: (69584, 72, 6), y_forecast_test: (69584, 20, 3).
Class distribution stays balanced in both splits (24.5-25.3% train,
24.4-26.1% test). 0 windows skipped for missing class.

## Files created/updated
- `src/windowing_bay_of_bengal.py`
- `bay_of_bengal/data/windows/X_train.npy, y_class_train.npy, y_forecast_train.npy`
- `bay_of_bengal/data/windows/X_test.npy, y_class_test.npy, y_forecast_test.npy` (all gitignored)
