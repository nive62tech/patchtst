# Phase 8 — Sequence Windowing (Arabian Sea)

## What was built
`src/windowing_arabian_sea.py` — builds 72h input windows and matching
class/forecast targets separately per split (never spanning the boundary),
saves 6 .npy files to arabian_sea/data/windows/.

## How to run

python .\src\windowing_arabian_sea.py


## Key technical decisions
- Same windowing formula as Bay of Bengal/Atlantic/Pacific.

## Result — this run
X_train: (162520, 72, 6), y_forecast_train: (162520, 20, 3).
X_test: (69584, 72, 6), y_forecast_test: (69584, 20, 3).
Class distribution stays balanced (24.2-25.5% train, 23.7-26.8% test).
0 windows skipped for missing class.

## Files created/updated
- `src/windowing_arabian_sea.py`
- `arabian_sea/data/windows/X_train.npy, y_class_train.npy, y_forecast_train.npy`
- `arabian_sea/data/windows/X_test.npy, y_class_test.npy, y_forecast_test.npy` (all gitignored)
