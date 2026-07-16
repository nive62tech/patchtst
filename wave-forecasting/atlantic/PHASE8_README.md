# Phase 8 — Sequence Windowing

## What was built
- `src/windowing.py`: builds 72-hour input windows and matching classification/forecasting targets, separately within `atlantic_train_scaled.csv` and `atlantic_test_scaled.csv` (never spanning the train/test boundary), and saves the resulting arrays as `.npy` files

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\windowing.py
```
Output: 6 `.npy` files under `atlantic\data\windows\`

## Key technical decisions
- Windowing formula reuses Phase 5's exact index logic (`i+6` through `i+120`, 6-hour steps) reimplemented inline for self-containment, rather than importing `targets.py` as a module — keeps `windowing.py` runnable standalone
- The **forecast target** (last index `i+120`) is more restrictive than the **classification target** (`i+72`), so the forecast horizon determines how many samples are usable at the tail of each split — 120 rows lost at the end of train, 120 at the end of test
- `mwp_class` string labels converted to integers (`Low=0, Moderate=1, High=2, Very High=3`) here, since PyTorch's `CrossEntropyLoss` expects integer class indices, not strings
- Input features (`X`) and forecast targets (`y_forecast`) both come from the *scaled* CSVs from Phase 7 — the forecasting head trains and evaluates in scaled space, with the saved scaler available to invert predictions later

## Result — this run
- **Train:** 162,520 windows — `X_train (162520, 72, 6)`, `y_class_train (162520,)`, `y_forecast_train (162520, 20, 3)`
- **Test:** 69,584 windows — `X_test (69584, 72, 6)`, `y_class_test (69584,)`, `y_forecast_test (69584, 20, 3)`
- Sample counts consistent across all three arrays within each split (verified via assertion)
- Class balance held up close to Phase 6's split-level numbers (train ~23.9–25.6% per class, test ~23.6–27.5% per class) — windowing only trims 120 rows off each tail, so no meaningful distribution shift
- Correct dtypes confirmed: `float32` for `X`/`y_forecast`, `int64` for `y_class`

## Files created/updated
- `src/windowing.py`
- `atlantic/data/windows/X_train.npy`, `y_class_train.npy`, `y_forecast_train.npy`, `X_test.npy`, `y_class_test.npy`, `y_forecast_test.npy` (all gitignored, not committed)