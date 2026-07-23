# Pacific Phase 8 — Sequence Windowing

## What was built
- `src/windowing_pacific.py`: builds 72-hour input windows and matching classification/forecasting targets, separately within `pacific_train_scaled.csv` and `pacific_test_scaled.csv`, saves `.npy` files — mirrors `src/windowing.py` (Atlantic) exactly

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\windowing_pacific.py
```
Output: 6 `.npy` files under `pacific\data\windows\`

## Key technical decisions
- Identical windowing logic to Atlantic — same index formulas for classification target (`i+72`) and forecast target (`i+6` to `i+120`), same `mwp_class` integer mapping (`Low=0, Moderate=1, High=2, Very High=3`)

## Result — this run
- **Train:** 162,520 windows — `X_train (162520, 72, 6)`, `y_class_train (162520,)`, `y_forecast_train (162520, 20, 3)`
- **Test:** 69,584 windows — `X_test (69584, 72, 6)`, `y_class_test (69584,)`, `y_forecast_test (69584, 20, 3)`
- Identical shapes to Atlantic's, since Pacific has the same row counts throughout
- Class balance close to Phase 6's split-level numbers (train ~24.5–25.6%, test ~23.6–26.2% per class)
- Sample-count consistency verified across all arrays in both splits
- Correct dtypes confirmed: `float32` for `X`/`y_forecast`, `int64` for `y_class`

## Files created/updated
- `src/windowing_pacific.py`
- `pacific/data/windows/X_train.npy`, `y_class_train.npy`, `y_forecast_train.npy`, `X_test.npy`, `y_class_test.npy`, `y_forecast_test.npy` (all gitignored, not committed)