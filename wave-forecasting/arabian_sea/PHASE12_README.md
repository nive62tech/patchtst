# Phase 12 — Model Evaluation on Test Set (Arabian Sea)

## What was built
`src/evaluate_arabian_sea.py` — loads best checkpoint (epoch 22), runs
inference on the full 69,584-sample test set, computes classification
metrics, per-horizon forecasting metrics, and year-by-year breakdown.
Saves confusion_matrix.png, horizon_error_curve.png, metrics_report.json.

## How to run

python .\src\evaluate_arabian_sea.py


## Key technical decisions
- Same metric set and per-horizon breakdown as Atlantic/Pacific/Bay of Bengal.

## Result — this run
Test accuracy: 0.9694 (weighted F1: 0.9694, macro F1: 0.9697). All 4 classes
balanced (F1 0.960-0.980). Forecast R2: 0.9619 at t+6h, degrading to 0.5183
at t+120h. Overall forecast MAE=0.1809, RMSE=0.3790, R2=0.8484. Year-by-year
accuracy stable (0.962-0.976), no meaningful drift across 2018-2026.

## Files created/updated
- `src/evaluate_arabian_sea.py`
- `arabian_sea/results/confusion_matrix.png`
- `arabian_sea/results/horizon_error_curve.png`
- `arabian_sea/results/metrics_report.json`

## Extended Validation Metrics (real units)

`src/validation_metrics_arabian_sea.py` - inverse-transforms predictions
back to real units and computes correlation, scatter index, bias, and RMSE
per forecast variable.

| Variable | r | SI | Bias | RMSE |
|---|---|---|---|---|
| swh | 0.9829 | 0.1027 | -0.0019 | 0.1673 m |
| mwp | 0.9259 | 0.0647 | 0.0075 | 0.5239 s |
| mwd | 0.8584 | 0.2356 | 0.4992 | 45.0370 deg |

Files: `arabian_sea/results/validation_metrics.json`,
`arabian_sea/results/validation_metrics_table.md`
