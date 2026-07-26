# Phase 12 — Model Evaluation on Test Set (Bay of Bengal)

## What was built
`src/evaluate_bay_of_bengal.py` — loads best checkpoint (epoch 16), runs
inference on the full 69,584-sample test set, computes classification
metrics, per-horizon forecasting metrics, and year-by-year breakdown.
Saves confusion_matrix.png, horizon_error_curve.png, metrics_report.json.

## How to run

python .\src\evaluate_bay_of_bengal.py


## Key technical decisions
- Same metric set and per-horizon breakdown as Atlantic/Pacific.
- Year-by-year check used to screen for concept drift.

## Result — this run
Test accuracy: 0.9656 (weighted F1: 0.9656, macro F1: 0.9654). All 4 classes
balanced (F1 0.949-0.982). Forecast R2: 0.9419 at t+6h, degrading to 0.4435
at t+120h (expected for 5-day horizon). Overall forecast MAE=0.2091,
RMSE=0.4274, R2=0.8259. Year-by-year accuracy stable (0.958-0.972), no
meaningful drift across 2018-2026.

## Files created/updated
- `src/evaluate_bay_of_bengal.py`
- `bay_of_bengal/results/confusion_matrix.png`
- `bay_of_bengal/results/horizon_error_curve.png`
- `bay_of_bengal/results/metrics_report.json`

## Extended Validation Metrics (real units)

`src/validation_metrics_bay_of_bengal.py` - inverse-transforms predictions
back to real units and computes correlation, scatter index, bias, and RMSE
per forecast variable.

| Variable | r | SI | Bias | RMSE |
|---|---|---|---|---|
| swh | 0.9521 | 0.1477 | -0.0112 | 0.2233 m |
| mwp | 0.9047 | 0.0759 | -0.0020 | 0.6672 s |
| mwd | 0.8741 | 0.1362 | -0.0952 | 24.4528 deg |

Files: `bay_of_bengal/results/validation_metrics.json`,
`bay_of_bengal/results/validation_metrics_table.md`
