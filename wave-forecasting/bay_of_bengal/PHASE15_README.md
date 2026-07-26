# Phase 15 — Combined Results Dashboard (Bay of Bengal)

## What was built
`src/results_dashboard_bay_of_bengal.py` — 6-panel summary figure: training
curves (embedded from Phase 11), MWP/SWH predicted-vs-actual (t+6h, first
200 samples), horizon-wise RMSE, confusion matrix, MWP scatter (5000
subsampled points, 1:1 line, RMSE-annotated).

## How to run

python .\src\results_dashboard_bay_of_bengal.py


## Key technical decisions
- Reuses the real training_curves.png image rather than fabricating
  per-epoch loss data (per project convention).
- All predicted-vs-actual panels in real units (inverse-transformed).

## Result — this run
Dashboard saved successfully, all 6 panels populated.

## Files created/updated
- `src/results_dashboard_bay_of_bengal.py`
- `bay_of_bengal/results/results_dashboard.png`
