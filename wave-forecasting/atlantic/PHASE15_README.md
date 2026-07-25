# Phase 15 — Combined Results Dashboard (Atlantic)

## What was built
- `src/results_dashboard.py`: builds a single 6-panel summary figure combining the training loss curve, predicted-vs-actual timeseries for MWP and SWH, horizon-wise RMSE, the confusion matrix, and an MWP scatter plot with 1:1 line

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\results_dashboard.py
```
Output: `atlantic\results\results_dashboard.png`

## Key technical decisions
- **Panel 1 (loss curves) reuses the already-saved `training_curves.png` image directly** rather than fabricating loss values — Phase 11's training loop only saved the rendered plot, not per-epoch numbers as data, so this panel embeds the real existing image instead of inventing numbers
- All other 5 panels are built from a fresh inference pass over the real test set, with MWP/SWH values inverse-transformed to real units via the saved scaler
- MWP scatter plot subsamples to 5,000 points (fixed random seed) to avoid overplotting on ~69K test samples, while still representing the full distribution

## Result
- Dashboard generated successfully, all 6 panels populated with real data
- Harmless matplotlib `set_ticklabels()` warning during generation — cosmetic only, doesn't affect output

## Files created/updated
- `src/results_dashboard.py`
- `atlantic/results/results_dashboard.png`