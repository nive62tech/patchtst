# Pacific Phase 15 — Combined Results Dashboard

## What was built
- `src/results_dashboard_pacific.py`: same 6-panel dashboard as Atlantic's, using Pacific's checkpoint, scaler, and test data — mirrors `src/results_dashboard.py` exactly

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\results_dashboard_pacific.py
```
Output: `pacific\results\results_dashboard.png`

## Key technical decisions
- Same design as Atlantic's Phase 15 — Panel 1 reuses Pacific's already-saved `training_curves.png`, other 5 panels from fresh inference in real units

## Result
- Dashboard generated successfully
- Caught and fixed a cosmetic bug: the console header printed "PHASE 15 (ATLANTIC)" instead of "(PACIFIC)" due to an incomplete find-and-replace when adapting the script from Atlantic's version — text-only, did not affect the checkpoint loaded (correctly epoch 16, Pacific's), the scaler used (`scaler_pacific.pkl`), or the output file (`pacific/results/results_dashboard.png`)

## Files created/updated
- `src/results_dashboard_pacific.py`
- `pacific/results/results_dashboard.png`