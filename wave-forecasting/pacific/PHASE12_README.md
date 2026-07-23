# Pacific Phase 12 — Model Evaluation on Test Set

## What was built
- `src/evaluate_pacific.py`: loads the best Pacific checkpoint, runs inference over the full held-out Pacific test set, computes classification/forecasting metrics, saves a confusion matrix, horizon error curve, and metrics report — mirrors `src/evaluate.py` (Atlantic) exactly

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\evaluate_pacific.py
```
Output: `pacific\results\confusion_matrix.png`, `horizon_error_curve.png`, `metrics_report.json`

## Key technical decisions
- Identical evaluation methodology to Atlantic — same reference-date reconstruction, same per-horizon breakdown, same year-by-year drift check

## Result — this run
- **Classification: 96.87% accuracy, 96.87% weighted F1, 96.88% macro F1** — all four classes between 0.96–0.98 F1, even slightly ahead of Atlantic's test performance (96.39%/96.38%)
- **Forecasting holds up slightly better at long horizons than Atlantic**: R² starts at 0.95 (t+6h), stays above 0.94 through t+72h, and only falls to 0.35 by t+120h (vs. Atlantic's 0.27) — Pacific's wave patterns may be somewhat more persistent/predictable at longer lead times
- **No concept drift**: year-by-year accuracy stays in a 95.8%–97.6% band across the full 2018–2026 test period
- Overall forecast metrics: MAE 0.2278, RMSE 0.4299, R² 0.8037 (scaled units)

## Files created/updated
- `src/evaluate_pacific.py`
- `pacific/results/confusion_matrix.png`
- `pacific/results/horizon_error_curve.png`
- `pacific/results/metrics_report.json`