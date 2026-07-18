# Phase 12 — Model Evaluation on Test Set

## What was built
- `src/evaluate.py`: loads the best checkpoint, runs a single inference pass over the full held-out test set (never touched during training or the internal validation split), computes classification and forecasting metrics, and saves a confusion matrix, a horizon error curve, and a full metrics report

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\evaluate.py
```
Output: `atlantic\results\confusion_matrix.png`, `atlantic\results\horizon_error_curve.png`, `atlantic\results\metrics_report.json`

## Key technical decisions
- **Reference dates reconstructed from `atlantic_test_scaled.csv`**, not stored during windowing — the `.npy` arrays from Phase 8 are pure numeric arrays with no timestamps, so the year-by-year breakdown re-derives each window's date by reapplying the exact same start-index formula used in `windowing.py`, then asserting the reconstructed count matches the window count exactly before trusting it
- **Inference-only, CPU, larger batch size (256)** — no gradients held, so this runs comfortably on the local machine without needing Colab, unlike training
- **Per-horizon metrics computed separately for all 20 forecast steps**, not just an aggregate — this is what actually reveals the model's real behavior (see results below), which an overall RMSE alone would have hidden

## Result — this run
- **Classification: 96.39% accuracy, 96.38% weighted F1, 96.34% macro F1** — consistent across all four classes (F1 ranges 0.94–0.98), and *slightly above* Phase 11's validation accuracy (96.19%), a good sign the model generalizes rather than overfitting to the validation slice
- **Forecasting degrades predictably with horizon**: R² starts at 0.94 (t+6h) and falls to 0.27 by t+120h — RMSE roughly triples from t+6h (0.23) to t+120h (0.85). This is the expected, physically sensible pattern for multi-step forecasting, not a bug — near-term wave conditions are far more predictable than 5-day-ahead ones
- **No concept drift across the test period**: year-by-year accuracy stays in a tight 95.6%–97.1% band from 2018 through 2026, with no degrading trend over time
- Overall forecast metrics: MAE 0.2484, RMSE 0.4678, R² 0.7754 (all in scaled units, per Phase 7's StandardScaler)

## Files created/updated
- `src/evaluate.py`
- `atlantic/results/confusion_matrix.png`
- `atlantic/results/horizon_error_curve.png`
- `atlantic/results/metrics_report.json`