# Pacific Phase 10 — Training Setup

## What was built
- `src/train_config_pacific.py`: defines the joint loss, optimizer, LR scheduler, early stopping logic, and internal train/validation split for Pacific — verified with a real (tiny) local smoke test on CPU. Mirrors `src/train_config.py` (Atlantic) exactly.

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\train_config_pacific.py
```
No file output — configuration definition + verification only.

## Key technical decisions
- Same design as Atlantic's Phase 10 throughout — class weights computed on the fit portion only, chronological last-10% internal validation slice, warmup+cosine LR schedule via `SequentialLR`

## Result — this run
- Internal split: **146,268 fit / 16,252 validation** samples (10% of the 162,520 train windows)
- Class weights: **Low 0.9862, Moderate 1.0061, High 1.0154, Very High 0.9929** — even closer to 1.0 than Atlantic's, confirming Pacific's classes are essentially balanced
- LR schedule confirmed correct shape: ramps to `0.0001` by epoch 5, cosine decay begins epoch 6
- Smoke test on real Pacific data: finite losses (total 1.0970, class 1.4676, forecast 0.7265)
- Early stopping logic verified: triggers exactly at step 12, matching patience=10

## Files created/updated
- `src/train_config_pacific.py`