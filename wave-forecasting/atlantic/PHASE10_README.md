# Phase 10 — Training Setup

## What was built
- `src/train_config.py`: defines the joint loss, optimizer, LR scheduler, early stopping logic, and the internal train/validation split — all verified with a real (but tiny) local smoke test on CPU. No full training happens here; that's Phase 11 on Colab.

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\train_config.py
```
No file output — this is configuration definition + verification.

## Key technical decisions
- **Class weights computed from the fit portion only** (not the internal validation slice, not test) — even though the data is balanced (confirmed in Phase 4/8), the weighting logic is implemented properly rather than skipped, since the spec calls for it generically
- **Internal validation slice is the last 10% of train, chronologically** — never shuffled, consistent with the project's time-series discipline throughout. It's still "train" conceptually; it's just excluded from weight updates and used only for early stopping and per-epoch monitoring
- **Warmup + cosine schedule** implemented via `SequentialLR` chaining a `LinearLR` warmup (5 epochs) into a `CosineAnnealingLR` decay — PyTorch has no single built-in scheduler for this combination
- **Fixed a `ModuleNotFoundError`** during this phase: `patchtst_model.py` lives at the project root but `train_config.py` lives in `src/`, and Python's module search path is based on the invoked script's location, not the shell's current directory. Fixed by explicitly inserting the project root into `sys.path` at the top of the script — makes it robust to being run from anywhere

## Result — this run
- Internal split: **146,268 fit / 16,252 validation** samples (10% of the 162,520 train windows)
- Class weights: **Low 1.0258, Moderate 0.9905, High 0.9705, Very High 1.0150** — all within ~3% of 1.0, confirming the balanced dataset needs essentially no reweighting
- LR schedule confirmed correct shape: ramps from near-zero to `0.0001` across the 5-epoch warmup, then begins cosine decay from epoch 6 onward
- Smoke test on real data: finite, non-NaN losses (total ≈0.91–1.20, class ≈1.35–1.94, forecast ≈0.45–0.47 across two runs — some variance expected since weights aren't seeded)
- Early stopping logic verified: triggers exactly at step 12 in the test sequence, matching the designed patience=10 behavior

## Files created/updated
- `src/train_config.py`