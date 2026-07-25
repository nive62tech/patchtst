# Phase 10 — Training Setup (Bay of Bengal)

## What was built
`src/train_config_bay_of_bengal.py` — defines JointLoss (0.5*CE + 0.5*MSE),
class-weight computation, internal 90/10 chronological validation split,
AdamW + warmup/cosine scheduler, EarlyStopping. Runs a CPU smoke test (one
real optimizer step + early-stopping logic check) - does not train fully.

## How to run

python .\src\train_config_bay_of_bengal.py


## Key technical decisions
- Internal validation: last 10% of the 162,520 train windows (16,252),
  chronological, not shuffled.
- Class weights computed on the fit portion only (146,268 samples).
- AdamW lr=1e-4, 5-epoch linear warmup -> CosineAnnealingLR, batch 64,
  up to 100 epochs, early stopping patience=10.

## Result — this run
Class weights near 1.0 (0.987-1.023) - balanced. LR ramps 0->1e-4 over
5 warmup epochs then begins cosine decay. Smoke test: total_loss=3.0468
(one real optimizer step, finite, no NaNs). Early stopping demo fired
correctly at step 12.

## Files created/updated
- `src/train_config_bay_of_bengal.py`
