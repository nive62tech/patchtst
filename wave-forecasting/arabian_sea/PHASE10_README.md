# Phase 10 — Training Setup (Arabian Sea)

## What was built
`src/train_config_arabian_sea.py` — same JointLoss, class-weight, validation
split, optimizer/scheduler, and EarlyStopping setup as Bay of Bengal. CPU
smoke test only.

## How to run

python .\src\train_config_arabian_sea.py


## Key technical decisions
- Same as Bay of Bengal: 90/10 chronological internal split, AdamW lr=1e-4,
  5-epoch warmup -> cosine, batch 64, patience=10.

## Result — this run
Class weights near 1.0 (0.978-1.044) - balanced. LR schedule matches Bay of
Bengal exactly (shared logic, same epoch counts). Smoke test: total_loss=1.1740
(finite, no NaNs). Early stopping demo fired correctly at step 12.

## Files created/updated
- `src/train_config_arabian_sea.py`
