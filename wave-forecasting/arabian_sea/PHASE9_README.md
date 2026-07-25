# Phase 9 — Architecture (Bay of Bengal)

## What was built
Nothing new. `patchtst_model.py` at the repo root is ocean-agnostic and
reused as-is - same PatchTST class, same hyperparameters, imported directly
via `from patchtst_model import PatchTST` (see Phase 10''s
train_config_bay_of_bengal.py).

## How to run
N/A - no per-sea script. Verified already in Phase 10''s smoke test (model
instantiates, forward pass runs on real Bay of Bengal data).

## Key technical decisions
- No sea-specific architecture file. One shared model definition across
  Atlantic, Pacific, Bay of Bengal, Arabian Sea.

## Result — this run
N/A (confirmation only).

## Files created/updated
None - `patchtst_model.py` unchanged.
