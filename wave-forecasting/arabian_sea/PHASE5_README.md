# Phase 5 — Forecasting Targets (Arabian Sea)

## What was built
`src/targets_arabian_sea.py` — verification module defining/checking the
target-index formula (t+6h...t+120h, 20 steps) for swh/mwp/mwd. Reproduced
inline in Phase 8''s windowing script; produces no output file itself.

## How to run

python .\src\targets_arabian_sea.py


## Key technical decisions
- Same formula as Bay of Bengal/Atlantic/Pacific.

## Result — this run
Last valid start_idx: 232223 (120 tail rows unusable). Demo target shapes
both (20, 3) as expected. One-past-last-valid correctly returns None.

## Files created/updated
- `src/targets_arabian_sea.py` (no data file produced)
