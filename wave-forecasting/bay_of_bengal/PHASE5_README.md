# Phase 5 — Forecasting Targets (Bay of Bengal)

## What was built
`src/targets_bay_of_bengal.py` — verification module defining/checking the
target-index formula (t+6h...t+120h, 20 steps) for swh/mwp/mwd. Reproduced
inline in Phase 8''s windowing script; produces no output file itself.

## How to run

python .\src\targets_bay_of_bengal.py


## Key technical decisions
- Target columns: swh, mwp, raw mwd (unencoded degrees).
- Same formula as Atlantic/Pacific: indices = i+6 to i+120 step 6.

## Result — this run
Last valid start_idx: 232223 (120 tail rows unusable). Demo target shapes
both (20, 3) as expected. One-past-last-valid correctly returns None.

## Files created/updated
- `src/targets_bay_of_bengal.py` (no data file produced)
