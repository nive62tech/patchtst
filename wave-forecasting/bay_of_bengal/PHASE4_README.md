# Phase 4 — Label Definition (Bay of Bengal)

## What was built
`src/labels_bay_of_bengal.py` — computes quartile-based mwp_class bin edges
from this sea''s own mwp distribution, applies via pd.cut(), adds mwp_class
column back into bay_of_bengal_features.csv.

## How to run

python .\src\labels_bay_of_bengal.py


## Key technical decisions
- Bin edges computed fresh from Bay of Bengal''s own data (not reused from
  any other sea) — now CONFIRMED and hardcoded going forward:
  `[4.509097, 7.7588763, 8.6737255, 9.79438075, 15.519651]`
- Label order: Low, Moderate, High, Very High (0-3).

## Result — this run
mwp range [4.51, 15.52]s, mean 8.81s. All 4 classes exactly 25.00% (quartile
split by construction). 0 unassigned rows.

## Files created/updated
- `src/labels_bay_of_bengal.py`
- `bay_of_bengal/data/bay_of_bengal_features.csv` (gitignored, mwp_class column added)
