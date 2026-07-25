# Phase 4 — Label Definition (Arabian Sea)

## What was built
`src/labels_arabian_sea.py` — computes quartile-based mwp_class bin edges
from this sea''s own mwp distribution, applies via pd.cut(), adds mwp_class
column back into arabian_sea_features.csv.

## How to run

python .\src\labels_arabian_sea.py


## Key technical decisions
- Bin edges computed fresh from Arabian Sea''s own data — now CONFIRMED and
  hardcoded going forward: `[4.674424, 7.0532007, 8.16263, 9.05299975, 14.895967]`
- Label order: Low, Moderate, High, Very High (0-3).

## Result — this run
mwp range [4.67, 14.90]s, mean 8.11s. All 4 classes exactly 25.00% (quartile
split by construction). 0 unassigned rows.

## Files created/updated
- `src/labels_arabian_sea.py`
- `arabian_sea/data/arabian_sea_features.csv` (gitignored, mwp_class column added)
