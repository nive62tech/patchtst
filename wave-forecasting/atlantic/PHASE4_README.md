# Phase 4 — Label Definition (Classification Target)

## What was built
- `src/labels.py`: loads `atlantic_features.csv`, sanity-checks `mwp` stats against the pre-confirmed Atlantic bin edges, applies those hardcoded edges via `pd.cut()` to create `mwp_class`, reports class distribution and imbalance ratio, saves the file back out with the new column

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\labels.py
```
Output: `atlantic\data\atlantic_features.csv` (in place, `mwp_class` column added)

## Key technical decisions
- Bin edges are **hardcoded, not recalculated** — `[4.615818, 6.937847, 7.637388, 8.518619, 13.928587]` → `Low/Moderate/High/Very High`. This is a hard requirement: the teammate's Mamba model must use these exact same edges for the Atlantic comparison to be valid
- `pd.cut(..., include_lowest=True)` ensures the minimum `mwp` value is captured in the "Low" bin rather than falling outside all bins
- Script includes an out-of-range check before applying `pd.cut()` — since these edges were computed on a specific Atlantic distribution, this run's actual `mwp` min/max needed verifying against them, not assumed to match

## Result — this run
- `mwp` (Tm01) range: `[4.615818, 13.928587]` — matches the bin edges' own min/max **exactly**, confirming these edges were computed on this same cleaned dataset
- **0 rows** out of range, **0 unassigned** `mwp_class` values
- Class distribution: **Low 58,086 (25.00%) / Moderate 58,086 (25.00%) / High 58,086 (25.00%) / Very High 58,086 (25.00%)** — a textbook-perfect quartile split
- Imbalance ratio: **1.00** — classes are perfectly balanced, so Phase 10's classification loss does not need class weighting for the Atlantic model

## Files created/updated
- `src/labels.py`
- `atlantic/data/atlantic_features.csv` (gitignored, not committed — now has `mwp_class` column)