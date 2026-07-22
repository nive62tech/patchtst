# Pacific Phase 6 — Train/Test Split (70/30 by Date)

## What was built
- `src/split_pacific.py`: loads `pacific_features.csv`, computes `split_idx = int(len(df) * 0.70)`, splits chronologically, verifies boundary and class coverage, saves `pacific_train.csv` / `pacific_test.csv` — mirrors `src/split.py` (Atlantic) exactly

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\split_pacific.py
```
Output: `pacific\data\pacific_train.csv`, `pacific\data\pacific_test.csv`

## Key technical decisions
- Same 70/30 chronological split as Atlantic — kept consistent across oceans per the original spec's split strategy

## Result — this run
- `split_idx = 162640` → **70.00% train / 30.00% test** — identical index to Atlantic's split, since Pacific has the same row count
- **Train:** 162,640 rows, 2000-01-01 00:00 to 2018-07-21 15:00
- **Test:** 69,704 rows, 2018-07-21 16:00 to 2026-07-03 23:00
- Boundary gap exactly 1 hour, no overlap
- All four `mwp_class` categories present in both splits — train ranges 24.47%–25.60% per class, test ranges 23.59%–26.25% per class, similar natural drift pattern to Atlantic's split
- Row count sum verified: 162,640 + 69,704 = 232,344

## Files created/updated
- `src/split_pacific.py`
- `pacific/data/pacific_train.csv`, `pacific/data/pacific_test.csv` (gitignored, not committed)