# Phase 6 — Train/Test Split (70/30 by Date)

## What was built
- `src/split.py`: loads `atlantic_features.csv`, computes `split_idx = int(len(df) * 0.70)`, splits strictly chronologically (no shuffling), verifies no date overlap and a clean 1-hour boundary gap, checks `mwp_class` coverage in both splits, saves `atlantic_train.csv` and `atlantic_test.csv`

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\split.py
```
Output: `atlantic\data\atlantic_train.csv`, `atlantic\data\atlantic_test.csv`

## Key technical decisions
- Split is **strictly chronological by index**, not by shuffled sampling — `iloc[:split_idx]` / `iloc[split_idx:]` on data already sorted by `valid_time`. This is the only split method that makes sense for time series; anything else leaks future information into training
- Boundary sanity checks built into the script itself (no date overlap, exactly a 1-hour gap between train's last timestamp and test's first) rather than trusting the split math implicitly
- Class coverage explicitly checked and printed for both splits — confirmed all four `mwp_class` categories present in each, even though the confirmed bin edges were computed on the *full* dataset, not per-split

## Result — this run
- `split_idx = 162640` → **70.00% train / 30.00% test**, exact
- **Train:** 162,640 rows, 2000-01-01 00:00 to 2018-07-21 15:00
- **Test:** 69,704 rows, 2018-07-21 16:00 to 2026-07-03 23:00
- Boundary gap: exactly 1 hour, no overlap
- All four `mwp_class` categories present in both splits — train ranges 23.91%–25.63% per class, test ranges 23.53%–27.53% per class. Some natural drift between the two periods (test skews slightly toward "Low"), but no class is anywhere near absent
- Row count sum verified: 162,640 + 69,704 = 232,344, matching the original dataset exactly — no rows lost or duplicated

## Files created/updated
- `src/split.py`
- `atlantic/data/atlantic_train.csv`, `atlantic/data/atlantic_test.csv` (gitignored, not committed)