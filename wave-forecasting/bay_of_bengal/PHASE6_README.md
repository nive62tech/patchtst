# Phase 6 — Train/Test Split (Bay of Bengal)

## What was built
`src/split_bay_of_bengal.py` — chronological 70/30 split (no shuffling),
saves bay_of_bengal_train.csv / bay_of_bengal_test.csv.

## How to run

python .\src\split_bay_of_bengal.py


## Key technical decisions
- split_idx = int(232344 * 0.70) = 162640, boundary date 2018-07-21 16:00:00.

## Result — this run
Train: 162,640 rows (2000-01-01 to 2018-07-21). Test: 69,704 rows
(2018-07-21 to 2026-07-03). No overlap, exact 1h gap. All 4 mwp_class labels
present in both splits (train 24.6-25.3%, test 24.4-26.0% — mild natural
drift, expected and fine).

## Files created/updated
- `src/split_bay_of_bengal.py`
- `bay_of_bengal/data/bay_of_bengal_train.csv`, `bay_of_bengal_test.csv` (gitignored)
