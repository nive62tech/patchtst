# Phase 6 — Train/Test Split (Arabian Sea)

## What was built
`src/split_arabian_sea.py` — chronological 70/30 split (no shuffling),
saves arabian_sea_train.csv / arabian_sea_test.csv.

## How to run

python .\src\split_arabian_sea.py


## Key technical decisions
- split_idx = int(232344 * 0.70) = 162640, boundary date 2018-07-21 16:00:00.

## Result — this run
Train: 162,640 rows (2000-01-01 to 2018-07-21). Test: 69,704 rows
(2018-07-21 to 2026-07-03). No overlap, exact 1h gap. All 4 mwp_class labels
present in both splits (train 24.2-25.5%, test 23.8-26.8% — mild natural
drift, expected and fine).

## Files created/updated
- `src/split_arabian_sea.py`
- `arabian_sea/data/arabian_sea_train.csv`, `arabian_sea_test.csv` (gitignored)
