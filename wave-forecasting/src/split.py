"""
Phase 6 - Train/Test Split - 70/30 by Date

Loads atlantic/data/atlantic_features.csv, computes a strict chronological
70/30 split (no shuffling - this is a time series, future data must never
leak into training), and saves atlantic_train.csv / atlantic_test.csv.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "atlantic" / "data"
INPUT_PATH = DATA_DIR / "atlantic_features.csv"
TRAIN_OUTPUT_PATH = DATA_DIR / "atlantic_train.csv"
TEST_OUTPUT_PATH = DATA_DIR / "atlantic_test.csv"

TRAIN_FRACTION = 0.70
LABELS = ["Low", "Moderate", "High", "Very High"]


def main():
    print("=" * 60)
    print("PHASE 6 - TRAIN/TEST SPLIT (70/30 BY DATE)")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Phase 5 first.")

    df = pd.read_csv(INPUT_PATH)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = df.sort_values("valid_time").reset_index(drop=True)
    print(f"[OK] Loaded {INPUT_PATH.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    n_rows = len(df)
    split_idx = int(n_rows * TRAIN_FRACTION)
    split_date = df.loc[split_idx, "valid_time"]

    print("\n--- Split calculation ---")
    print(f"[INFO] Total rows: {n_rows}")
    print(f"[INFO] Train fraction: {TRAIN_FRACTION}")
    print(f"[INFO] split_idx = int({n_rows} * {TRAIN_FRACTION}) = {split_idx}")
    print(f"[INFO] Split boundary date: {split_date}")

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print("\n--- Train set ---")
    print(f"[OK] Rows: {len(train_df)} ({len(train_df) / n_rows * 100:.2f}%)")
    print(f"[OK] Date range: {train_df['valid_time'].min()} to {train_df['valid_time'].max()}")

    print("\n--- Test set ---")
    print(f"[OK] Rows: {len(test_df)} ({len(test_df) / n_rows * 100:.2f}%)")
    print(f"[OK] Date range: {test_df['valid_time'].min()} to {test_df['valid_time'].max()}")

    # ---- Boundary sanity checks: no overlap, correct gap ----
    print("\n--- Boundary sanity checks ---")
    overlap = train_df["valid_time"].max() >= test_df["valid_time"].min()
    print(f"[CHECK] Train max >= Test min (should be False): {overlap}")
    if overlap:
        print("[ERROR] Train and test date ranges overlap!")

    gap_hours = (test_df["valid_time"].min() - train_df["valid_time"].max()).total_seconds() / 3600
    print(f"[CHECK] Gap between train end and test start: {gap_hours} hour(s) (should be 1.0, hourly data)")

    # ---- mwp_class coverage check ----
    print("\n--- mwp_class coverage check ---")
    train_classes = set(train_df["mwp_class"].dropna().unique())
    test_classes = set(test_df["mwp_class"].dropna().unique())
    all_classes = set(df["mwp_class"].dropna().unique())
    print(f"[INFO] All classes in full dataset: {[l for l in LABELS if l in all_classes]}")
    print(f"[INFO] Classes present in train: {[l for l in LABELS if l in train_classes]}")
    print(f"[INFO] Classes present in test:  {[l for l in LABELS if l in test_classes]}")

    missing_in_train = all_classes - train_classes
    missing_in_test = all_classes - test_classes
    if missing_in_train:
        print(f"[WARNING] Classes missing from train: {missing_in_train}")
    if missing_in_test:
        print(f"[WARNING] Classes missing from test: {missing_in_test}")
    if not missing_in_train and not missing_in_test:
        print("[OK] All classes present in both splits")

    print("\n--- Train class distribution ---")
    train_counts = train_df["mwp_class"].value_counts()
    for label in LABELS:
        cnt = train_counts.get(label, 0)
        print(f"     {label:12s}: {cnt:>8} rows  ({cnt / len(train_df) * 100:>5.2f}%)")

    print("\n--- Test class distribution ---")
    test_counts = test_df["mwp_class"].value_counts()
    for label in LABELS:
        cnt = test_counts.get(label, 0)
        print(f"     {label:12s}: {cnt:>8} rows  ({cnt / len(test_df) * 100:>5.2f}%)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_df.to_csv(TEST_OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {TRAIN_OUTPUT_PATH}")
    print(f"[SAVED] {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()