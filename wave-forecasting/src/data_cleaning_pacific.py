"""
Phase 2 (Pacific) - Data Cleaning

Loads pacific/data/pacific_merged.csv, finds missing timestamps by comparing
against the expected full hourly range, counts NaNs per column, linear-
interpolates gaps of <=3 consecutive missing hours, drops rows for larger
gaps, removes duplicate timestamps, and saves pacific/data/pacific_clean.csv.

Mirrors src/data_cleaning.py (Atlantic) exactly, renamed for Pacific.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "pacific" / "data"
INPUT_PATH = DATA_DIR / "pacific_merged.csv"
OUTPUT_PATH = DATA_DIR / "pacific_clean.csv"

VALUE_COLS = ["u10", "v10", "swh", "mwp", "mwd"]
MAX_INTERP_GAP_HOURS = 3


def main():
    print("=" * 60)
    print("PHASE 2 (PACIFIC) - DATA CLEANING")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Pacific Phase 1 first.")

    df = pd.read_csv(INPUT_PATH)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    print(f"[OK] Loaded {INPUT_PATH.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n--- Duplicate timestamps ---")
    before = len(df)
    df = df.drop_duplicates(subset="valid_time", keep="first")
    dupes_removed = before - len(df)
    print(f"[INFO] Removed {dupes_removed} duplicate valid_time rows (kept first occurrence)")

    df = df.sort_values("valid_time").reset_index(drop=True)

    print("\n--- Missing timestamp detection ---")
    full_range = pd.date_range(start=df["valid_time"].min(), end=df["valid_time"].max(), freq="h")
    print(
        f"[INFO] Expected {len(full_range)} hourly timestamps between "
        f"{df['valid_time'].min()} and {df['valid_time'].max()}"
    )
    print(f"[INFO] Actual rows before reindex: {len(df)}")
    missing_count = len(full_range) - len(df)
    print(f"[INFO] Missing timestamps: {missing_count}")

    df = df.set_index("valid_time").reindex(full_range)
    df.index.name = "valid_time"

    print("\n--- NaN counts per column (post-reindex) ---")
    for col in VALUE_COLS:
        print(f"     {col}: {df[col].isna().sum()} NaNs")

    print(f"\n--- Interpolating gaps <= {MAX_INTERP_GAP_HOURS} hours ---")
    df[VALUE_COLS] = df[VALUE_COLS].interpolate(
        method="linear", limit=MAX_INTERP_GAP_HOURS, limit_area="inside"
    )

    remaining_nans = df[VALUE_COLS].isna().any(axis=1).sum()
    print(
        f"[INFO] Rows still containing NaN after interpolation (gaps > "
        f"{MAX_INTERP_GAP_HOURS}h, or at start/end edges): {remaining_nans}"
    )

    before_drop = len(df)
    df = df.dropna(subset=VALUE_COLS)
    dropped = before_drop - len(df)
    print(f"[INFO] Dropped {dropped} rows with unfillable gaps")

    df = df.reset_index()
    print("\n--- Final result ---")
    print(f"[OK] Final row count: {len(df)}")
    print(f"[OK] Date range: {df['valid_time'].min()} to {df['valid_time'].max()}")

    final_nan_check = df[VALUE_COLS].isna().sum().sum()
    print(f"[OK] Remaining NaNs (should be 0): {final_nan_check}")

    dupe_check = df["valid_time"].duplicated().sum()
    print(f"[OK] Remaining duplicate timestamps (should be 0): {dupe_check}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()