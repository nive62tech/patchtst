"""
Phase 1 - Data Loading & Merging: Bay of Bengal
Merges the raw ERA5 wind-variable file and wave-variable file into a single
chronologically-sorted CSV for the Bay of Bengal buoy location (BD08: 17.833N, 89.216E).
"""

import pandas as pd
from pathlib import Path

SEA = "bay_of_bengal"
DATA_DIR = Path(__file__).resolve().parent.parent / SEA / "data"

RAW_FILE1 = DATA_DIR / f"{SEA}_raw_file1.csv"   # wind vars: u10, v10, u100, v100, fg10, blh
RAW_FILE2 = DATA_DIR / f"{SEA}_raw_file2.csv"   # wave vars: mwd, mwp, swh
MERGED_OUT = DATA_DIR / f"{SEA}_merged.csv"


def main():
    print(f"[{SEA}] Loading raw files...")
    df1 = pd.read_csv(RAW_FILE1, parse_dates=["valid_time"])
    df2 = pd.read_csv(RAW_FILE2, parse_dates=["valid_time"])

    print(f"[{SEA}] File 1 (wind): {df1.shape[0]} rows, columns: {list(df1.columns)}")
    print(f"[{SEA}] File 2 (wave): {df2.shape[0]} rows, columns: {list(df2.columns)}")

    # File 2 carries its own lat/lon (slightly different ERA5 grid snap) - drop before merge,
    # keep File 1's lat/lon as the record of location.
    df2 = df2.drop(columns=["latitude", "longitude"], errors="ignore")

    merged = pd.merge(df1, df2, on="valid_time", how="inner")
    merged = merged.sort_values("valid_time").reset_index(drop=True)

    n_dupes = merged["valid_time"].duplicated().sum()
    assert n_dupes == 0, f"[{SEA}] Found {n_dupes} duplicate valid_time rows after merge!"

    expected_cols = {"valid_time", "u10", "v10", "u100", "v100", "fg10", "blh",
                      "latitude", "longitude", "mwd", "mwp", "swh"}
    missing = expected_cols - set(merged.columns)
    assert not missing, f"[{SEA}] Missing expected columns after merge: {missing}"

    print(f"[{SEA}] Merged shape: {merged.shape}")
    print(f"[{SEA}] Date range: {merged['valid_time'].min()} to {merged['valid_time'].max()}")
    print(f"[{SEA}] File1 rows: {len(df1)}, File2 rows: {len(df2)}, Merged (inner join) rows: {len(merged)}")

    merged.to_csv(MERGED_OUT, index=False)
    print(f"[{SEA}] Saved merged file to: {MERGED_OUT}")


if __name__ == "__main__":
    main()
