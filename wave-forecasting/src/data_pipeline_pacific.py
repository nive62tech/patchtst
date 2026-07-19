"""
Phase 1 (Pacific) - Data Loading & Merging

Loads:
  pacific/data/pacific_raw_file1.csv  (u10, v10, u100, v100, fg10)
  pacific/data/pacific_raw_file2.csv  (mwd, mwp, swh)

Merges on valid_time (inner join), sorts ascending, drops unneeded columns,
and saves pacific/data/pacific_merged.csv with final columns:
  valid_time, u10, v10, swh, mwp, mwd

Mirrors src/data_pipeline.py (Atlantic) exactly, renamed for Pacific.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "pacific" / "data"
FILE1_PATH = DATA_DIR / "pacific_raw_file1.csv"
FILE2_PATH = DATA_DIR / "pacific_raw_file2.csv"
OUTPUT_PATH = DATA_DIR / "pacific_merged.csv"

FILE1_EXPECTED_COLS = {"valid_time", "u10", "v10", "u100", "v100", "fg10"}
FILE2_EXPECTED_COLS = {"valid_time", "mwd", "mwp", "swh"}

DROP_COLS = ["u100", "v100", "fg10", "latitude", "longitude"]
FINAL_COLS = ["valid_time", "u10", "v10", "swh", "mwp", "mwd"]


def load_file(path: Path, expected_cols: set, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found at {path}. "
            f"Place the raw CSV there before running this script."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = expected_cols - set(df.columns)
    if missing:
        print(f"[WARNING] {label}: missing expected columns: {missing}")
        print(f"[WARNING] {label}: columns found: {list(df.columns)}")

    if "valid_time" not in df.columns:
        raise ValueError(f"{label} has no 'valid_time' column - cannot proceed.")

    df["valid_time"] = pd.to_datetime(df["valid_time"])

    print(f"[OK] Loaded {label}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"     Columns: {list(df.columns)}")
    print(f"     Date range: {df['valid_time'].min()} to {df['valid_time'].max()}")
    return df


def check_frequency(df: pd.DataFrame, label: str):
    diffs = df["valid_time"].sort_values().diff().dropna()
    if len(diffs) == 0:
        print(f"[WARNING] {label}: not enough rows to check frequency")
        return
    mode_diff = diffs.mode().iloc[0]
    print(f"[CHECK] {label}: most common timestamp gap = {mode_diff}")
    non_hourly = diffs[diffs != pd.Timedelta(hours=1)]
    if len(non_hourly) > 0:
        print(
            f"[WARNING] {label}: {len(non_hourly)} gaps are not exactly 1 hour "
            f"(expected if there are missing timestamps - Phase 2 handles this)"
        )


def main():
    print("=" * 60)
    print("PHASE 1 (PACIFIC) - DATA LOADING & MERGING")
    print("=" * 60)

    df1 = load_file(FILE1_PATH, FILE1_EXPECTED_COLS, "File 1 (wind vars)")
    df2 = load_file(FILE2_PATH, FILE2_EXPECTED_COLS, "File 2 (wave vars)")

    print("\n--- Frequency check ---")
    check_frequency(df1, "File 1")
    check_frequency(df2, "File 2")

    print("\n--- Merging on valid_time (inner join) ---")
    merged = pd.merge(df1, df2, on="valid_time", how="inner")
    print(f"[OK] Merged shape: {merged.shape}")
    if merged.shape[0] == 0:
        raise ValueError(
            "Merge produced 0 rows. Check that valid_time formats/timezones "
            "match between File 1 and File 2."
        )

    print("\n--- Sorting by valid_time ascending ---")
    merged = merged.sort_values("valid_time").reset_index(drop=True)

    print("\n--- Dropping unneeded columns ---")
    cols_present_to_drop = [c for c in DROP_COLS if c in merged.columns]
    print(f"[INFO] Dropping: {cols_present_to_drop}")
    merged = merged.drop(columns=cols_present_to_drop, errors="ignore")

    print("\n--- Verifying final columns ---")
    missing_final = set(FINAL_COLS) - set(merged.columns)
    extra_final = set(merged.columns) - set(FINAL_COLS)
    if missing_final:
        print(f"[ERROR] Missing expected final columns: {missing_final}")
    if extra_final:
        print(f"[WARNING] Unexpected extra columns still present: {extra_final}")

    merged = merged[[c for c in FINAL_COLS if c in merged.columns]]

    print(f"\n[OK] Final merged dataframe: {merged.shape[0]} rows, {merged.shape[1]} columns")
    print(f"     Columns: {list(merged.columns)}")
    print(f"     Date range: {merged['valid_time'].min()} to {merged['valid_time'].max()}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()