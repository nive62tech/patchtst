"""
Phase 4 - Label Definition (Classification Target)

Loads atlantic/data/atlantic_features.csv, confirms mwp (Tm01) stats against
the pre-confirmed Atlantic bin edges, applies those hardcoded edges via
pd.cut() to create mwp_class, checks class distribution, and saves the
dataframe back out with the mwp_class column added.

Bin edges are ATLANTIC-SPECIFIC and already confirmed via quartile split -
they are hardcoded here and must NOT be recalculated. Pacific and Indian
oceans will each get their own independently-computed edges later.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "atlantic" / "data"
INPUT_PATH = DATA_DIR / "atlantic_features.csv"
OUTPUT_PATH = DATA_DIR / "atlantic_features.csv"  # same file, mwp_class column added

# ---- CONFIRMED Atlantic bin edges - hardcoded, do NOT recalculate ----
BIN_EDGES = [4.615818, 6.937847, 7.637388, 8.518619, 13.928587]
LABELS = ["Low", "Moderate", "High", "Very High"]


def main():
    print("=" * 60)
    print("PHASE 4 - LABEL DEFINITION (CLASSIFICATION TARGET)")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Phase 3 first.")

    df = pd.read_csv(INPUT_PATH)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    print(f"[OK] Loaded {INPUT_PATH.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    if "mwp" not in df.columns:
        raise ValueError("mwp column missing - cannot proceed.")

    print("\n--- mwp (Tm01) sanity-check stats ---")
    print(f"[INFO] min:  {df['mwp'].min():.6f}")
    print(f"[INFO] max:  {df['mwp'].max():.6f}")
    print(f"[INFO] mean: {df['mwp'].mean():.6f}")
    print(f"[INFO] std:  {df['mwp'].std():.6f}")

    print("\n--- Confirmed Atlantic bin edges (hardcoded, not recalculated) ---")
    print(f"[INFO] bin_edges = {BIN_EDGES}")
    print(f"[INFO] labels    = {LABELS}")

    out_of_range = ((df["mwp"] < BIN_EDGES[0]) | (df["mwp"] > BIN_EDGES[-1])).sum()
    if out_of_range > 0:
        print(
            f"[WARNING] {out_of_range} rows have mwp outside the confirmed bin range "
            f"[{BIN_EDGES[0]}, {BIN_EDGES[-1]}] - these will become NaN in mwp_class "
            f"(pd.cut does not extrapolate beyond the given edges)"
        )
    else:
        print("[OK] All mwp values fall within the confirmed bin range")

    print("\n--- Applying pd.cut() ---")
    df["mwp_class"] = pd.cut(
        df["mwp"], bins=BIN_EDGES, labels=LABELS, include_lowest=True
    )

    unassigned = df["mwp_class"].isna().sum()
    print(f"[INFO] Rows with unassigned mwp_class (out of range): {unassigned}")

    print("\n--- Class distribution ---")
    counts = df["mwp_class"].value_counts().reindex(LABELS)
    pct = (counts / len(df) * 100).round(2)
    for label in LABELS:
        print(f"     {label:12s}: {counts[label]:>8} rows  ({pct[label]:>5.2f}%)")

    imbalance_ratio = counts.max() / counts.min()
    print(f"\n[INFO] Class imbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print(
            "[INFO] Noticeable imbalance - Phase 10's weighted CrossEntropyLoss "
            "will need to account for this"
        )
    else:
        print(
            "[INFO] Classes are reasonably balanced (expected, since bins are "
            "quartile-based on this same data)"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH} (mwp_class column added)")


if __name__ == "__main__":
    main()