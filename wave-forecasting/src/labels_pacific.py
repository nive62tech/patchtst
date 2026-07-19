"""
Phase 4 (Pacific) - Label Definition (Classification Target)

Loads pacific/data/pacific_features.csv, computes Pacific's OWN quartile-based
mwp_class bin edges (independently from Atlantic's - not shared, since wave
period ranges differ by location), applies them via pd.cut(), checks class
distribution, and saves the dataframe back out with the mwp_class column
added.

Bin edges here are computed FRESH from Pacific's own cleaned mwp distribution
via quartile split - the same method originally used to derive Atlantic's
edges. Once this run prints the edges, they should be treated as CONFIRMED
and hardcoded for all future Pacific runs, exactly as Atlantic's are.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "pacific" / "data"
INPUT_PATH = DATA_DIR / "pacific_features.csv"
OUTPUT_PATH = DATA_DIR / "pacific_features.csv"  # same file, mwp_class column added

LABELS = ["Low", "Moderate", "High", "Very High"]


def main():
    print("=" * 60)
    print("PHASE 4 (PACIFIC) - LABEL DEFINITION (CLASSIFICATION TARGET)")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Pacific Phase 3 first.")

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

    print("\n--- Computing Pacific's OWN quartile bin edges (NOT reusing Atlantic's) ---")
    quantile_points = [0, 0.25, 0.5, 0.75, 1.0]
    BIN_EDGES = df["mwp"].quantile(quantile_points).tolist()
    print(f"[INFO] Quantile points: {quantile_points}")
    print(f"[COMPUTED] bin_edges = {BIN_EDGES}")
    print(f"[INFO] labels        = {LABELS}")
    print(
        "\n[ACTION REQUIRED] Treat these edges as CONFIRMED from this point on - "
        "do NOT recalculate on future runs. They get recorded in this phase's README."
    )

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