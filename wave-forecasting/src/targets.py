"""
Phase 5 - Forecasting Targets

Defines the target construction logic for the forecasting head: given a
starting index i (the same starting index as the Phase 8 input window),
builds a sequence of 20 future values (t+6h, t+12h, ..., t+120h) for swh,
mwp, and raw mwd. This module is imported by Phase 8's windowing script -
it does not itself split the data or produce final .npy files (that happens
in Phase 8, separately for train and test, after Phase 6's split and
Phase 7's scaling).

Forecast target variables: swh, mwp (Tm01), raw mwd (un-encoded, degrees).
mdts/mdww are not part of this project (dropped in Phase 1/3 - not available
in the ERA5 time-series product being used) and were never planned as
targets even in the original 10-channel design.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "atlantic" / "data"
INPUT_PATH = DATA_DIR / "atlantic_features.csv"

TARGET_COLS = ["swh", "mwp", "mwd"]
FORECAST_STEPS = 20
STEP_HOURS = 6
INPUT_WINDOW_LEN = 72  # from Phase 8 spec - used here only for sanity-checking sample counts


def get_forecast_target_indices(start_idx: int) -> list:
    """
    Returns the 20 row indices used for a forecast target starting at
    start_idx, matching the Phase 8 spec exactly:
        y_forecast[i] = [swh, mwp, mwd][i+6 : i+126 : 6]
    i.e. 20 steps, t+6h through t+120h relative to start_idx.
    """
    return list(
        range(start_idx + STEP_HOURS, start_idx + STEP_HOURS + FORECAST_STEPS * STEP_HOURS, STEP_HOURS)
    )


def build_single_target(values: np.ndarray, start_idx: int):
    """
    values: 2D numpy array of shape [n_rows, 3] for columns [swh, mwp, mwd]
    start_idx: the window's starting index i

    Returns shape [20, 3], or None if any required index is out of bounds.
    """
    idxs = get_forecast_target_indices(start_idx)
    if idxs[-1] >= len(values):
        return None
    return values[idxs, :]


def main():
    print("=" * 60)
    print("PHASE 5 - FORECASTING TARGETS")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Phase 4 first.")

    df = pd.read_csv(INPUT_PATH)
    print(f"[OK] Loaded {INPUT_PATH.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing forecast target columns: {missing}")

    print(f"\n[INFO] Forecast target columns: {TARGET_COLS}")
    print(
        f"[INFO] Forecast horizon: t+{STEP_HOURS}h to t+{FORECAST_STEPS * STEP_HOURS}h, "
        f"{FORECAST_STEPS} steps at {STEP_HOURS}h intervals"
    )

    values = df[TARGET_COLS].to_numpy()
    n_rows = len(values)

    print("\n--- Index structure check ---")
    example_idxs = get_forecast_target_indices(0)
    print(f"[INFO] For start_idx=0, target row indices = {example_idxs}")
    print("[INFO] (i.e., t+6h through t+120h relative to the window start)")

    # ---- Coverage: how many starting indices have a complete 20-step target? ----
    last_valid_target_start = n_rows - (FORECAST_STEPS * STEP_HOURS) - 1
    print("\n--- Coverage check ---")
    print(f"[INFO] Total rows: {n_rows}")
    print(f"[INFO] Last start_idx with a complete 20-step target: {last_valid_target_start}")
    print(
        f"[INFO] Rows unusable at the tail (can't form a full target): "
        f"{n_rows - 1 - last_valid_target_start}"
    )

    # ---- Demo: build one target and show it ----
    print("\n--- Demo target for start_idx=0 ---")
    demo_target = build_single_target(values, 0)
    print(f"[OK] Demo target shape: {demo_target.shape}")
    print(f"[OK] Demo target values:\n{demo_target}")

    # ---- Demo: last valid target ----
    demo_target_last = build_single_target(values, last_valid_target_start)
    print(f"\n--- Demo target for start_idx={last_valid_target_start} (last valid) ---")
    print(f"[OK] Shape: {demo_target_last.shape}")

    # ---- Confirm one-past-the-end fails as expected ----
    should_be_none = build_single_target(values, last_valid_target_start + 1)
    print(
        f"\n[CHECK] start_idx={last_valid_target_start + 1} (one past last valid) "
        f"returns: {should_be_none} (should be None)"
    )

    print(
        "\n[OK] Target construction logic verified. This module (src/targets.py) "
        "will be imported by Phase 8's windowing script to build y_forecast arrays "
        "separately for train and test splits."
    )


if __name__ == "__main__":
    main()