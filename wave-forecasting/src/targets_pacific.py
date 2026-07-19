"""
Phase 5 (Pacific) - Forecasting Targets

Defines and verifies the target construction logic for the forecasting head,
identical to src/targets.py (Atlantic), renamed for Pacific: given a starting
index i (the same starting index as the Phase 8 input window), builds a
sequence of 20 future values (t+6h, t+12h, ..., t+120h) for swh, mwp, and raw
mwd. This module is imported (logic reproduced inline) by Phase 8's Pacific
windowing script - it does not itself split the data or produce final .npy
files.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "pacific" / "data"
INPUT_PATH = DATA_DIR / "pacific_features.csv"

TARGET_COLS = ["swh", "mwp", "mwd"]
FORECAST_STEPS = 20
STEP_HOURS = 6
INPUT_WINDOW_LEN = 72


def get_forecast_target_indices(start_idx: int) -> list:
    return list(
        range(start_idx + STEP_HOURS, start_idx + STEP_HOURS + FORECAST_STEPS * STEP_HOURS, STEP_HOURS)
    )


def build_single_target(values: np.ndarray, start_idx: int):
    idxs = get_forecast_target_indices(start_idx)
    if idxs[-1] >= len(values):
        return None
    return values[idxs, :]


def main():
    print("=" * 60)
    print("PHASE 5 (PACIFIC) - FORECASTING TARGETS")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Pacific Phase 4 first.")

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

    last_valid_target_start = n_rows - (FORECAST_STEPS * STEP_HOURS) - 1
    print("\n--- Coverage check ---")
    print(f"[INFO] Total rows: {n_rows}")
    print(f"[INFO] Last start_idx with a complete 20-step target: {last_valid_target_start}")
    print(
        f"[INFO] Rows unusable at the tail (can't form a full target): "
        f"{n_rows - 1 - last_valid_target_start}"
    )

    print("\n--- Demo target for start_idx=0 ---")
    demo_target = build_single_target(values, 0)
    print(f"[OK] Demo target shape: {demo_target.shape}")

    demo_target_last = build_single_target(values, last_valid_target_start)
    print(f"\n--- Demo target for start_idx={last_valid_target_start} (last valid) ---")
    print(f"[OK] Shape: {demo_target_last.shape}")

    should_be_none = build_single_target(values, last_valid_target_start + 1)
    print(
        f"\n[CHECK] start_idx={last_valid_target_start + 1} (one past last valid) "
        f"returns: {should_be_none} (should be None)"
    )

    print(
        "\n[OK] Target construction logic verified for Pacific. This module's logic "
        "will be reproduced in Phase 8's windowing_pacific.py to build y_forecast arrays "
        "separately for train and test splits."
    )


if __name__ == "__main__":
    main()