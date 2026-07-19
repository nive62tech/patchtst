"""
Phase 3 (Pacific) - Feature Engineering (Direction Encoding)

Loads pacific/data/pacific_clean.csv, applies sin/cos circular encoding to
mwd (Formula 3), keeps a copy of raw mwd in the file for later use as a
Phase 5 forecasting target, and saves pacific/data/pacific_features.csv.

Final 6 model input channels: u10, v10, swh, mwp, sin_mwd, cos_mwd
(mwd itself stays in the file, unscaled, purely as forecasting-target
material - it is NOT one of the model's input channels)

Mirrors src/feature_engineering.py (Atlantic) exactly, renamed for Pacific.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "pacific" / "data"
INPUT_PATH = DATA_DIR / "pacific_clean.csv"
OUTPUT_PATH = DATA_DIR / "pacific_features.csv"

MODEL_INPUT_COLS = ["u10", "v10", "swh", "mwp", "sin_mwd", "cos_mwd"]


def main():
    print("=" * 60)
    print("PHASE 3 (PACIFIC) - FEATURE ENGINEERING (DIRECTION ENCODING)")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"{INPUT_PATH} not found - run Pacific Phase 2 first.")

    df = pd.read_csv(INPUT_PATH)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    print(f"[OK] Loaded {INPUT_PATH.name}: {df.shape[0]} rows, {df.shape[1]} columns")

    if "mwd" not in df.columns:
        raise ValueError("mwd column missing - cannot apply circular encoding.")

    print("\n--- Applying sin/cos circular encoding to mwd (Formula 3) ---")
    print("[INFO] sin_mwd = sin(mwd * pi/180), cos_mwd = cos(mwd * pi/180)")

    radians = np.deg2rad(df["mwd"])
    df["sin_mwd"] = np.sin(radians)
    df["cos_mwd"] = np.cos(radians)

    print(f"[OK] sin_mwd range: [{df['sin_mwd'].min():.4f}, {df['sin_mwd'].max():.4f}]")
    print(f"[OK] cos_mwd range: [{df['cos_mwd'].min():.4f}, {df['cos_mwd'].max():.4f}]")

    unit_circle_check = df["sin_mwd"] ** 2 + df["cos_mwd"] ** 2
    max_dev = (unit_circle_check - 1).abs().max()
    print(f"[CHECK] max deviation of sin^2+cos^2 from 1.0: {max_dev:.2e} (should be ~0)")

    print("\n--- Column layout ---")
    print("[INFO] Model input channels (6):", MODEL_INPUT_COLS)
    print(
        "[INFO] mwd (raw, degrees) is KEPT in the file - not a model input, "
        "needed as a Phase 5 forecasting target"
    )

    missing_inputs = [c for c in MODEL_INPUT_COLS if c not in df.columns]
    if missing_inputs:
        print(f"[ERROR] Missing expected model input columns: {missing_inputs}")

    final_cols = ["valid_time"] + MODEL_INPUT_COLS + ["mwd"]
    df = df[final_cols]

    print(f"\n[OK] Final dataframe: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"     Columns: {list(df.columns)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()