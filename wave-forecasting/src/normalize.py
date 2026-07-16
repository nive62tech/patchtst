"""
Phase 7 - Normalization

Fits a StandardScaler on the training set only (atlantic_train.csv), applies
it to both atlantic_train.csv and atlantic_test.csv, and saves the fitted
scaler as atlantic/scaler_atlantic.pkl. mwp_class (the classification label)
and valid_time are never scaled.

Scaled columns: u10, v10, swh, mwp, sin_mwd, cos_mwd, mwd
  - The first 6 are the model's actual input channels.
  - mwd is also scaled because it doubles as a Phase 5 forecasting target
    (along with swh and mwp) - scaling it keeps train/eval consistent with
    the other two forecast targets, and the saved scaler lets Phase 12
    inverse-transform predictions back to real units for reporting.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "atlantic" / "data"
ATLANTIC_DIR = Path(__file__).resolve().parent.parent / "atlantic"

TRAIN_PATH = DATA_DIR / "atlantic_train.csv"
TEST_PATH = DATA_DIR / "atlantic_test.csv"
TRAIN_OUTPUT_PATH = DATA_DIR / "atlantic_train_scaled.csv"
TEST_OUTPUT_PATH = DATA_DIR / "atlantic_test_scaled.csv"
SCALER_OUTPUT_PATH = ATLANTIC_DIR / "scaler_atlantic.pkl"

SCALE_COLS = ["u10", "v10", "swh", "mwp", "sin_mwd", "cos_mwd", "mwd"]


def main():
    print("=" * 60)
    print("PHASE 7 - NORMALIZATION")
    print("=" * 60)

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "atlantic_train.csv / atlantic_test.csv not found - run Phase 6 first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df["valid_time"] = pd.to_datetime(train_df["valid_time"])
    test_df["valid_time"] = pd.to_datetime(test_df["valid_time"])

    print(f"[OK] Loaded train: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"[OK] Loaded test:  {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    missing_train = [c for c in SCALE_COLS if c not in train_df.columns]
    if missing_train:
        raise ValueError(f"Missing columns in train set: {missing_train}")

    print(f"\n[INFO] Columns to scale: {SCALE_COLS}")
    print("[INFO] Columns NOT scaled: valid_time, mwp_class")

    print("\n--- Fitting StandardScaler on TRAIN ONLY ---")
    scaler = StandardScaler()
    scaler.fit(train_df[SCALE_COLS])

    print("[OK] Scaler fitted. Per-column mean/std (from train):")
    for col, mean, std in zip(SCALE_COLS, scaler.mean_, scaler.scale_):
        print(f"     {col:10s}: mean={mean:.6f}  std={std:.6f}")

    print("\n--- Transforming train and test using the SAME fitted scaler ---")
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled[SCALE_COLS] = scaler.transform(train_df[SCALE_COLS])
    test_scaled[SCALE_COLS] = scaler.transform(test_df[SCALE_COLS])

    print("[OK] Transform applied to both sets")

    print("\n--- Post-scaling sanity check (train) ---")
    for col in SCALE_COLS:
        m = train_scaled[col].mean()
        s = train_scaled[col].std()
        print(f"     {col:10s}: mean={m:.6f} (should be ~0)  std={s:.6f} (should be ~1)")

    print(
        "\n--- Post-scaling stats (test - NOT expected to be exactly 0/1, "
        "since scaler was fit on train only) ---"
    )
    for col in SCALE_COLS:
        m = test_scaled[col].mean()
        s = test_scaled[col].std()
        print(f"     {col:10s}: mean={m:.6f}  std={s:.6f}")

    print(f"\n[INFO] mwp_class left untouched - dtype: {train_scaled['mwp_class'].dtype}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATLANTIC_DIR.mkdir(parents=True, exist_ok=True)

    train_scaled.to_csv(TRAIN_OUTPUT_PATH, index=False)
    test_scaled.to_csv(TEST_OUTPUT_PATH, index=False)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)

    print(f"\n[SAVED] {TRAIN_OUTPUT_PATH}")
    print(f"[SAVED] {TEST_OUTPUT_PATH}")
    print(f"[SAVED] {SCALER_OUTPUT_PATH}")


if __name__ == "__main__":
    main()