"""
Phase 8 - Sequence Windowing

Builds 72-hour input windows and matching classification/forecasting targets,
separately within atlantic_train_scaled.csv and atlantic_test_scaled.csv -
windows never span the train/test boundary. Saves the resulting numpy arrays
under atlantic/data/windows/.

For each valid starting index i within a split:
  X[i]          = features[i : i+72]                      -> shape [72, 6]
  y_class[i]    = mwp_class[i+72]  (as an integer index)   -> scalar
  y_forecast[i] = [swh, mwp, mwd][i+6 : i+126 : 6]         -> shape [20, 3]

Index formula for y_forecast matches src/targets.py (Phase 5) exactly.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "atlantic" / "data"
WINDOWS_DIR = DATA_DIR / "windows"

TRAIN_PATH = DATA_DIR / "atlantic_train_scaled.csv"
TEST_PATH = DATA_DIR / "atlantic_test_scaled.csv"

MODEL_INPUT_COLS = ["u10", "v10", "swh", "mwp", "sin_mwd", "cos_mwd"]
TARGET_COLS = ["swh", "mwp", "mwd"]
LABELS = ["Low", "Moderate", "High", "Very High"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

INPUT_WINDOW_LEN = 72
FORECAST_STEPS = 20
STEP_HOURS = 6


def get_forecast_target_indices(start_idx: int) -> list:
    """Same formula as Phase 5's src/targets.py."""
    return list(
        range(start_idx + STEP_HOURS, start_idx + STEP_HOURS + FORECAST_STEPS * STEP_HOURS, STEP_HOURS)
    )


def build_windows(df: pd.DataFrame, label: str):
    print(f"\n--- Building windows for {label} ---")
    n_rows = len(df)

    X_values = df[MODEL_INPUT_COLS].to_numpy()
    target_values = df[TARGET_COLS].to_numpy()
    class_values = df["mwp_class"].map(LABEL_TO_IDX).to_numpy()

    n_missing_class = pd.isna(df["mwp_class"]).sum()
    if n_missing_class > 0:
        print(f"[WARNING] {n_missing_class} rows have missing mwp_class - affected windows will be skipped")

    # Binding constraints on the last valid start_idx:
    #   classification needs row i+72 to exist      -> i <= n_rows - 73
    #   forecast needs row i+120 to exist            -> i <= n_rows - 121
    max_start_for_class = n_rows - INPUT_WINDOW_LEN - 1
    max_start_for_forecast = n_rows - (FORECAST_STEPS * STEP_HOURS) - 1
    last_valid_start = min(max_start_for_class, max_start_for_forecast)

    print(f"[INFO] Total rows: {n_rows}")
    print(f"[INFO] Last valid start_idx: {last_valid_start}")
    print(f"[INFO] Total windows to build: {last_valid_start + 1}")

    X_list = []
    y_class_list = []
    y_forecast_list = []
    skipped_nan_class = 0

    for i in range(last_valid_start + 1):
        class_label = class_values[i + INPUT_WINDOW_LEN]
        if pd.isna(class_label):
            skipped_nan_class += 1
            continue

        X_window = X_values[i : i + INPUT_WINDOW_LEN]
        forecast_idxs = get_forecast_target_indices(i)
        y_forecast_window = target_values[forecast_idxs, :]

        X_list.append(X_window)
        y_class_list.append(class_label)
        y_forecast_list.append(y_forecast_window)

    X = np.array(X_list, dtype=np.float32)
    y_class = np.array(y_class_list, dtype=np.int64)
    y_forecast = np.array(y_forecast_list, dtype=np.float32)

    print(f"[OK] X shape: {X.shape}")
    print(f"[OK] y_class shape: {y_class.shape}")
    print(f"[OK] y_forecast shape: {y_forecast.shape}")
    if skipped_nan_class > 0:
        print(f"[INFO] Skipped {skipped_nan_class} windows due to missing mwp_class")

    return X, y_class, y_forecast


def main():
    print("=" * 60)
    print("PHASE 8 - SEQUENCE WINDOWING")
    print("=" * 60)

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "atlantic_train_scaled.csv / atlantic_test_scaled.csv not found - run Phase 7 first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"[OK] Loaded train: {train_df.shape[0]} rows")
    print(f"[OK] Loaded test:  {test_df.shape[0]} rows")

    print(f"\n[INFO] Model input channels (6): {MODEL_INPUT_COLS}")
    print(f"[INFO] Forecast target channels (3): {TARGET_COLS}")
    print(f"[INFO] Class label mapping: {LABEL_TO_IDX}")

    X_train, y_class_train, y_forecast_train = build_windows(train_df, "TRAIN")
    X_test, y_class_test, y_forecast_test = build_windows(test_df, "TEST")

    print("\n--- Final shape summary ---")
    print(f"[OK] X_train:          {X_train.shape}  (expected [samples, 72, 6])")
    print(f"[OK] y_class_train:    {y_class_train.shape}  (expected [samples])")
    print(f"[OK] y_forecast_train: {y_forecast_train.shape}  (expected [samples, 20, 3])")
    print(f"[OK] X_test:           {X_test.shape}")
    print(f"[OK] y_class_test:     {y_class_test.shape}")
    print(f"[OK] y_forecast_test:  {y_forecast_test.shape}")

    print("\n--- y_class distribution (train) ---")
    for lbl, idx in LABEL_TO_IDX.items():
        cnt = (y_class_train == idx).sum()
        print(f"     {lbl:12s} (idx {idx}): {cnt:>8} windows  ({cnt / len(y_class_train) * 100:>5.2f}%)")

    print("\n--- y_class distribution (test) ---")
    for lbl, idx in LABEL_TO_IDX.items():
        cnt = (y_class_test == idx).sum()
        print(f"     {lbl:12s} (idx {idx}): {cnt:>8} windows  ({cnt / len(y_class_test) * 100:>5.2f}%)")

    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(WINDOWS_DIR / "X_train.npy", X_train)
    np.save(WINDOWS_DIR / "y_class_train.npy", y_class_train)
    np.save(WINDOWS_DIR / "y_forecast_train.npy", y_forecast_train)
    np.save(WINDOWS_DIR / "X_test.npy", X_test)
    np.save(WINDOWS_DIR / "y_class_test.npy", y_class_test)
    np.save(WINDOWS_DIR / "y_forecast_test.npy", y_forecast_test)

    print(f"\n[SAVED] All 6 .npy files to {WINDOWS_DIR}")


if __name__ == "__main__":
    main()