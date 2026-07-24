"""
Extended Validation Metrics - Atlantic

Loads the best checkpoint, runs inference on the full Atlantic test set,
inverse-transforms predictions and targets back to REAL UNITS using the
saved scaler, and computes four validation statistics per forecast variable
(swh, mwp, mwd):
  - Correlation coefficient (Pearson r)
  - Scatter Index (SI = RMSE / mean(actual))
  - Bias (mean(predicted - actual))
  - RMS Error (RMSE, in real units)

Saves: atlantic/results/validation_metrics.json
       atlantic/results/validation_metrics_table.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchtst_model import PatchTST

ATLANTIC_DIR = Path(__file__).resolve().parent.parent / "atlantic"
DATA_DIR = ATLANTIC_DIR / "data"
WINDOWS_DIR = DATA_DIR / "windows"
RESULTS_DIR = ATLANTIC_DIR / "results"
CHECKPOINT_PATH = ATLANTIC_DIR / "atlantic_patchtst_best.pt"
SCALER_PATH = ATLANTIC_DIR / "scaler_atlantic.pkl"

# Column order the scaler was fit on (from Phase 7) - needed to know which
# mean_/scale_ index belongs to which forecast variable
SCALE_COLS = ["u10", "v10", "swh", "mwp", "sin_mwd", "cos_mwd", "mwd"]
TARGET_COLS = ["swh", "mwp", "mwd"]
BATCH_SIZE = 256


def run_inference(model, X, batch_size=BATCH_SIZE, device="cpu"):
    model.eval()
    all_forecasts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            _, forecast = model(batch)
            all_forecasts.append(forecast.cpu().numpy())
    return np.concatenate(all_forecasts)


def inverse_transform_var(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    """values: any shape, scaled -> real units."""
    return values * scale + mean


def compute_stats(pred_real: np.ndarray, actual_real: np.ndarray) -> dict:
    pred_flat = pred_real.reshape(-1)
    actual_flat = actual_real.reshape(-1)

    correlation = float(np.corrcoef(pred_flat, actual_flat)[0, 1])
    bias = float(np.mean(pred_flat - actual_flat))
    rmse = float(np.sqrt(np.mean((pred_flat - actual_flat) ** 2)))
    mean_actual = float(np.mean(actual_flat))
    scatter_index = float(rmse / mean_actual) if mean_actual != 0 else float("nan")

    return {
        "correlation_coefficient": correlation,
        "scatter_index": scatter_index,
        "bias": bias,
        "rmse": rmse,
        "mean_actual": mean_actual,
    }


def main():
    print("=" * 60)
    print("EXTENDED VALIDATION METRICS - ATLANTIC")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found - run Phase 11 first.")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"{SCALER_PATH} not found - run Phase 7 first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = PatchTST().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")

    scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Loaded scaler: {SCALER_PATH.name}")

    X_test = np.load(WINDOWS_DIR / "X_test.npy")
    y_forecast_test = np.load(WINDOWS_DIR / "y_forecast_test.npy")  # [N, 20, 3], scaled
    print(f"[OK] Test set: X={X_test.shape}, y_forecast={y_forecast_test.shape}")

    print("\n--- Running inference ---")
    forecast_pred = run_inference(model, X_test, device=device)  # [N, 20, 3], scaled
    print("[OK] Inference complete")

    print("\n--- Inverse-transforming to real units ---")
    results = {}
    table_rows = []

    for i, var in enumerate(TARGET_COLS):
        scale_idx = SCALE_COLS.index(var)
        mean = scaler.mean_[scale_idx]
        scale = scaler.scale_[scale_idx]

        pred_scaled = forecast_pred[:, :, i]
        actual_scaled = y_forecast_test[:, :, i]

        pred_real = inverse_transform_var(pred_scaled, mean, scale)
        actual_real = inverse_transform_var(actual_scaled, mean, scale)

        stats = compute_stats(pred_real, actual_real)
        results[var] = stats

        print(f"\n[{var}]")
        print(f"     Correlation coefficient (r): {stats['correlation_coefficient']:.4f}")
        print(f"     Scatter Index (SI):          {stats['scatter_index']:.4f}")
        print(f"     Bias:                        {stats['bias']:.4f}")
        print(f"     RMS Error:                   {stats['rmse']:.4f}")

        table_rows.append(
            f"| {var} | {stats['correlation_coefficient']:.4f} | {stats['scatter_index']:.4f} | "
            f"{stats['bias']:.4f} | {stats['rmse']:.4f} |"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "validation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {RESULTS_DIR / 'validation_metrics.json'}")

    table_md = (
        "# Extended Validation Metrics — Atlantic\n\n"
        "Computed in real units (inverse-transformed via scaler_atlantic.pkl), "
        "aggregated across all 20 forecast steps (t+6h to t+120h).\n\n"
        "| Variable | Correlation Coefficient (r) | Scatter Index | Bias | RMS Error |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(table_rows)
        + "\n\nUnits: swh (m), mwp (s), mwd (deg).\n"
    )
    with open(RESULTS_DIR / "validation_metrics_table.md", "w") as f:
        f.write(table_md)
    print(f"[SAVED] {RESULTS_DIR / 'validation_metrics_table.md'}")

    print("\n[OK] Extended validation metrics complete for Atlantic.")


if __name__ == "__main__":
    main()