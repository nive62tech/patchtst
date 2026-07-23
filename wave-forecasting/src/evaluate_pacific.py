"""
Phase 12 (Pacific) - Model Evaluation on Test Set

Loads the best Pacific checkpoint, runs inference on the full Pacific test
set (never used in training or validation), and computes classification and
forecasting metrics, plus a year-by-year breakdown.

Saves: pacific/results/confusion_matrix.png, horizon_error_curve.png,
metrics_report.json

Mirrors src/evaluate.py (Atlantic) exactly, renamed for Pacific.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchtst_model import PatchTST

PACIFIC_DIR = Path(__file__).resolve().parent.parent / "pacific"
DATA_DIR = PACIFIC_DIR / "data"
WINDOWS_DIR = DATA_DIR / "windows"
RESULTS_DIR = PACIFIC_DIR / "results"
CHECKPOINT_PATH = PACIFIC_DIR / "pacific_patchtst_best.pt"
TEST_CSV_PATH = DATA_DIR / "pacific_test_scaled.csv"

LABELS = ["Low", "Moderate", "High", "Very High"]
FORECAST_HORIZONS = [f"t+{6 * (i + 1)}h" for i in range(20)]

INPUT_WINDOW_LEN = 72
FORECAST_STEPS = 20
STEP_HOURS = 6
BATCH_SIZE = 256


def get_window_reference_dates(df: pd.DataFrame) -> np.ndarray:
    n_rows = len(df)
    max_start_for_class = n_rows - INPUT_WINDOW_LEN - 1
    max_start_for_forecast = n_rows - (FORECAST_STEPS * STEP_HOURS) - 1
    last_valid_start = min(max_start_for_class, max_start_for_forecast)

    valid_times = df["valid_time"].to_numpy()
    ref_dates = valid_times[INPUT_WINDOW_LEN : INPUT_WINDOW_LEN + last_valid_start + 1]
    return ref_dates


def run_inference(model, X, batch_size=BATCH_SIZE, device="cpu"):
    model.eval()
    all_class_logits = []
    all_forecasts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            class_logits, forecast = model(batch)
            all_class_logits.append(class_logits.cpu().numpy())
            all_forecasts.append(forecast.cpu().numpy())
    return np.concatenate(all_class_logits), np.concatenate(all_forecasts)


def main():
    print("=" * 60)
    print("PHASE 12 (PACIFIC) - MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"{CHECKPOINT_PATH} not found - run Pacific Phase 11 first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = PatchTST().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"[INFO] Checkpoint val_loss: {checkpoint['val_loss']:.4f}")

    X_test = np.load(WINDOWS_DIR / "X_test.npy")
    y_class_test = np.load(WINDOWS_DIR / "y_class_test.npy")
    y_forecast_test = np.load(WINDOWS_DIR / "y_forecast_test.npy")
    print(f"\n[OK] Test set: X={X_test.shape}, y_class={y_class_test.shape}, y_forecast={y_forecast_test.shape}")

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df["valid_time"] = pd.to_datetime(test_df["valid_time"])
    ref_dates = get_window_reference_dates(test_df)
    assert len(ref_dates) == len(
        y_class_test
    ), f"Reference date count ({len(ref_dates)}) doesn't match window count ({len(y_class_test)})"
    print(f"[OK] Reference dates reconstructed and aligned: {len(ref_dates)} dates")

    print("\n--- Running inference on full test set ---")
    class_logits, forecast_pred = run_inference(model, X_test, device=device)
    class_preds = class_logits.argmax(axis=1)
    print("[OK] Inference complete")

    print("\n--- Classification metrics ---")
    accuracy = accuracy_score(y_class_test, class_preds)
    f1_weighted = f1_score(y_class_test, class_preds, average="weighted")
    f1_macro = f1_score(y_class_test, class_preds, average="macro")
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_class_test, class_preds, labels=[0, 1, 2, 3]
    )

    print(f"[OK] Accuracy: {accuracy:.4f}")
    print(f"[OK] Weighted F1: {f1_weighted:.4f}")
    print(f"[OK] Macro F1: {f1_macro:.4f}")
    print("\n--- Per-class metrics ---")
    for i, label in enumerate(LABELS):
        print(
            f"     {label:12s}: precision={precision[i]:.4f}  recall={recall[i]:.4f}  "
            f"f1={f1_per_class[i]:.4f}  support={support[i]}"
        )

    cm = confusion_matrix(y_class_test, class_preds, labels=[0, 1, 2, 3])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Sea State Confusion Matrix (Pacific Test Set)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\n[SAVED] {RESULTS_DIR / 'confusion_matrix.png'}")

    print("\n--- Forecasting metrics (per horizon step) ---")
    mae_per_step = []
    rmse_per_step = []
    r2_per_step = []

    for step in range(FORECAST_STEPS):
        y_true_step = y_forecast_test[:, step, :].reshape(-1)
        y_pred_step = forecast_pred[:, step, :].reshape(-1)
        mae = mean_absolute_error(y_true_step, y_pred_step)
        rmse = np.sqrt(mean_squared_error(y_true_step, y_pred_step))
        r2 = r2_score(y_true_step, y_pred_step)
        mae_per_step.append(mae)
        rmse_per_step.append(rmse)
        r2_per_step.append(r2)
        print(f"     {FORECAST_HORIZONS[step]:8s}: MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    overall_r2 = r2_score(y_forecast_test.reshape(-1), forecast_pred.reshape(-1))
    overall_rmse = np.sqrt(mean_squared_error(y_forecast_test.reshape(-1), forecast_pred.reshape(-1)))
    overall_mae = mean_absolute_error(y_forecast_test.reshape(-1), forecast_pred.reshape(-1))
    print(f"\n[OK] Overall forecast MAE: {overall_mae:.4f}  RMSE: {overall_rmse:.4f}  R2: {overall_r2:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(FORECAST_HORIZONS, rmse_per_step, marker="o")
    plt.xlabel("Forecast horizon")
    plt.ylabel("RMSE (scaled units)")
    plt.title("Horizon-wise RMSE (Pacific Test Set)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "horizon_error_curve.png", dpi=150)
    plt.close()
    print(f"[SAVED] {RESULTS_DIR / 'horizon_error_curve.png'}")

    print("\n--- Year-by-year breakdown ---")
    ref_years = pd.DatetimeIndex(ref_dates).year
    year_breakdown = {}
    for year in sorted(np.unique(ref_years)):
        mask = ref_years == year
        n_samples = int(mask.sum())
        if n_samples == 0:
            continue
        yr_acc = accuracy_score(y_class_test[mask], class_preds[mask])
        yr_f1 = f1_score(y_class_test[mask], class_preds[mask], average="weighted")
        yr_rmse = np.sqrt(
            mean_squared_error(y_forecast_test[mask].reshape(-1), forecast_pred[mask].reshape(-1))
        )
        year_breakdown[str(year)] = {
            "n_samples": n_samples,
            "accuracy": float(yr_acc),
            "f1_weighted": float(yr_f1),
            "forecast_rmse": float(yr_rmse),
        }
        print(f"     {year}: n={n_samples:>6}  accuracy={yr_acc:.4f}  f1={yr_f1:.4f}  forecast_rmse={yr_rmse:.4f}")

    metrics_report = {
        "checkpoint_epoch": checkpoint["epoch"],
        "test_set_size": int(len(y_class_test)),
        "classification": {
            "accuracy": float(accuracy),
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "per_class": {
                LABELS[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1_per_class[i]),
                    "support": int(support[i]),
                }
                for i in range(4)
            },
            "confusion_matrix": cm.tolist(),
        },
        "forecasting": {
            "overall_mae": float(overall_mae),
            "overall_rmse": float(overall_rmse),
            "overall_r2": float(overall_r2),
            "per_horizon": {
                FORECAST_HORIZONS[i]: {
                    "mae": float(mae_per_step[i]),
                    "rmse": float(rmse_per_step[i]),
                    "r2": float(r2_per_step[i]),
                }
                for i in range(FORECAST_STEPS)
            },
        },
        "year_by_year": year_breakdown,
    }

    with open(RESULTS_DIR / "metrics_report.json", "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"\n[SAVED] {RESULTS_DIR / 'metrics_report.json'}")

    print("\n[OK] Phase 12 evaluation complete for Pacific.")


if __name__ == "__main__":
    main()