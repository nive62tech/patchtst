"""
Phase 15 (Atlantic) - Combined Results Dashboard

Builds a single 6-panel summary figure:
  1. Joint Loss Curves - reuses the already-saved training_curves.png image
     (per-epoch loss history was not saved as data during Phase 11, only the
     rendered plot, so this panel embeds that existing image rather than
     fabricating numbers)
  2. MWP: Predicted vs Actual (t+6h, first 200 test samples, real units)
  3. SWH: Predicted vs Actual (t+6h, first 200 test samples, real units)
  4. Horizon-wise RMSE for MWP and SWH (real units, all 20 forecast steps)
  5. Sea State Confusion Matrix (test set)
  6. MWP Scatter plot (predicted vs actual, t+6h, with 1:1 line and RMSE)

Saves: atlantic/results/results_dashboard.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchtst_model import PatchTST

ATLANTIC_DIR = Path(__file__).resolve().parent.parent / "atlantic"
DATA_DIR = ATLANTIC_DIR / "data"
WINDOWS_DIR = DATA_DIR / "windows"
RESULTS_DIR = ATLANTIC_DIR / "results"
CHECKPOINT_PATH = ATLANTIC_DIR / "atlantic_patchtst_best.pt"
SCALER_PATH = ATLANTIC_DIR / "scaler_atlantic.pkl"
EXISTING_TRAINING_CURVES = RESULTS_DIR / "training_curves.png"

SCALE_COLS = ["u10", "v10", "swh", "mwp", "sin_mwd", "cos_mwd", "mwd"]
TARGET_COLS = ["swh", "mwp", "mwd"]
LABELS = ["Low", "Moderate", "High", "Very High"]
FORECAST_HORIZONS = [f"t+{6 * (i + 1)}h" for i in range(20)]
BATCH_SIZE = 256


def run_inference(model, X, batch_size=BATCH_SIZE, device="cpu"):
    model.eval()
    all_class_logits, all_forecasts = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            class_logits, forecast = model(batch)
            all_class_logits.append(class_logits.cpu().numpy())
            all_forecasts.append(forecast.cpu().numpy())
    return np.concatenate(all_class_logits), np.concatenate(all_forecasts)


def inv(values, mean, scale):
    return values * scale + mean


def main():
    print("=" * 60)
    print("PHASE 15 (ATLANTIC) - COMBINED RESULTS DASHBOARD")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = PatchTST().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[OK] Loaded checkpoint from epoch {checkpoint['epoch']}")

    scaler = joblib.load(SCALER_PATH)
    swh_idx, mwp_idx, mwd_idx = (SCALE_COLS.index(c) for c in ["swh", "mwp", "mwd"])

    X_test = np.load(WINDOWS_DIR / "X_test.npy")
    y_class_test = np.load(WINDOWS_DIR / "y_class_test.npy")
    y_forecast_test = np.load(WINDOWS_DIR / "y_forecast_test.npy")

    print("\n--- Running inference ---")
    class_logits, forecast_pred = run_inference(model, X_test, device=device)
    class_preds = class_logits.argmax(axis=1)
    print("[OK] Inference complete")

    # Inverse-transform MWP and SWH to real units, all horizons
    mwp_pred_real = inv(forecast_pred[:, :, 1], scaler.mean_[mwp_idx], scaler.scale_[mwp_idx])
    mwp_actual_real = inv(y_forecast_test[:, :, 1], scaler.mean_[mwp_idx], scaler.scale_[mwp_idx])
    swh_pred_real = inv(forecast_pred[:, :, 0], scaler.mean_[swh_idx], scaler.scale_[swh_idx])
    swh_actual_real = inv(y_forecast_test[:, :, 0], scaler.mean_[swh_idx], scaler.scale_[swh_idx])

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Atlantic - PatchTST Evaluation Results", fontsize=15, fontweight="bold")

    # --- Panel 1: existing training curves image, embedded as-is ---
    ax1 = fig.add_subplot(2, 3, 1)
    if EXISTING_TRAINING_CURVES.exists():
        img = mpimg.imread(EXISTING_TRAINING_CURVES)
        ax1.imshow(img)
        ax1.axis("off")
        ax1.set_title("Joint Loss Curves (from Phase 11)")
    else:
        ax1.text(0.5, 0.5, "training_curves.png not found", ha="center", va="center")
        ax1.axis("off")

    # --- Panel 2: MWP predicted vs actual, t+6h, first 200 samples ---
    ax2 = fig.add_subplot(2, 3, 2)
    n_show = 200
    ax2.plot(mwp_actual_real[:n_show, 0], label="Actual MWP", color="tab:blue")
    ax2.plot(mwp_pred_real[:n_show, 0], label="Predicted MWP", color="tab:orange", linestyle="--")
    ax2.set_xlabel("Test sample")
    ax2.set_ylabel("MWP (s)")
    ax2.set_title("MWP: Predicted vs Actual (t+6h)")
    ax2.legend()

    # --- Panel 3: SWH predicted vs actual, t+6h, first 200 samples ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(swh_actual_real[:n_show, 0], label="Actual SWH", color="tab:blue")
    ax3.plot(swh_pred_real[:n_show, 0], label="Predicted SWH", color="tab:green", linestyle="--")
    ax3.set_xlabel("Test sample")
    ax3.set_ylabel("SWH (m)")
    ax3.set_title("SWH: Predicted vs Actual (t+6h)")
    ax3.legend()

    # --- Panel 4: Horizon-wise RMSE for MWP and SWH ---
    ax4 = fig.add_subplot(2, 3, 4)
    mwp_rmse_per_h = [
        np.sqrt(mean_squared_error(mwp_actual_real[:, h], mwp_pred_real[:, h])) for h in range(20)
    ]
    swh_rmse_per_h = [
        np.sqrt(mean_squared_error(swh_actual_real[:, h], swh_pred_real[:, h])) for h in range(20)
    ]
    ax4.plot(FORECAST_HORIZONS, mwp_rmse_per_h, marker="o", label="PatchTST MWP", color="tab:blue")
    ax4.plot(FORECAST_HORIZONS, swh_rmse_per_h, marker="s", label="PatchTST SWH", color="tab:green")
    ax4.set_xlabel("Forecast horizon")
    ax4.set_ylabel("RMSE (real units)")
    ax4.set_title("Horizon-wise RMSE")
    ax4.set_xticklabels(FORECAST_HORIZONS, rotation=45)
    ax4.legend()

    # --- Panel 5: Confusion matrix ---
    ax5 = fig.add_subplot(2, 3, 5)
    cm = confusion_matrix(y_class_test, class_preds, labels=[0, 1, 2, 3])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, ax=ax5)
    ax5.set_xlabel("Predicted")
    ax5.set_ylabel("True")
    ax5.set_title("Sea State Confusion Matrix")

    # --- Panel 6: MWP scatter, t+6h ---
    ax6 = fig.add_subplot(2, 3, 6)
    rmse_mwp_t6 = np.sqrt(mean_squared_error(mwp_actual_real[:, 0], mwp_pred_real[:, 0]))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(mwp_actual_real), size=min(5000, len(mwp_actual_real)), replace=False)
    ax6.scatter(
        mwp_actual_real[sample_idx, 0], mwp_pred_real[sample_idx, 0], alpha=0.3, s=8, color="tab:blue"
    )
    lims = [
        min(mwp_actual_real[:, 0].min(), mwp_pred_real[:, 0].min()),
        max(mwp_actual_real[:, 0].max(), mwp_pred_real[:, 0].max()),
    ]
    ax6.plot(lims, lims, "r--", label="1:1 line")
    ax6.set_xlabel("Actual MWP (s)")
    ax6.set_ylabel("Predicted MWP (s)")
    ax6.set_title(f"MWP Scatter — RMSE={rmse_mwp_t6:.3f}s")
    ax6.legend()

    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "results_dashboard.png", dpi=150)
    print(f"\n[SAVED] {RESULTS_DIR / 'results_dashboard.png'}")


if __name__ == "__main__":
    main()