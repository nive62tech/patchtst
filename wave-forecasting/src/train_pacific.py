"""
Phase 11 (Pacific) - Model Training

Core training loop for the PatchTST dual-head model on Pacific data. Mirrors
src/train.py (Atlantic) exactly, renamed for Pacific. Designed to be imported
both locally (quick CPU sanity check) and inside the Colab notebook
(notebooks/colab_train_pacific.ipynb) for the actual full training run on GPU.
"""

import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchtst_model import PatchTST
from src.train_config_pacific import (
    JointLoss,
    compute_class_weights,
    build_internal_validation_split,
    build_optimizer_and_scheduler,
    EarlyStopping,
    BATCH_SIZE,
    NUM_EPOCHS,
    WARMUP_EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    ALPHA,
    BETA,
    NUM_CLASSES,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "pacific" / "data"
WINDOWS_DIR = DATA_DIR / "windows"
PACIFIC_DIR = Path(__file__).resolve().parent.parent / "pacific"
CHECKPOINT_PATH = PACIFIC_DIR / "pacific_patchtst_best.pt"


def load_and_split_train_data():
    X_train = np.load(WINDOWS_DIR / "X_train.npy")
    y_class_train = np.load(WINDOWS_DIR / "y_class_train.npy")
    y_forecast_train = np.load(WINDOWS_DIR / "y_forecast_train.npy")

    n = len(y_class_train)
    fit_size, val_size = build_internal_validation_split(n)

    X_fit, X_val = X_train[:fit_size], X_train[fit_size:]
    yc_fit, yc_val = y_class_train[:fit_size], y_class_train[fit_size:]
    yf_fit, yf_val = y_forecast_train[:fit_size], y_forecast_train[fit_size:]

    return (X_fit, yc_fit, yf_fit), (X_val, yc_val, yf_val)


def make_loader(X, yc, yf, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(yc, dtype=torch.int64),
        torch.tensor(yf, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_class_loss, total_forecast_loss = 0.0, 0.0, 0.0
    all_preds, all_targets = [], []
    all_forecast_preds, all_forecast_targets = [], []
    n_batches = 0

    with torch.no_grad():
        for X, yc, yf in loader:
            X, yc, yf = X.to(device), yc.to(device), yf.to(device)
            class_logits, forecast_pred = model(X)
            loss, class_loss, forecast_loss = loss_fn(class_logits, forecast_pred, yc, yf)

            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_forecast_loss += forecast_loss.item()
            n_batches += 1

            preds = class_logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yc.cpu().numpy())
            all_forecast_preds.append(forecast_pred.cpu().numpy())
            all_forecast_targets.append(yf.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_forecast_preds = np.concatenate(all_forecast_preds)
    all_forecast_targets = np.concatenate(all_forecast_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="weighted")
    rmse = float(np.sqrt(np.mean((all_forecast_preds - all_forecast_targets) ** 2)))

    return {
        "loss": total_loss / n_batches,
        "class_loss": total_class_loss / n_batches,
        "forecast_loss": total_forecast_loss / n_batches,
        "accuracy": accuracy,
        "f1": f1,
        "rmse": rmse,
    }


def train_model(num_epochs=NUM_EPOCHS, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    (X_fit, yc_fit, yf_fit), (X_val, yc_val, yf_val) = load_and_split_train_data()
    print(f"[INFO] Fit samples: {len(yc_fit)}, Validation samples: {len(yc_val)}")

    fit_loader = make_loader(X_fit, yc_fit, yf_fit, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, yc_val, yf_val, BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(yc_fit, NUM_CLASSES).to(device)
    loss_fn = JointLoss(class_weights, alpha=ALPHA, beta=BETA)

    model = PatchTST().to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, num_epochs, WARMUP_EPOCHS, LEARNING_RATE)
    early_stopper = EarlyStopping(EARLY_STOPPING_PATIENCE)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": [], "val_rmse": []}
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for X, yc, yf in fit_loader:
            X, yc, yf = X.to(device), yc.to(device), yf.to(device)
            optimizer.zero_grad()
            class_logits, forecast_pred = model(X)
            loss, _, _ = loss_fn(class_logits, forecast_pred, yc, yf)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / n_batches

        val_metrics = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_rmse"].append(val_metrics["rmse"])

        epoch_time = time.time() - epoch_start
        if verbose:
            print(
                f"[Epoch {epoch:3d}/{num_epochs}] "
                f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  val_f1={val_metrics['f1']:.4f}  "
                f"val_rmse={val_metrics['rmse']:.4f}  ({epoch_time:.1f}s)"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            PACIFIC_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "val_rmse": val_metrics["rmse"],
                },
                CHECKPOINT_PATH,
            )
            if verbose:
                print(f"          -> New best model saved (val_loss={best_val_loss:.4f})")

        if early_stopper.step(val_metrics["loss"]):
            print(f"[INFO] Early stopping triggered at epoch {epoch}")
            break

    return history


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 11 (PACIFIC) - LOCAL SANITY CHECK (2 epochs, CPU)")
    print("=" * 60)
    print("[WARNING] This is a short local smoke test, NOT the real training run.")
    print("[WARNING] Real training (up to 100 epochs) must run in Colab on GPU.")
    result_history = train_model(num_epochs=2, device="cpu")
    print("\n[OK] Local sanity check completed - training loop runs end-to-end without errors.")
    print(f"[INFO] History: {result_history}")