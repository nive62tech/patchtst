"""
Phase 10 - Training Setup

Defines all training configuration for Phase 11's Colab training loop:
- Combined loss (weighted CrossEntropy + MSE, alpha=0.5/beta=0.5)
- AdamW optimizer, lr=1e-4
- CosineAnnealingLR scheduler with a 5-epoch linear warmup
- Batch size, epoch range, early stopping (patience=10)
- Internal validation slice: last ~10% of the TRAIN windows, chronologically
  (still "train" overall - just held out from weight updates, used only for
  early stopping and per-epoch monitoring)

This script only defines/verifies the setup - it does NOT run full training
(that happens in Phase 11, on Colab GPU). It does run a small local smoke
test: one forward+backward+optimizer step+scheduler step, on CPU, using a
tiny slice of real windowed data, to confirm everything is wired together
correctly before moving to Colab.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Ensure the project root (parent of src/) is on sys.path, so patchtst_model
# can be found regardless of how/where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from patchtst_model import PatchTST

DATA_DIR = Path(__file__).resolve().parent.parent / "atlantic" / "data"
WINDOWS_DIR = DATA_DIR / "windows"

# ---- Training hyperparameters (per confirmed spec) ----
BATCH_SIZE = 64
NUM_EPOCHS = 100  # upper bound - early stopping will likely stop sooner
WARMUP_EPOCHS = 5
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 10
ALPHA = 0.5  # classification loss weight
BETA = 0.5  # forecasting loss weight
VAL_FRACTION = 0.10  # last 10% of train, chronologically

NUM_CLASSES = 4
LABELS = ["Low", "Moderate", "High", "Very High"]


def compute_class_weights(y_class: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss. Balanced data
    (like this project's, confirmed in Phase 4/8) will produce weights
    very close to 1.0 for every class - this is general-purpose/defensive,
    not because imbalance is actually expected here.
    """
    counts = np.bincount(y_class, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero for any missing class
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class JointLoss(nn.Module):
    """Combined loss: alpha * CrossEntropyLoss(class) + beta * MSELoss(forecast)."""

    def __init__(self, class_weights: torch.Tensor, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.forecast_loss_fn = nn.MSELoss()

    def forward(self, class_logits, forecast_pred, class_target, forecast_target):
        class_loss = self.class_loss_fn(class_logits, class_target)
        forecast_loss = self.forecast_loss_fn(forecast_pred, forecast_target)
        total_loss = self.alpha * class_loss + self.beta * forecast_loss
        return total_loss, class_loss, forecast_loss


def build_internal_validation_split(n_train_samples: int, val_fraction: float = VAL_FRACTION):
    """
    Splits the TRAIN windows chronologically: first (1-val_fraction) stays as
    the actual fit set (used for weight updates), last val_fraction becomes
    an internal validation slice (used only for early stopping / per-epoch
    monitoring). Both are still "train" in the Phase 6 sense - the true
    held-out test set from Phase 6/8 is never touched here.
    """
    val_size = int(n_train_samples * val_fraction)
    fit_size = n_train_samples - val_size
    return fit_size, val_size


def build_optimizer_and_scheduler(model: nn.Module, num_epochs: int, warmup_epochs: int, lr: float):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs - warmup_epochs, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    return optimizer, scheduler


class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def main():
    print("=" * 60)
    print("PHASE 10 - TRAINING SETUP")
    print("=" * 60)

    if not (WINDOWS_DIR / "y_class_train.npy").exists():
        raise FileNotFoundError("Phase 8 windowed arrays not found - run Phase 8 first.")

    y_class_train_full = np.load(WINDOWS_DIR / "y_class_train.npy")
    n_train_samples = len(y_class_train_full)

    print(f"\n[INFO] Total train windows (from Phase 8): {n_train_samples}")

    # ---- Internal validation split ----
    fit_size, val_size = build_internal_validation_split(n_train_samples)
    print(f"\n--- Internal validation split (chronological, last {VAL_FRACTION * 100:.0f}%) ---")
    print(f"[INFO] Fit (actual training) samples: {fit_size}")
    print(f"[INFO] Internal validation samples:    {val_size}")
    print("[INFO] Both remain part of 'train' overall - Phase 6/8's true test set is untouched")

    # ---- Class weights (computed on the FIT portion only, not val or test) ----
    y_class_fit = y_class_train_full[:fit_size]
    class_weights = compute_class_weights(y_class_fit, NUM_CLASSES)
    print("\n--- Class weights (computed on fit portion, for CrossEntropyLoss) ---")
    for label, w in zip(LABELS, class_weights.tolist()):
        print(f"     {label:12s}: weight={w:.4f}")

    # ---- Loss ----
    loss_fn = JointLoss(class_weights, alpha=ALPHA, beta=BETA)
    print(f"\n[INFO] Combined loss: {ALPHA} * CrossEntropyLoss(weighted) + {BETA} * MSELoss")

    # ---- Model, optimizer, scheduler ----
    model = PatchTST()
    optimizer, scheduler = build_optimizer_and_scheduler(model, NUM_EPOCHS, WARMUP_EPOCHS, LEARNING_RATE)
    early_stopper = EarlyStopping(EARLY_STOPPING_PATIENCE)

    print(f"\n[INFO] Optimizer: AdamW, lr={LEARNING_RATE}")
    print(f"[INFO] Scheduler: LinearLR warmup ({WARMUP_EPOCHS} epochs) -> CosineAnnealingLR")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    print(f"[INFO] Epoch range: up to {NUM_EPOCHS} (early stopping patience={EARLY_STOPPING_PATIENCE})")

    # ---- LR schedule preview across a few epochs ----
    print("\n--- LR schedule preview (first 8 epochs) ---")
    for epoch in range(8):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"     epoch {epoch + 1}: lr={lr_now:.8f}")
        scheduler.step()

    # ---- Local smoke test: one real forward+backward+step, on a tiny batch of REAL data ----
    print("\n--- Local smoke test (CPU, tiny batch, real data - not full training) ---")
    X_train = np.load(WINDOWS_DIR / "X_train.npy")
    y_forecast_train = np.load(WINDOWS_DIR / "y_forecast_train.npy")

    batch_X = torch.tensor(X_train[:BATCH_SIZE], dtype=torch.float32)
    batch_y_class = torch.tensor(y_class_train_full[:BATCH_SIZE], dtype=torch.int64)
    batch_y_forecast = torch.tensor(y_forecast_train[:BATCH_SIZE], dtype=torch.float32)

    model.train()
    optimizer.zero_grad()
    class_logits, forecast_pred = model(batch_X)
    total_loss, class_loss, forecast_loss = loss_fn(
        class_logits, forecast_pred, batch_y_class, batch_y_forecast
    )
    total_loss.backward()
    optimizer.step()

    print(f"[OK] Total loss:     {total_loss.item():.4f}")
    print(f"[OK] Class loss:     {class_loss.item():.4f}")
    print(f"[OK] Forecast loss:  {forecast_loss.item():.4f}")
    print("[OK] One optimizer step completed successfully on real data")

    # ---- Early stopping logic smoke test ----
    print("\n--- Early stopping logic smoke test ---")
    dummy_val_losses = [0.9, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    for i, vl in enumerate(dummy_val_losses):
        stopped = stopper.step(vl)
        print(
            f"     step {i + 1}: val_loss={vl}  best={stopper.best_loss}  "
            f"counter={stopper.counter}  stop={stopped}"
        )
        if stopped:
            print(f"[OK] Early stopping triggered correctly at step {i + 1}")
            break

    print("\n[OK] Training setup verified. Ready for Phase 11 (actual training on Colab GPU).")


if __name__ == "__main__":
    main()