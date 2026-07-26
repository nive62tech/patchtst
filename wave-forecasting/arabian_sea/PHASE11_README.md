# Phase 11 — Colab GPU Training (Arabian Sea)

## What was built
`notebooks/colab_train_arabian_sea.ipynb` — clones the repo on Colab, pulls
windowed .npy data from Drive (`patchtst_data/arabian_sea_windows/`), runs
full training via `src/train_arabian_sea.py` on a T4 GPU, plots
training_curves.png, copies checkpoint + plot back to Drive
(`patchtst_data/arabian_sea_outputs/`) for local download.

## How to run
Open in Colab, Runtime > T4 GPU, run top to bottom. ~20-40 min depending on
early stopping.

## Key technical decisions
- Same architecture, loss, optimizer, and internal validation split as
  Phase 9/10 (unchanged, ocean-agnostic model).
- Early stopping patience=10 on val_loss, up to 100 epochs.

## Result — this run
Training stopped early at epoch 32. Best val_loss: 0.1373. Final
val_accuracy: 0.9624. Final val_f1: 0.9626. Final val_rmse: 0.4285
(scaled units).

## Files created/updated
- `notebooks/colab_train_arabian_sea.ipynb`
- `src/train_arabian_sea.py`
- `arabian_sea/arabian_sea_patchtst_best.pt` (gitignored)
- `arabian_sea/results/training_curves.png`
