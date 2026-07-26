# Phase 11 — Colab GPU Training (Bay of Bengal)

## What was built
`notebooks/colab_train_bay_of_bengal.ipynb` — clones the repo on Colab,
pulls windowed .npy data from Drive (`patchtst_data/bay_of_bengal_windows/`),
runs full training via `src/train_bay_of_bengal.py` on a T4 GPU, plots
training_curves.png, copies checkpoint + plot back to Drive
(`patchtst_data/bay_of_bengal_outputs/`) for local download.

## How to run
Open in Colab, Runtime > T4 GPU, run top to bottom. ~20-40 min depending on
early stopping.

## Key technical decisions
- Same architecture, loss, optimizer, and internal validation split as
  Phase 9/10 (unchanged, ocean-agnostic model).
- Early stopping patience=10 on val_loss, up to 100 epochs.

## Result — this run
Training stopped early at epoch 26 (10 non-improving epochs after the best).
Best val_loss: 0.1418. Final val_accuracy: 0.9641. Final val_f1: 0.9641.
Final val_rmse: 0.4392 (scaled units).

## Files created/updated
- `notebooks/colab_train_bay_of_bengal.ipynb`
- `src/train_bay_of_bengal.py`
- `bay_of_bengal/bay_of_bengal_patchtst_best.pt` (gitignored)
- `bay_of_bengal/results/training_curves.png`
