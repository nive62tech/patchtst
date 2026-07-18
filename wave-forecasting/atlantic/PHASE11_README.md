# Phase 11 — Model Training (Colab GPU)

## What was built
- `src/train.py`: the full training loop (fit/validate/checkpoint/early-stop), importable both locally for a quick CPU sanity check and inside Colab for the real run
- `notebooks/colab_train.ipynb`: end-to-end Colab notebook — clones the repo, installs deps, pulls the windowed data from Google Drive, runs training on a T4 GPU, plots training curves, copies outputs back to Drive for download

## How to run
- Local sanity check: `python src\train.py` (2 epochs, CPU, confirms no bugs before spending GPU time)
- Real training: open `colab_train.ipynb` in Google Colab with a T4 GPU runtime, run cells top to bottom
- Outputs: `atlantic\atlantic_patchtst_best.pt`, `atlantic\results\training_curves.png`

## Key technical decisions
- **Data transfer via Google Drive**, not direct browser upload or embedding data in the repo — the windowed `.npy` files total ~450MB, too large and not appropriate for git (already gitignored), and more reliable than a direct browser upload widget for files this size
- **Best-checkpoint saving on every validation-loss improvement**, not just at the end — protects against early stopping (or a Colab session drop) losing the best model found so far
- **Checkpoint stores metrics alongside weights** (`val_loss`, `val_accuracy`, `val_f1`, `val_rmse`, `epoch`) — makes the saved file self-describing, so Phase 12's evaluation script (and this README) don't need a separate log file to know what the best epoch achieved
- `training_curves.png` is committed to git (a results artifact); `atlantic_patchtst_best.pt` is not (gitignored per the `*.pt` rule — a large generated artifact, not source)

## Result — this run
- **Best epoch: 19** (out of up to 100 allowed, early stopping patience=10)
- **Validation loss: 0.1609**
- **Validation accuracy: 96.19%**
- **Validation F1 (weighted): 96.19%**
- **Validation RMSE (forecasting, scaled units): 0.4704**
- Strong classification performance on the held-out internal validation slice — a good sign heading into Phase 12's evaluation on the true test set

## Files created/updated
- `src/train.py`
- `notebooks/colab_train.ipynb`
- `atlantic/atlantic_patchtst_best.pt` (gitignored, not committed)
- `atlantic/results/training_curves.png` (committed)