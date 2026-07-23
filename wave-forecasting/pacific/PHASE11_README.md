# Pacific Phase 11 — Model Training (Colab GPU)

## What was built
- `src/train_pacific.py`: the full training loop for Pacific, mirrors `src/train.py` (Atlantic) exactly
- `notebooks/colab_train_pacific.ipynb`: Colab notebook — clones the repo, installs deps, pulls Pacific's windowed data from a dedicated Drive folder, trains on T4 GPU, plots curves, copies outputs back to Drive

## How to run
- Local sanity check: `python src\train_pacific.py`
- Real training: open `colab_train_pacific.ipynb` in Colab with a T4 GPU runtime, run top to bottom
- Outputs: `pacific\pacific_patchtst_best.pt`, `pacific\results\training_curves.png`

## Key technical decisions
- Same design as Atlantic's Phase 11 — best-checkpoint saving on every validation improvement, metrics stored alongside weights in the checkpoint
- Used a **separate Google Drive folder** (`patchtst_data/pacific_windows/` and `pacific_outputs/`) from Atlantic's, to avoid mixing up the two oceans' files during upload/download

## Result — this run
- **Best epoch: 16** (out of up to 100 allowed, early stopping patience=10)
- **Validation loss: 0.1579**
- **Validation accuracy: 96.40%**
- **Validation F1 (weighted): 96.41%**
- **Validation RMSE (forecasting, scaled units): 0.4724**
- Nearly identical performance to Atlantic (96.19% / 96.19% / 0.4704) — strong sign the architecture generalizes across oceans rather than being tuned specifically to Atlantic's data

## Files created/updated
- `src/train_pacific.py`
- `notebooks/colab_train_pacific.ipynb`
- `pacific/pacific_patchtst_best.pt` (gitignored, not committed)
- `pacific/results/training_curves.png` (committed)