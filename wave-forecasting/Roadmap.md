# 1. What I'm Building and Why It's Impressive

**Project:** Comparative Analysis of PatchTST and Mamba for Wave Period Prediction — Atlantic Ocean

You're building the PatchTST half of a head-to-head deep learning architecture comparison — a dual-task transformer model that forecasts ocean wave behavior at a fixed Atlantic point (27.5°N, 63°W) using 26 years of hourly ERA5 reanalysis data (2000–2026), sourced from the ERA5 hourly time-series product. A teammate builds the Mamba (state-space model) counterpart on an identical data pipeline, so the two architectures can be compared fairly at the end.

Why it stands out to hiring managers:
- **Literature-grounded formula choices** — Tm01 and mean wave direction are backed by cited papers, not arbitrary picks. You can defend every variable choice in an interview, including *why you excluded* peak period (Tp).
- **True multi-task architecture** — one shared transformer representation feeding both a classification head (sea-state category) and a forecasting head (5-day-ahead multi-variable regression).
- **Correct time-series discipline** — strict chronological split, per-split windowing with zero train/test leakage, scaler fit only on training data.
- **Circular-variable handling** — sin/cos encoding applied to mean wave direction, a subtle but common ML detail.
- **Real architecture comparison, not a toy benchmark** — PatchTST vs Mamba on identical inputs/targets, done across three oceans, shows the pipeline is designed for reuse, not a one-off script.

---

# 2. Full Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.11 |
| Data wrangling | pandas, numpy |
| Scaling / metrics | scikit-learn |
| Model + training | PyTorch |
| Training compute | Google Colab (free T4 GPU) |
| Local dev | VSCode + PowerShell (CPU only — data prep, code authoring, shape/debug tests) |
| Plots | matplotlib, seaborn |
| Serialization | joblib (scaler), torch checkpoint (.pt) |
| Version control | Git + GitHub |
| Env management | Python venv (native, no conda) |

No Docker, no cloud infra, no Ollama, no Android Studio.

**Repo location:** Git repo root is `D:\INCOIS\patchtst`; this project lives inside it at `D:\INCOIS\patchtst\wave-forecasting`. All `git add`/`commit`/`push` commands run from `D:\INCOIS\patchtst`.

---

# 3. GitHub Folder Structure

```
wave-forecasting/
├── atlantic/
│   ├── data/
│   │   ├── atlantic_raw_file1.csv        (u10, v10, u100, v100, fg10)
│   │   ├── atlantic_raw_file2.csv        (mwd, mwp, swh)
│   │   ├── atlantic_merged.csv
│   │   ├── atlantic_clean.csv
│   │   ├── atlantic_features.csv
│   │   └── windows/
│   ├── scaler_atlantic.pkl
│   ├── atlantic_patchtst_best.pt
│   └── results/
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── horizon_error_curve.png
│       └── metrics_report.json
├── src/
│   ├── data_pipeline.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── labels.py
│   ├── targets.py
│   ├── split.py
│   ├── normalize.py
│   ├── windowing.py
│   ├── train_config.py
│   ├── train.py
│   ├── evaluate.py
│   └── compare.py
├── patchtst_model.py
├── notebooks/
│   └── colab_train.ipynb
├── comparison/
│   └── patchtst_vs_mamba.csv
├── docs/
│   ├── formulas.md
│   └── report.md
├── pacific/
│   └── .gitkeep
├── indian/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

`pacific/` and `indian/` start empty (placeholder `.gitkeep` files) in Phase 0. Once their data is ready, each gets built out to mirror `atlantic/` exactly, reusing the same `src/` pipeline scripts, in this same repo.

---

# 4. Phase-Wise Roadmap

**Phase 0 — Repo & Environment Setup**
- Init git repo, create full folder skeleton above
- Create empty `pacific/` and `indian/` folders with `.gitkeep` placeholders
- Create Python venv, install requirements.txt
- Set up `.gitignore` (data/, *.pt, *.pkl, venv/)
- Write `docs/formulas.md` documenting Tm01, MWD arctan formula, sin/cos rationale, with citations
- Files: `requirements.txt`, `.gitignore`, `docs/formulas.md`, `pacific/.gitkeep`, `indian/.gitkeep`, folder skeleton

**Phase 1 — Data Loading & Merging**
- Load File 1 (`u10, v10, u100, v100, fg10`) and File 2 (`mwd, mwp, swh`) — both from the ERA5 time-series product, no re-download needed
- Confirm matching `valid_time` format/frequency
- Merge on `valid_time`, inner join, sort ascending
- Drop unneeded columns (`u100, v100, fg10, latitude, longitude`)
- Save merged output — final columns: `valid_time, u10, v10, swh, mwp, mwd`
- Files: `src/data_pipeline.py` → `atlantic/data/atlantic_merged.csv`

**Phase 2 — Data Cleaning**
- Build expected full hourly timestamp range, find missing timestamps
- Count NaNs per column
- Linear-interpolate gaps ≤3 hours; drop rows for larger gaps
- Remove duplicate timestamps
- Record final row count and start/end dates
- Files: `src/data_cleaning.py` → `atlantic/data/atlantic_clean.csv`

**Phase 3 — Feature Engineering (Direction Encoding)**
- Apply sin/cos encoding to `mwd` only (`mdts`/`mdww` dropped — not available in the ERA5 time-series product)
- Drop raw `mwd` from model inputs; keep a copy for use as a forecasting target later
- Confirm final 6 input channels: `u10, v10, swh, mwp, sin_mwd, cos_mwd`
- Files: `src/feature_engineering.py` → `atlantic_features.csv`

**Phase 4 — Label Definition (Classification Target)**
- Confirm target variable is `mwp` (Tm01)
- Compute sanity-check stats against confirmed bin edges
- Apply confirmed Atlantic bin edges via `pd.cut()`
- Check class distribution for imbalance
- Files: `src/labels.py`

**Phase 5 — Forecasting Targets**
- Define forecast targets: `swh`, `mwp`, raw `mwd`
- Build 20-step future target sequences (t+6h to t+120h)
- Files: `src/targets.py`

**Phase 6 — Train/Test Split**
- Compute chronological split index and boundary date (ratio TBD — 70/30 vs 80/20 to be confirmed)
- Split into train/test by date, no shuffling
- Record row counts, date ranges
- Confirm all `mwp_class` categories present in both splits
- Files: `src/split.py`

**Phase 7 — Normalization**
- Fit `StandardScaler` on train set only
- Transform test set with the same fitted scaler
- Exclude `mwp_class` from scaling
- Save scaler
- Files: `src/normalize.py` → `atlantic/scaler_atlantic.pkl`

**Phase 8 — Sequence Windowing**
- Build 72-hour input windows, separately within train and test
- Construct `X` (shape `[samples, 72, 6]`), `y_class`, `y_forecast` arrays
- Files: `src/windowing.py` → `atlantic/data/windows/*.npy`

**Phase 9 — PatchTST Architecture**
- Implement patch splitting (patch_size=12, stride=6) and linear projection to d_model=128
- Input shape `[batch, 72, 6]`; each patch flattens to `12 × 6 = 72` raw values
- Add learnable positional encoding
- Build transformer encoder (4 layers, 8 heads, ff=256, dropout=0.1)
- Add global average pooling
- Build classification head and forecasting head
- Test with dummy batch `[64, 72, 6]`
- Files: `patchtst_model.py`

**Phase 10 — Training Setup**
- Define combined loss (CrossEntropy + MSE, α=0.5/β=0.5)
- Configure AdamW optimizer, CosineAnnealingLR with 5-epoch warmup
- Set batch size, epoch range, early stopping (patience=10)
- Carve out internal validation slice from train (last ~10%, chronological)
- Files: `src/train_config.py`

**Phase 11 — Model Training** (Colab GPU)
- Initialize model, build DataLoaders
- Run training loop with combined loss, validation each epoch
- Log per-epoch metrics
- Save best checkpoint
- Plot training curves
- Files: `notebooks/colab_train.ipynb`, `src/train.py` → `atlantic_patchtst_best.pt`, `results/training_curves.png`

**Phase 12 — Model Evaluation on Test Set**
- Load best checkpoint, run inference on test set only
- Compute classification metrics (accuracy, F1, confusion matrix, per-class)
- Compute forecasting metrics (MAE/RMSE per step, R², horizon error curve)
- Break results down year-by-year
- Files: `src/evaluate.py` → `results/confusion_matrix.png`, `results/horizon_error_curve.png`, `results/metrics_report.json`

**Phase 13 — Comparison with Mamba**
- Populate comparison table (accuracy, F1, MAE/RMSE, training time, params, inference speed)
- Files: `src/compare.py` → `comparison/patchtst_vs_mamba.csv`

**Phase 14 — Documentation & Reporting**
- Build architecture diagram comparing PatchTST and Mamba
- Write variable/formula definitions section with citations
- Note Tp was reviewed and intentionally excluded
- Include training curves, confusion matrix, horizon error curve
- Write comparative analysis of where/why each model wins
- Files: `docs/report.md`

---

# 5. Main README.md

## Change Log

- **Input channels reduced from 10 to 6**: `mdts`/`mdww` are not available in the ERA5 time-series product this project uses, so they and their sin/cos encodings were dropped. Final channels: `u10, v10, swh, mwp, sin_mwd, cos_mwd`.
- **No File 2 re-download needed**: staying on the time-series product; File 2 is just `mwd, mwp, swh` as originally downloaded.
- **Split ratio unresolved**: spec originally says 70/30, one mention said 80/20 — needs confirming before Phase 6.
- **Repo location clarified**: git repo root is `D:\INCOIS\patchtst`; `wave-forecasting` is a subfolder inside it, not the repo root itself.

That's the roadmap and README. Say **"Phase 0"** or the next phase you're ready for when set.