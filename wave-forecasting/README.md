# Comparative Analysis of PatchTST and Mamba for Wave Period Prediction — Atlantic Ocean

A dual-task deep learning pipeline that forecasts ocean wave behavior at a fixed Atlantic point (27.5°N, 63°W) using 26 years of hourly ERA5 reanalysis data (2000–2026), sourced from the ERA5 hourly time-series product. This repo covers the PatchTST half of a head-to-head architecture comparison against a teammate's Mamba model, built on an identical data pipeline so results are directly comparable.

**Input channels: 6** — `u10, v10, swh, mwp, sin_mwd, cos_mwd`. (`mdts`/`mdww` were originally planned but are not available in the ERA5 time-series product this project uses, so they were dropped along with their sin/cos encodings.)

Multi-ocean scope: Atlantic is the first of three oceans planned (Pacific and Indian to follow once their data is downloaded/confirmed). Each ocean gets its own quartile-based mwp_class bin edges calculated from that ocean's own data — bin edges are not shared across oceans, since wave period ranges differ by location. Both team members must use identical bin edges within a given ocean's dataset; cross-ocean comparison is about architecture performance (PatchTST vs Mamba), not literal severity thresholds, so oceans don't need matching bins.

Formula grounding: All core variables are backed by cited literature, documented in docs/formulas.md:
- Mean Wave Period (Tm01 = m0/m1) — used as both a forecasting target and the source variable for mwp_class. Peak period (Tp) was reviewed but intentionally excluded.
- Mean Wave Direction (θm = arctan(∬sin(θ)G(f,θ)dfdθ / ∬cos(θ)G(f,θ)dfdθ)) — the ECMWF-standard definition, compatible with ERA5.
- Circular sin/cos encoding — applied to mwd (mdts/mdww were originally planned but dropped; not available in the ERA5 time-series product this project uses).

## Why this is impressive
- Formula choices are literature-backed, not arbitrary — each has paper citations documented
- Multi-task architecture: shared representation feeding a classification head (sea-state category) and a forecasting head (5-day-ahead wave metrics)
- Correct time-series discipline throughout: strict chronological split, per-split windowing (no leakage across the train/test boundary), scaler fit only on train data
- Circular-variable handling consistently applied across three direction variables, not just one

## Tech Stack

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

## Folder Structure

```
wave-forecasting/
├── atlantic/
│   ├── data/
│   │   ├── atlantic_raw_file1.csv        (u10, v10, u100, v100, fg10)
│   │   ├── atlantic_raw_file2.csv        (mwd, mwp, swh)
│   │   ├── atlantic_merged.csv
│   │   ├── atlantic_clean.csv
│   │   ├── atlantic_features.csv
│   │   ├── atlantic_train.csv, atlantic_test.csv
│   │   ├── atlantic_train_scaled.csv, atlantic_test_scaled.csv
│   │   └── windows/
│   │       ├── X_train.npy, y_class_train.npy, y_forecast_train.npy
│   │       └── X_test.npy, y_class_test.npy, y_forecast_test.npy
│   ├── scaler_atlantic.pkl
│   ├── atlantic_patchtst_best.pt
│   ├── PHASE1_README.md ... PHASE12_README.md
│   └── results/
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── horizon_error_curve.png
│       └── metrics_report.json
├── pacific/
│   ├── data/
│   │   ├── pacific_raw_file1.csv         (u10, v10, u100, v100, fg10)
│   │   ├── pacific_raw_file2.csv         (mwd, mwp, swh)
│   │   ├── pacific_merged.csv
│   │   ├── pacific_clean.csv
│   │   ├── pacific_features.csv
│   │   ├── pacific_train.csv, pacific_test.csv
│   │   ├── pacific_train_scaled.csv, pacific_test_scaled.csv
│   │   └── windows/
│   ├── scaler_pacific.pkl
│   ├── pacific_patchtst_best.pt
│   └── results/
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── horizon_error_curve.png
│       └── metrics_report.json
├── src/
│   ├── data_pipeline.py       (ocean-parameterized)
│   ├── data_cleaning.py       (ocean-parameterized)
│   ├── feature_engineering.py (ocean-parameterized)
│   ├── labels.py               (ocean-parameterized)
│   ├── targets.py
│   ├── split.py                (ocean-parameterized)
│   ├── normalize.py            (ocean-parameterized)
│   ├── windowing.py            (ocean-parameterized)
│   ├── train_config.py
│   ├── train.py                (ocean-parameterized)
│   ├── evaluate.py             (ocean-parameterized)
│   └── compare.py
├── patchtst_model.py
├── notebooks/
│   └── colab_train.ipynb
├── comparison/
│   └── patchtst_vs_mamba.csv
├── docs/
│   ├── formulas.md
│   └── report.md
├── indian/
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

`pacific/` is now built out to mirror `atlantic/` exactly — same subfolder structure, own independently-computed `mwp_class` bin edges, own scaler, own checkpoint. `indian/` remains a placeholder until its location is confirmed. The `src/` scripts marked "ocean-parameterized" get refactored in Pacific's Phase 0 to accept an ocean name rather than being hardcoded to Atlantic paths — `targets.py`, `train_config.py`, and `patchtst_model.py`/`compare.py` don't need this since they're already ocean-agnostic (pure logic/architecture, no hardcoded ocean paths).

## Phase Progress

| Phase | Name | Covers | Status |
|---|---|---|---|
| 0 | Repo & Environment Setup | Git init, venv, folder skeleton (incl. empty pacific/ and indian/ placeholders), requirements, formulas.md | Pending |
| 1 | Data Loading & Merging | Load + merge raw CSVs (mwd, mwp, swh — no mdts/mdww, not in time-series product) | Complete |
| 2 | Data Cleaning | Gap detection, interpolation, dedup | Complete |
| 3 | Feature Engineering | sin/cos encoding on mwd → 6 input channels | Complete |
| 4 | Label Definition | mwp_class bins (Atlantic edges confirmed via quartile split) | Complete |
| 5 | Forecasting Targets | 20-step future target construction (swh, mwp, mwd) | Complete |
| 6 | Train/Test Split | Chronological 70/30 split | Complete |
| 7 | Normalization | Train-only StandardScaler fit/apply | Complete |
| 8 | Sequence Windowing | 72h input windows [72,6], per-split | Complete |
| 9 | PatchTST Architecture | Patch embedding (72-dim raw patch) + transformer + dual heads | Complete |
| 10 | Training Setup | Loss, optimizer, scheduler, val slice | Complete |
| 11 | Model Training | Full training loop on Colab GPU | Complete |
| 12 | Model Evaluation | Classification + forecasting metrics | Complete |
| 13 | Comparison with Mamba | Head-to-head results table | Held (batched across all 3 oceans) |
| 14 | Documentation & Reporting | Diagrams, formula citations, written analysis, final report | Complete (Atlantic; Mamba section pending Phase 13) |

## Pacific Phase Progress

Reuses the same pipeline and phase structure as Atlantic. Own independently-computed `mwp_class` bin edges (not shared with Atlantic). Location: **TBD — needed before Phase 1 can run.**

| Phase | Name | Covers | Status |
|---|---|---|---|
| 0 | Pacific Setup | Create pacific/data + pacific/results skeleton (dedicated per-ocean scripts used instead of a shared parameterized refactor) | Complete |
| 1 | Data Loading & Merging | Load + merge Pacific raw CSVs | Complete |
| 2 | Data Cleaning | Gap detection, interpolation, dedup | Complete |
| 3 | Feature Engineering | sin/cos encoding on mwd → 6 input channels | Complete |
| 4 | Label Definition | mwp_class bins (Pacific edges confirmed via quartile split: [5.750778, 7.5365458, 8.3817325, 9.60272075, 16.787636]) | Complete |
| 5 | Forecasting Targets | 20-step future target construction (swh, mwp, mwd) | Complete |
| 6 | Train/Test Split | Chronological 70/30 split | Complete |
| 7 | Normalization | Train-only StandardScaler fit/apply, own scaler_pacific.pkl | Complete |
| 8 | Sequence Windowing | 72h input windows [72,6], per-split | Complete |
| 9 | PatchTST Architecture | Same architecture, new training run | Complete |
| 10 | Training Setup | Same loss/optimizer/scheduler config | Pending |
| 11 | Model Training | Full training loop on Colab GPU | Pending |
| 12 | Model Evaluation | Classification + forecasting metrics | Pending |
| 13 | Comparison with Mamba | Held (batched across all 3 oceans) | Held |
| 14 | Documentation & Reporting | Pacific section of docs/report.md | Pending |