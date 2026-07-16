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
| 8 | Sequence Windowing | 72h input windows [72,6], per-split | Pending |
| 9 | PatchTST Architecture | Patch embedding (72-dim raw patch) + transformer + dual heads | Pending |
| 10 | Training Setup | Loss, optimizer, scheduler, val slice | Pending |
| 11 | Model Training | Full training loop on Colab GPU | Pending |
| 12 | Model Evaluation | Classification + forecasting metrics | Pending |
| 13 | Comparison with Mamba | Head-to-head results table | Pending |
| 14 | Documentation & Reporting | Diagrams, formula citations, written analysis, final report | Pending |