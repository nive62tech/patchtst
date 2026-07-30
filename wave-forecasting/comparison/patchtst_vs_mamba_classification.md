# Classification Metrics Comparison — PatchTST vs Mamba

Sea-state classification (4 classes: Low/Moderate/High/Very High), evaluated on each ocean's held-out test set. Winner determined by accuracy.

| Ocean | PatchTST Acc | Mamba Acc | PatchTST Weighted F1 | Mamba Weighted F1 | Winner |
|---|---|---|---|---|---|
| Atlantic | 0.9639 | 0.8500 | 0.9638 | 0.8500 | **PatchTST** |
| Pacific | 0.9687 | 0.8800 | 0.9687 | 0.8800 | **PatchTST** |
| Bay of Bengal | 0.9656 | 0.8400 | 0.9656 | 0.8500 | **PatchTST** |
| Arabian Sea | 0.9694 | 0.8700 | 0.9694 | 0.8700 | **PatchTST** |

**PatchTST wins classification accuracy in all 4 oceans** (consistent with the earlier finding that PatchTST tends to edge out Mamba on the mwd/direction variable too - both point to PatchTST having some advantage on classification-adjacent/discrete signal, while Mamba leads on smooth continuous regression targets swh/mwp).

## Direction error (Mamba only, circular RMSE, t+6h)

Not directly comparable to PatchTST's mwd RMSE from the validation comparison (different metric definition - circular vs linear, single-horizon vs all-horizon). Reported separately for reference.

| Ocean | Mamba Direction cRMSE (t+6h) |
|---|---|
| Atlantic | 30.51 deg |
| Pacific | 29.31 deg |
| Bay of Bengal | 19.06 deg |
| Arabian Sea | 27.64 deg |
