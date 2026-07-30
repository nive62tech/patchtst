# Validation Metrics Comparison — PatchTST vs Mamba

Mamba numbers from the teammate's re-trained models. Computed in real units across all 20 forecast steps. Winner determined by lower RMSE.

| Ocean | Variable | PatchTST r | Mamba r | PatchTST RMSE | Mamba RMSE | Winner (RMSE) |
|---|---|---|---|---|---|---|
| Atlantic | swh | 0.9102 | 0.9787 | 0.3075 | 0.1716 | **Mamba** |
| Atlantic | mwp | 0.8972 | 0.9760 | 0.5692 | 0.3413 | **Mamba** |
| Atlantic | mwd | 0.8393 | 0.9279 | 54.2857 | 18.3805 | **Mamba** |
| Pacific | swh | 0.9181 | 0.9811 | 0.2724 | 0.1539 | **Mamba** |
| Pacific | mwp | 0.9120 | 0.9813 | 0.6662 | 0.3177 | **Mamba** |
| Pacific | mwd | 0.8639 | 0.8492 | 56.1545 | 60.2926 | **PatchTST** |
| Bay of Bengal | swh | 0.9521 | 0.9750 | 0.2233 | 0.1833 | **Mamba** |
| Bay of Bengal | mwp | 0.9047 | 0.9762 | 0.6672 | 0.2760 | **Mamba** |
| Bay of Bengal | mwd | 0.8741 | 0.8094 | 24.4528 | 60.0275 | **PatchTST** |
| Arabian Sea | swh | 0.9829 | 0.9918 | 0.1673 | 0.1561 | **Mamba** |
| Arabian Sea | mwp | 0.9259 | 0.9810 | 0.5239 | 0.2667 | **Mamba** |
| Arabian Sea | mwd | 0.8584 | 0.8454 | 45.0370 | 47.8113 | **PatchTST** |

Units: swh (m), mwp (s), mwd (deg).
