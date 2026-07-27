# Validation Metrics Comparison — PatchTST vs Mamba

Computed in real units across all 20 forecast steps. Winner determined by lower RMSE (the unambiguous accuracy metric of the four).

| Ocean | Variable | PatchTST r | Mamba r | PatchTST RMSE | Mamba RMSE | Winner (RMSE) |
|---|---|---|---|---|---|---|
| Atlantic | swh | 0.9102 | 0.9748 | 0.3075 | 0.1800 | **Mamba** |
| Atlantic | mwp | 0.8972 | 0.9769 | 0.5692 | 0.2713 | **Mamba** |
| Atlantic | mwd | 0.8393 | 0.8115 | 54.2857 | 59.4123 | **PatchTST** |
| Pacific | swh | 0.9181 | 0.9826 | 0.2724 | 0.1520 | **Mamba** |
| Pacific | mwp | 0.9120 | 0.9823 | 0.6662 | 0.2994 | **Mamba** |
| Pacific | mwd | 0.8639 | 0.8545 | 56.1545 | 58.9412 | **PatchTST** |
| Bay of Bengal | swh | 0.9521 | 0.9778 | 0.2233 | 0.1754 | **Mamba** |
| Bay of Bengal | mwp | 0.9047 | 0.9753 | 0.6672 | 0.3432 | **Mamba** |
| Bay of Bengal | mwd | 0.8741 | 0.9376 | 24.4528 | 17.0343 | **Mamba** |
| Arabian Sea | swh | 0.9829 | 0.9925 | 0.1673 | 0.1613 | **Mamba** |
| Arabian Sea | mwp | 0.9259 | 0.9804 | 0.5239 | 0.2710 | **Mamba** |
| Arabian Sea | mwd | 0.8584 | 0.8657 | 45.0370 | 44.2710 | **Mamba** |

Units: swh (m), mwp (s), mwd (deg).
