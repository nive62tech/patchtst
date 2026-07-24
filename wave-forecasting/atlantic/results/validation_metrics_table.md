# Extended Validation Metrics — Atlantic

Computed in real units (inverse-transformed via scaler_atlantic.pkl), aggregated across all 20 forecast steps (t+6h to t+120h).

| Variable | Correlation Coefficient (r) | Scatter Index | Bias | RMS Error |
|---|---|---|---|---|
| swh | 0.9102 | 0.1747 | 0.0141 | 0.3075 |
| mwp | 0.8972 | 0.0732 | -0.0050 | 0.5692 |
| mwd | 0.8393 | 0.4433 | 0.9597 | 54.2857 |

Units: swh (m), mwp (s), mwd (deg).
