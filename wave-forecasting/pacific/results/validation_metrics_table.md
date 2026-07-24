# Extended Validation Metrics — Pacific

Computed in real units (inverse-transformed via scaler_pacific.pkl), aggregated across all 20 forecast steps (t+6h to t+120h).

| Variable | Correlation Coefficient (r) | Scatter Index | Bias | RMS Error |
|---|---|---|---|---|
| swh | 0.9181 | 0.1218 | -0.0102 | 0.2724 |
| mwp | 0.9120 | 0.0768 | -0.0045 | 0.6662 |
| mwd | 0.8639 | 0.4239 | 0.7182 | 56.1545 |

Units: swh (m), mwp (s), mwd (deg).
