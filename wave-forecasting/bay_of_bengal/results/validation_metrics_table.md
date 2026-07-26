# Extended Validation Metrics — Bay of Bengal

Computed in real units (inverse-transformed via scaler_bay_of_bengal.pkl), aggregated across all 20 forecast steps (t+6h to t+120h).

| Variable | Correlation Coefficient (r) | Scatter Index | Bias | RMS Error |
|---|---|---|---|---|
| swh | 0.9521 | 0.1477 | -0.0112 | 0.2233 |
| mwp | 0.9047 | 0.0759 | -0.0020 | 0.6672 |
| mwd | 0.8741 | 0.1362 | -0.0952 | 24.4528 |

Units: swh (m), mwp (s), mwd (deg).
