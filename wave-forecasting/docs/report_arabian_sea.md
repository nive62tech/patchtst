# Arabian Sea — PatchTST Wave Forecasting: Documentation & Report

**Status:** PatchTST results complete. Mamba comparison (Section 7) is held pending Phase 13, which will run once all four oceans are complete - see note in that section.

---

## 1. Overview

This report documents the PatchTST half of a dual-task ocean wave forecasting model, trained and evaluated on 26 years (2000-2026) of hourly ERA5 reanalysis data at a fixed Arabian Sea point (buoy AD07: 14.904N, 69.050E). The model performs two tasks from a single shared representation:

1. **Classification** - predicts a sea-state category (Low / Moderate / High / Very High mean wave period) at the next timestep
2. **Forecasting** - predicts significant wave height (`swh`), mean wave period (`mwp`), and mean wave direction (`mwd`) for 20 future timesteps, 6-hourly out to 120 hours ahead

Arabian Sea is the fourth of four oceans in this project (after Atlantic, Pacific, and Bay of Bengal), reusing the identical pipeline structure, input channels, split boundary logic, and model architecture - with its own independently-computed classification bin edges and its own scaler, per the project's multi-ocean design.

---

## 2. Architecture

Identical to the Atlantic model - same `PatchTST` class, same hyperparameters, only the trained weights differ.

- Input: `[batch, 72, 6]` (`u10, v10, swh, mwp, sin_mwd, cos_mwd`)
- Patch embedding: patch_size=12, stride=6, 11 patches, linear projection to `d_model=128`
- Transformer encoder: 4 layers, 8 heads, `ff_dim=256`, dropout=0.1
- Global average pooling -> classification head (4 classes) + forecasting head (20x3)
- 548,928 trainable parameters (same as Atlantic/Pacific/Bay of Bengal - architecture is ocean-agnostic)

### Mamba (teammate's model)

*Held pending Phase 13, same as the other three reports - the comparison is being done once for all four oceans together.*

---

## 3. Variable Definitions & Formulas

Identical to Atlantic - see `docs/formulas.md` for full detail and citations. Tm01 (`mwp`) is the sole period definition used; Tp was reviewed and excluded. Only `mwd` gets sin/cos circular encoding (`mdts`/`mdww` are not available in the ERA5 product used, same limitation as the other three oceans).

---

## 4. Data Pipeline Summary

| Stage | Result |
|---|---|
| Raw merge | 232,344 hourly rows, 2000-01-01 to 2026-07-03 - same row count and date range as the other three oceans |
| Cleaning | 0 duplicate timestamps, 0 missing timestamps, 0 rows dropped |
| Feature engineering | `sin_mwd`/`cos_mwd` added; unit-circle check passed (deviation ~2.22e-16) |
| Labels | **Arabian Sea's own** quartile bin edges: `[4.674424, 7.0532007, 8.16263, 9.05299975, 14.895967]` - perfectly balanced 25.00% per class. Mean wave period 8.11s - notably lower than Bay of Bengal (8.81s) |
| Split | Chronological 70/30 - train 162,640 rows, test 69,704 rows - identical index split to the other three oceans |
| Normalization | `StandardScaler` fit on train only, saved as `scaler_arabian_sea.pkl` (independent from other oceans' scalers) |
| Windowing | 162,520 train windows, 69,584 test windows - identical shapes to the other three oceans |

---

## 5. Training Results

![Training Curves](../arabian_sea/results/training_curves.png)

- Best checkpoint: **epoch 22** (early stopping patience=10, training ran to epoch 32)
- Validation loss: **0.1373**
- Validation accuracy: **96.24%**, weighted F1: **96.26%**
- Validation forecast RMSE: **0.4285** (scaled units)
- Trained on Google Colab's free T4 GPU tier
- Consistent with Atlantic (96.19%), Pacific (96.40%), and Bay of Bengal (96.41%) training results - the architecture continues to generalize well across all four ocean wave climates

---

## 6. Test Set Evaluation Results

All metrics computed on the true held-out test set (69,584 windows, 2018-2026) - never used in training or the internal validation slice.

![Confusion Matrix](../arabian_sea/results/confusion_matrix.png)
![Horizon Error Curve](../arabian_sea/results/horizon_error_curve.png)

**Classification**

| Metric | Value |
|---|---|
| Accuracy | 96.94% |
| Weighted F1 | 96.94% |
| Macro F1 | 96.97% |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Low | 0.9817 | 0.9750 | 0.9783 |
| Moderate | 0.9606 | 0.9612 | 0.9609 |
| High | 0.9601 | 0.9593 | 0.9597 |
| Very High | 0.9764 | 0.9834 | 0.9799 |

**Forecasting** (overall: MAE 0.1809, RMSE 0.3790, R2 0.8484, scaled units)

| Horizon | RMSE | R2 |
|---|---|---|
| t+6h | 0.1900 | 0.9619 |
| t+24h | 0.1939 | 0.9603 |
| t+48h | 0.1937 | 0.9604 |
| t+72h | 0.2053 | 0.9555 |
| t+96h | 0.5356 | 0.6971 |
| t+120h | 0.6755 | 0.5183 |

Same expected pattern as the other three oceans - error stays low and stable through t+72h, then rises sharply from t+78h onward. Arabian Sea shows the **best long-horizon retention of all four oceans** (R2 0.52 at t+120h, vs. Bay of Bengal's 0.44, Pacific's 0.35, Atlantic's 0.27), suggesting the most persistent wave patterns at this location.

**Extended validation metrics (real units, inverse-transformed via scaler):**

| Variable | Correlation (r) | Scatter Index | Bias | RMSE |
|---|---|---|---|---|
| swh | 0.9829 | 0.1027 | -0.0019 | 0.1673 m |
| mwp | 0.9259 | 0.0647 | 0.0075 | 0.5239 s |
| mwd | 0.8584 | 0.2356 | 0.4992 | 45.0370 deg |

**Year-by-year stability:** accuracy stayed within a 96.2%-97.6% band across every year from 2018 through 2026, with no degrading trend.

---

## 7. Comparison with Mamba

**Held pending Phase 13.** Same as the other three reports - done once for all four oceans together, now that all four are complete.

---

## 8. Conclusion (Arabian Sea, PatchTST - partial)

The PatchTST model performs very strongly on the Arabian Sea dataset - 96.94% classification accuracy, the highest of all four oceans (Atlantic 96.39%, Pacific 96.87%, Bay of Bengal 96.56%), with forecast R2 holding above 0.95 through 72 hours and the best long-horizon retention of the four (R2 0.52 at t+120h). Performance is stable across the full 8-year test period, with no meaningful concept drift. With all four oceans now complete, this provides strong, consistent evidence that the PatchTST architecture transfers cleanly across distinct ocean wave climates without architecture changes - only the data-derived bin edges and scaler need to be ocean-specific. Full conclusions comparing against Mamba across all four oceans will follow once Phase 13 is complete.
