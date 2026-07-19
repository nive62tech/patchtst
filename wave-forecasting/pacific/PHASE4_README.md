# Pacific Phase 4 — Label Definition (Fresh Bin Edges)

## What was built
- `src/labels_pacific.py`: loads `pacific_features.csv`, computes Pacific's own quartile-based `mwp_class` bin edges directly from Pacific's cleaned `mwp` distribution (does NOT reuse Atlantic's edges), applies them via `pd.cut()`, reports class distribution and imbalance, saves the file back out with the new column

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python src\labels_pacific.py
```
Output: `pacific\data\pacific_features.csv` (in place, `mwp_class` column added)

## Key technical decisions
- **Bin edges computed fresh from Pacific's own data**, not shared with Atlantic — per the original spec, wave period ranges differ by location, so each ocean gets its own quartile-based edges. The teammate's Mamba model for Pacific must use these exact same edges once confirmed here
- Same `pd.cut(..., include_lowest=True)` approach as Atlantic, same quartile method (`[0, 0.25, 0.5, 0.75, 1.0]`) — only the input distribution differs

## Confirmed Pacific bin edges (hardcoded going forward — do not recalculate)

```python
bin_edges = [5.750778, 7.5365458, 8.3817325, 9.60272075, 16.787636]
labels = ['Low', 'Moderate', 'High', 'Very High']
```
- Low = 5.750778s–7.5365458s
- Moderate = 7.5365458s–8.3817325s
- High = 8.3817325s–9.60272075s
- Very High = 9.60272075s–16.787636s

## Result — this run
- `mwp` (Tm01) range: `[5.750778, 16.787636]`, mean **8.75s**, std **1.63s** — noticeably higher mean wave period than Atlantic's (7.85s), consistent with a different ocean location
- **0 rows** out of range, **0 unassigned** `mwp_class` values
- Class distribution: **Low 58,087 (25.00%) / Moderate 58,085 (25.00%) / High 58,086 (25.00%) / Very High 58,086 (25.00%)**
- Imbalance ratio: **1.00** — no class weighting needed for the Pacific model

## Files created/updated
- `src/labels_pacific.py`
- `pacific/data/pacific_features.csv` (gitignored, not committed — now has `mwp_class` column)