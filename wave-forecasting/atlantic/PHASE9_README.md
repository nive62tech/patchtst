# Phase 9 — PatchTST Architecture

## What was built
- `patchtst_model.py` (project root, not `src/`): the full PatchTST model — patch embedding, learnable positional encoding, transformer encoder, global average pooling, and two heads (classification + forecasting). Includes a dummy-batch smoke test and a separate gradient-flow check.

## How to run
```powershell
Set-Location "D:\INCOIS\patchtst\wave-forecasting"
.\venv\Scripts\Activate.ps1
python patchtst_model.py
```
No file output — this is architecture definition + verification, consumed directly by Phase 11's training loop.

## Key technical decisions
- **Patch embedding via `unfold`**: rather than manually looping to slice patches, `x.unfold(dim=1, size=12, step=6)` extracts all 11 overlapping patches in one vectorized op, then a linear layer projects each flattened 72-value patch (12 timesteps × 6 channels) to `d_model=128`
- **Learnable positional encoding**: a single `nn.Parameter` of shape `[1, 11, 128]`, added to the patch embeddings before the encoder — lets the model learn patch-order information rather than using a fixed sinusoidal scheme
- **`batch_first=True`** on the transformer encoder layers, keeping tensor shapes as `[batch, seq, features]` throughout rather than PyTorch's older `[seq, batch, features]` default — avoids easy-to-miss transpose bugs
- **Global average pooling** over the 11 patch representations (not just the last patch, and not a CLS token) gives the shared representation fed to both heads — a simple, standard choice for this size of model
- **Classification head outputs raw logits**, not softmax — softmax is applied at evaluation/inference time or implicitly inside `CrossEntropyLoss` during training (Phase 10)
- **Forecasting head outputs a flat `[batch, 60]` tensor**, reshaped to `[batch, 20, 3]` — a single linear layer rather than an autoregressive decoder, matching the spec's "linear layer -> forecast" design

## Result — this run
- `num_patches = (72-12)//6 + 1 = 11`, confirmed
- **548,928 total trainable parameters** — appropriately sized for `d_model=128`
- Dummy batch `[64, 72, 6]` → classification `[64, 4]` ✓, forecast `[64, 20, 3]` ✓ — both exact matches
- Softmax probabilities sum to ~1.0 per sample, confirming the classification head is numerically sound
- **Gradient flow explicitly verified**: after a backward pass combining both heads' losses, every trainable parameter received a gradient — confirms the shared encoder genuinely connects to both heads, not just shape-compatible in isolation

## Files created/updated
- `patchtst_model.py`