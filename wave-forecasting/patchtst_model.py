"""
Phase 9 - PatchTST Architecture

Defines the PatchTST model: patch embedding -> learnable positional encoding
-> transformer encoder -> global average pooling -> two heads (classification
+ forecasting). Includes a dummy-batch smoke test to verify output shapes
before any real training happens (real training is Phase 11, on Colab GPU).
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, num_channels, patch_size, stride, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_channels = num_channels
        patch_dim = patch_size * num_channels
        self.projection = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        # x: [batch, seq_len, num_channels]
        # unfold along the time dimension -> [batch, num_patches, num_channels, patch_size]
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        patches = patches.permute(0, 1, 3, 2)  # -> [batch, num_patches, patch_size, num_channels]
        batch_size, num_patches, patch_size, num_channels = patches.shape
        patches = patches.reshape(batch_size, num_patches, patch_size * num_channels)
        embedded = self.projection(patches)  # [batch, num_patches, d_model]
        return embedded


class PatchTST(nn.Module):
    def __init__(
        self,
        num_channels=6,
        seq_len=72,
        patch_size=12,
        stride=6,
        d_model=128,
        num_layers=4,
        num_heads=8,
        ff_dim=256,
        dropout=0.1,
        num_classes=4,
        forecast_steps=20,
        forecast_channels=3,
    ):
        super().__init__()

        self.num_patches = (seq_len - patch_size) // stride + 1

        self.patch_embed = PatchEmbedding(num_channels, patch_size, stride, d_model)

        # Learnable positional encoding, one vector per patch position
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        # Classification head - raw logits, softmax applied outside (e.g. in loss/inference)
        self.classification_head = nn.Linear(d_model, num_classes)

        # Forecasting head - predicts forecast_steps * forecast_channels values, reshaped after
        self.forecast_steps = forecast_steps
        self.forecast_channels = forecast_channels
        self.forecasting_head = nn.Linear(d_model, forecast_steps * forecast_channels)

    def forward(self, x):
        # x: [batch, seq_len, num_channels]
        patches = self.patch_embed(x)  # [batch, num_patches, d_model]
        patches = patches + self.pos_embed
        patches = self.dropout(patches)

        encoded = self.encoder(patches)  # [batch, num_patches, d_model]

        pooled = encoded.mean(dim=1)  # global average pooling -> [batch, d_model]

        class_logits = self.classification_head(pooled)  # [batch, num_classes]

        forecast_flat = self.forecasting_head(pooled)  # [batch, forecast_steps*forecast_channels]
        forecast = forecast_flat.view(-1, self.forecast_steps, self.forecast_channels)  # [batch, 20, 3]

        return class_logits, forecast


def main():
    print("=" * 60)
    print("PHASE 9 - PATCHTST ARCHITECTURE")
    print("=" * 60)

    model = PatchTST(
        num_channels=6,
        seq_len=72,
        patch_size=12,
        stride=6,
        d_model=128,
        num_layers=4,
        num_heads=8,
        ff_dim=256,
        dropout=0.1,
        num_classes=4,
        forecast_steps=20,
        forecast_channels=3,
    )

    print("\n[INFO] Model config:")
    print("     num_channels=6, seq_len=72, patch_size=12, stride=6")
    print(f"     num_patches = (72-12)//6 + 1 = {model.num_patches}")
    print("     d_model=128, num_layers=4, num_heads=8, ff_dim=256, dropout=0.1")
    print("     num_classes=4, forecast_steps=20, forecast_channels=3")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    print("\n--- Dummy batch test ---")
    dummy_batch = torch.randn(64, 72, 6)
    print(f"[INFO] Dummy input shape: {dummy_batch.shape}")

    model.eval()
    with torch.no_grad():
        class_logits, forecast = model(dummy_batch)

    print(f"[OK] Classification output shape: {class_logits.shape}  (expected [64, 4])")
    print(f"[OK] Forecasting output shape:    {forecast.shape}  (expected [64, 20, 3])")

    assert class_logits.shape == (64, 4), "Classification output shape mismatch!"
    assert forecast.shape == (64, 20, 3), "Forecasting output shape mismatch!"
    print("\n[OK] Both output shapes verified correct.")

    # Quick softmax sanity check on the classification head
    probs = torch.softmax(class_logits, dim=1)
    print(f"\n[CHECK] Softmax probabilities sum per sample (first 3): {probs.sum(dim=1)[:3].tolist()}")
    print("        (should all be ~1.0)")

    print("\n[OK] PatchTST architecture verified. Ready for Phase 10 (training setup).")


if __name__ == "__main__":
    main()