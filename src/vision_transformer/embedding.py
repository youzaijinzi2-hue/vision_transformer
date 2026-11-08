"""ViTのEncoderに渡す、位置情報を持ったトークン列を作る。"""

import torch
from torch import nn

from vision_transformer.patch_embedding import PatchEmbedding


class VisionTransformerEmbedding(nn.Module):
    """パッチの射影 → CLS追加 → 位置Embedding加算 → Dropout。"""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, d_model)
        num_patches = (image_size // patch_size) ** 2

        # バッチ全体で共有する、学習可能なパラメータ。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 先頭のCLS用にも位置Embeddingを用意する。
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """固定サイズの画像 (B, C, H, W) → (B, N+1, D)。"""
        if images.ndim != 4:
            raise ValueError("images must have shape (B, C, H, W).")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("Image height and width must match image_size.")

        patches = self.patch_embedding(images)
        # 同じCLSを各画像の先頭に置く。expandでも勾配は元のパラメータに戻る。
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)
        return self.dropout(tokens + self.position_embedding)
