"""画像分類用のVision Transformer。"""

import torch
from torch import nn

from vision_transformer.embedding import VisionTransformerEmbedding
from vision_transformer.encoder import TransformerEncoder


class VisionTransformer(nn.Module):
    """画像 → Embedding → Encoder → CLSの線形分類Head。"""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.embedding = VisionTransformerEmbedding(
            image_size, patch_size, in_channels, d_model, dropout
        )
        self.encoder = TransformerEncoder(
            d_model, num_heads, d_ff, num_layers, dropout
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, num_classes) のlogits。

        CrossEntropyLossに直接渡せるよう、softmaxは適用しない。
        """
        tokens = self.embedding(images)
        encoded, _ = self.encoder(tokens)
        cls_output = encoded[:, 0]
        return self.head(cls_output)
