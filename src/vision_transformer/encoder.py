"""ViT用のPre-LN Transformer Encoderブロック。"""

import torch
from torch import nn

from vision_transformer.attention import MultiHeadAttention
from vision_transformer.normalization import LayerNormalization


class FeedForward(nn.Module):
    """各トークンに共通のMLP: D → d_ff → D。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive.")
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderBlock(nn.Module):
    """NormをAttention/MLPの前に置く。入出力はともに (B, N, D)。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # CLSを含む全トークンが互いを参照するため、causal maskは使わない。
        attention_output, weights = self.attention(self.norm1(x))
        x = x + self.dropout1(attention_output)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x, weights


class TransformerEncoder(nn.Module):
    """独立したL層のPre-LNブロックと、最後のLayerNorm。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        attention_maps = []
        for layer in self.layers:
            x, weights = layer(x)
            attention_maps.append(weights)
        return self.norm(x), attention_maps
