"""画像をTransformerへの入力となるパッチに分割する処理。"""

import torch
from torch import nn


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) の画像を (B, N, C * P * P) に変換する。

    P は patch_size、N は (H // P) * (W // P)。
    パッチは左上から行ごとに並べ、各パッチはチャネル、行、列の順に
    平坦化する。
    """
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    B, C, H, W = images.shape
    P = patch_size

    if H % P != 0 or W % P != 0:
        raise ValueError("H and W must be divisible by patch_size.")

    # 高さ・幅を「パッチの位置」と「パッチ内の位置」に分解する。
    x = images.reshape(B, C, H // P, P, W // P, P)

    # (画像, パッチ行, パッチ列, チャネル, パッチ内の行, パッチ内の列)
    x = x.permute(0, 2, 4, 1, 3, 5)

    return x.reshape(B, (H // P) * (W // P), C * P * P)


class PatchEmbedding(nn.Module):
    """各パッチを共通の線形層で d_model 次元のトークンに変換する。"""

    def __init__(self, in_channels: int, patch_size: int, d_model: int):
        super().__init__()
        if in_channels <= 0 or patch_size <= 0 or d_model <= 0:
            raise ValueError("in_channels, patch_size and d_model must be positive.")

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.projection = nn.Linear(in_channels * patch_size**2, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, d_model)。"""
        patches = patchify(images, self.patch_size)
        if images.shape[1] != self.in_channels:
            raise ValueError("Image channels must match in_channels.")

        # 同じ重みとバイアスを全パッチに適用する。
        return self.projection(patches)
