"""画像をTransformerへの入力となるパッチに分割する処理。"""

import torch


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
