"""torchvisionによるCIFAR-10の読み込み。"""

from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def load_cifar10(root: str | Path = "data", seed: int = 0, download: bool = True):
    """学習45,000枚・検証5,000枚・テスト10,000枚を返す。

    ToTensorでRGB画像を (3, 32, 32)、画素値0〜1に変換する。
    """
    transform = transforms.ToTensor()
    full_train = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
    test = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    train, validation = random_split(
        full_train, [45000, 5000], generator=torch.Generator().manual_seed(seed)
    )
    return train, validation, test
