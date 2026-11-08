from unittest.mock import patch

import torch
from PIL import Image
from torchvision import transforms

from vision_transformer.cifar10 import load_cifar10
from vision_transformer.model import VisionTransformer
from vision_transformer.training import evaluate, train_one_epoch
from torch.utils.data import DataLoader, TensorDataset


def test_cifar10_splits_are_disjoint_reproducible_and_use_official_test(tmp_path):
    class FakeCIFAR10:
        def __init__(self, root, train, download, transform):
            self.train = train
            self.transform = transform
            self.download = download
            assert root == tmp_path

        def __len__(self):
            return 50000 if self.train else 10000

    with patch("vision_transformer.cifar10.datasets.CIFAR10", FakeCIFAR10):
        train, val, test = load_cifar10(tmp_path, seed=7)
        repeated, _, _ = load_cifar10(tmp_path, seed=7)
        changed, _, _ = load_cifar10(tmp_path, seed=8, download=False)

    assert (len(train), len(val), len(test)) == (45000, 5000, 10000)
    assert train.dataset is val.dataset
    assert set(train.indices).isdisjoint(val.indices)
    assert set(train.indices) | set(val.indices) == set(range(50000))
    assert train.indices == repeated.indices
    assert train.indices != changed.indices
    assert train.dataset.train and not test.train
    assert train.dataset.download and not changed.dataset.download
    assert isinstance(train.dataset.transform, transforms.ToTensor)
    image = train.dataset.transform(Image.new("RGB", (32, 32), (255, 0, 128)))
    assert image.shape == (3, 32, 32)
    torch.testing.assert_close(image[:, 0, 0], torch.tensor([1., 0., 128 / 255]))


def test_cifar_shaped_batch_can_train_and_evaluate():
    model = VisionTransformer(32, 4, 3, 10, 8, 2, 16, 1)
    loader = DataLoader(TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 9])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train = train_one_epoch(model, loader, optimizer)
    test = evaluate(model, loader)

    assert train["loss"] > 0 and test["loss"] > 0
    assert 0 <= test["accuracy"] <= 1
