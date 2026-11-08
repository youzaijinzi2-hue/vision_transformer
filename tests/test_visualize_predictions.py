import pytest
import torch
from torch import nn

from vision_transformer.visualize_predictions import plot_predictions
from vision_transformer.visualize_predictions import sample_cifar10
from unittest.mock import patch


def test_plot_contains_correct_and_incorrect_predictions(tmp_path):
    class FixedModel(nn.Module):
        def forward(self, images):
            assert not self.training
            assert not torch.is_grad_enabled()
            return torch.tensor([[4., 0.], [4., 0.], [0., 4.]])

    output = tmp_path / "plots" / "predictions.png"
    result = plot_predictions(
        FixedModel(), torch.zeros(3, 1, 4, 4), torch.tensor([0, 1, 1]),
        ["vertical", "horizontal"], output,
    )

    assert result["accuracy"] == pytest.approx(2 / 3)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("image_count,label_count", [(0, 0), (2, 1)])
def test_plot_rejects_empty_or_mismatched_samples(tmp_path, image_count, label_count):
    with pytest.raises(ValueError, match="same non-zero length"):
        plot_predictions(
            nn.Identity(), torch.zeros(image_count, 1, 4, 4),
            torch.zeros(label_count, dtype=torch.long), ["vertical", "horizontal"],
            tmp_path / "predictions.png",
        )


def test_cifar_samples_use_test_set_and_reproducible_indices(tmp_path):
    class FakeCIFAR10:
        classes = [str(i) for i in range(10)]

        def __init__(self, root, train, download, transform):
            assert root == tmp_path
            assert train is False
            assert download is False

        def __len__(self):
            return 20

        def __getitem__(self, index):
            return torch.full((3, 32, 32), index / 20), index % 10

    with patch("vision_transformer.visualize_predictions.datasets.CIFAR10", FakeCIFAR10):
        images, labels, names = sample_cifar10(tmp_path, 5, 1, False)
        repeated, repeated_labels, _ = sample_cifar10(tmp_path, 5, 1, False)
        changed, _, _ = sample_cifar10(tmp_path, 5, 2, False)
        with pytest.raises(ValueError, match="between 1 and 20"):
            sample_cifar10(tmp_path, 21, 1, False)
    assert images.shape == (5, 3, 32, 32)
    assert images[:, 0, 0, 0].unique().numel() == 5
    torch.testing.assert_close(images, repeated)
    torch.testing.assert_close(labels, repeated_labels)
    assert not torch.equal(images, changed)
    assert names == FakeCIFAR10.classes

    output = tmp_path / "rgb.png"
    plot_predictions(nn.Sequential(nn.Flatten(), nn.Linear(3072, 10)), images, labels, names, output)
    assert output.read_bytes().startswith(b"\x89PNG")
