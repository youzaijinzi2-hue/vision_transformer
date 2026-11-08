import pytest
import torch
from torch import nn

from vision_transformer.visualize_predictions import plot_predictions


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
