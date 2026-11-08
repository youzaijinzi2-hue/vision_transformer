import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision_transformer.training import evaluate, train_one_epoch


def test_evaluation_weights_unequal_batches_and_disables_gradients():
    class CheckedLinear(nn.Linear):
        def forward(self, x):
            assert not self.training
            assert not torch.is_grad_enabled()
            return super().forward(x)

    model = CheckedLinear(2, 2)
    images = torch.tensor([[1., 0.], [0., 1.], [2., 1.]])
    labels = torch.tensor([0, 1, 0])
    with torch.no_grad():
        logits = F.linear(images, model.weight, model.bias)
    result = evaluate(model, DataLoader(TensorDataset(images, labels), batch_size=2))
    assert result["loss"] == pytest.approx(F.cross_entropy(logits, labels).item())
    assert result["accuracy"] == pytest.approx((logits.argmax(1) == labels).float().mean().item())
    assert all(p.grad is None for p in model.parameters())


def test_training_enables_training_and_updates_weights():
    model = nn.Linear(2, 2)
    model.eval()
    images = torch.tensor([[1., 0.], [0., 1.]])
    loader = DataLoader(TensorDataset(images, torch.tensor([0, 1])), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    result = train_one_epoch(model, loader, optimizer)

    assert model.training
    assert not torch.equal(before, model.weight)
    assert result["loss"] > 0
    assert 0 <= result["accuracy"] <= 1


def test_empty_loaders_are_rejected():
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="at least one sample"):
        evaluate(model, loader)
    with pytest.raises(ValueError, match="at least one sample"):
        train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1))
