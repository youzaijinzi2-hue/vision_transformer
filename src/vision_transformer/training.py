"""画像分類の1エポック分の学習と評価。"""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        count += labels.numel()
        total_loss += loss.item() * labels.numel()
        correct += (logits.detach().argmax(dim=1) == labels).sum().item()
    if count == 0:
        raise ValueError("loader must contain at least one sample.")
    return {"loss": total_loss / count, "accuracy": correct / count}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        count += labels.numel()
        total_loss += criterion(logits, labels).item() * labels.numel()
        correct += (logits.argmax(dim=1) == labels).sum().item()
    if count == 0:
        raise ValueError("loader must contain at least one sample.")
    return {"loss": total_loss / count, "accuracy": correct / count}
