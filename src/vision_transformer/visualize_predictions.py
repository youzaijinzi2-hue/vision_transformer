"""保存済みViTの予測と正解を、画像と一緒にPNGに描く。"""

import argparse
import math
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn
from torchvision import datasets, transforms

from vision_transformer.model import VisionTransformer
from vision_transformer.train import make_stripe_dataset


@torch.no_grad()
def plot_predictions(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    output: Path,
) -> dict[str, float]:
    """渡された全画像を描画する。緑は正解、赤は誤分類。"""
    if len(images) == 0 or len(images) != len(labels):
        raise ValueError("images and labels must have the same non-zero length.")
    model.eval()
    probabilities = model(images).softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    images, labels = images.cpu(), labels.cpu()
    predictions, confidence = predictions.cpu(), confidence.cpu()
    correct = predictions == labels
    accuracy = correct.float().mean().item()

    columns = min(4, len(images))
    rows = math.ceil(len(images) / columns)
    figure = Figure(figsize=(columns * 3, rows * 3), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= len(images):
            continue
        image = images[index]
        if image.shape[0] == 1:
            axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
        axis.set_title(
            f"True: {class_names[labels[index].item()]}\n"
            f"Pred: {class_names[predictions[index].item()]} "
            f"({confidence[index].item():.1%})",
            color="green" if correct[index].item() else "red",
        )
    figure.suptitle(
        f"Displayed samples: {correct.sum().item()}/{len(images)} correct "
        f"({accuracy:.1%})\nGreen = correct, Red = incorrect"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    return {"accuracy": accuracy}


def sample_cifar10(root: Path, num_samples: int, seed: int, download: bool):
    """公式テストデータから重複なしで無作為抽出する。"""
    dataset = datasets.CIFAR10(
        root=root, train=False, download=download, transform=transforms.ToTensor()
    )
    if not 0 < num_samples <= len(dataset):
        raise ValueError(f"num-samples must be between 1 and {len(dataset)}.")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    samples = [dataset[i.item()] for i in indices[:num_samples]]
    images = torch.stack([image for image, _ in samples])
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    return images, labels, dataset.classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/stripes.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.png"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("num-samples must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    dataset_name = checkpoint.get("dataset", "stripes")
    if dataset_name == "cifar10":
        images, labels, class_names = sample_cifar10(
            args.data_dir, args.num_samples, args.seed, not args.no_download
        )
    elif dataset_name == "stripes":
        if any(config[key] != value for key, value in {
            "image_size": 16, "in_channels": 1, "num_classes": 2,
        }.items()):
            parser.error("Stripe model requires 16x16 grayscale images and two classes.")
        images, labels = make_stripe_dataset(args.num_samples, args.seed).tensors
        class_names = ["vertical", "horizontal"]
    else:
        parser.error(f"Unsupported dataset: {dataset_name}")
    model = VisionTransformer(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = plot_predictions(
        model, images, labels, class_names, args.output
    )
    print(f"Displayed accuracy: {metrics['accuracy']:.1%}; Saved: {args.output}")


if __name__ == "__main__":
    main()
