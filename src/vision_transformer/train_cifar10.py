"""CIFAR-10をダウンロードし、スクラッチ実装のViTを学習する。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vision_transformer.cifar10 import load_cifar10
from vision_transformer.model import VisionTransformer
from vision_transformer.training import evaluate, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/cifar10.pt"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")

    torch.manual_seed(args.seed)
    train, validation, test = load_cifar10(args.data_dir, args.seed, not args.no_download)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    config = dict(
        image_size=32, patch_size=4, in_channels=3, num_classes=10,
        d_model=64, num_heads=4, d_ff=256, num_layers=3, dropout=0.1,
    )
    device = torch.device(args.device)
    model = VisionTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch + 1:02d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}",
            flush=True,
        )
        # モデル選択には検証データのみを使う。
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({
                "model_config": config, "model_state_dict": model.state_dict(),
                "dataset": "cifar10", "class_names": test.classes,
                "seed": args.seed, "epoch": epoch + 1,
            }, args.output)

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, test_loader, device)
    print(f"test_loss={metrics['loss']:.4f} test_acc={metrics['accuracy']:.3f}")
    print(f"Saved best validation model: {args.output}")


if __name__ == "__main__":
    main()
