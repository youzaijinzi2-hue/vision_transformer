"""ダウンロード不要の縦線・横線画像で、学習処理を確認する。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from vision_transformer.model import VisionTransformer
from vision_transformer.training import evaluate, train_one_epoch


def make_stripe_dataset(num_samples: int, seed: int) -> TensorDataset:
    """0: 縦線、1: 横線。ノイズ付き16×16のグレースケール画像。"""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(num_samples) % 2
    images = torch.randn(num_samples, 1, 16, 16, generator=generator) * 0.1
    for i, label in enumerate(labels):
        if label.item() == 0:
            images[i, :, :, 6:10] += 1.0
        else:
            images[i, :, 6:10, :] += 1.0
    return TensorDataset(images, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/stripes.pt"))
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = dict(
        image_size=16, patch_size=4, in_channels=1, num_classes=2,
        d_model=32, num_heads=4, d_ff=64, num_layers=2,
    )
    model = VisionTransformer(**config).to(device)
    # 評価用は異なる乱数で生成し、学習には使わない。
    train_loader = DataLoader(
        make_stripe_dataset(256, args.seed), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        make_stripe_dataset(128, args.seed + 1), batch_size=args.batch_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train = train_one_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch + 1:02d} train_loss={train['loss']:.4f} "
            f"train_acc={train['accuracy']:.3f} val_loss={val['loss']:.4f} "
            f"val_acc={val['accuracy']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": config, "model_state_dict": model.state_dict()}, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
