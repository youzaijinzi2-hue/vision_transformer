"""Position embedding initialization/scaling ablation; reuses existing training loops."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Subset

from vision_transformer.model import VisionTransformer
from vision_transformer.train import make_stripe_dataset
from vision_transformer.training import evaluate, train_one_epoch

LABELS = {
    "A": "Baseline", "B": "Large initialization", "C": "Proposed scaling",
    "D": "Small positional embedding", "E": "Scaled baseline",
    "F": "Intermediate scaling",
}


def conditions(d_model: int) -> dict[str, dict[str, float]]:
    root = math.sqrt(d_model)
    pairs = [(0.02, 1.0), (1.0, 1.0), (1 / root, root),
             (1 / root, 1.0), (0.02, root), (1 / root, d_model ** 0.25)]
    return {key: {"position_init_std": std, "position_scale": scale}
            for key, (std, scale) in zip(LABELS, pairs)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def initial_statistics(model: VisionTransformer, images: torch.Tensor) -> dict[str, float]:
    """Population std over all elements, mean token L2; excludes CLS in both."""
    patches = model.embedding.patch_embedding(images)
    positions = model.embedding.position_scale * model.embedding.position_embedding[:, 1:]
    patch_std = patches.std(correction=0).item()
    position_std = positions.std(correction=0).item()
    if patch_std == 0:
        raise ValueError("Patch std is zero; position/patch ratio is undefined")
    return {
        "patch_std": patch_std, "scaled_position_std": position_std,
        "position_patch_std_ratio": position_std / patch_std,
        "patch_mean_l2": patches.norm(dim=-1).mean().item(),
        "scaled_position_mean_l2": positions.norm(dim=-1).mean().item(),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def plot_results(output: Path) -> None:
    """Replot saved results without training; shaded areas are seed population std."""
    results = json.loads((output / "results.json").read_text())
    grouped = {key: [r for r in results if r["condition"] == key] for key in LABELS}

    def figure():
        fig = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(fig)
        return fig, fig.subplots()

    for metric, ylabel in [("train_loss", "Training cross-entropy loss"),
                           ("val_loss", "Validation cross-entropy loss"),
                           ("val_acc", "Validation accuracy (%)")]:
        fig, ax = figure()
        for key, runs in grouped.items():
            values = np.array([[row[metric] for row in run["history"]] for run in runs])
            if metric == "val_acc":
                values *= 100
            epochs = np.arange(1, values.shape[1] + 1)
            mean, std = values.mean(axis=0), values.std(axis=0)
            line, = ax.plot(epochs, mean, marker=".", label=f"{key}: {LABELS[key]}")
            ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.15)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(output / f"{metric}.png", dpi=150)

    summary = []
    for key, runs in grouped.items():
        values = [r["final_val_acc"] for r in runs]
        summary.append({"condition": key, "label": LABELS[key], "num_seeds": len(values),
                        "final_val_acc_mean": float(np.mean(values)),
                        "final_val_acc_std": float(np.std(values))})
    write_json(output / "summary.json", summary)
    fig, ax = figure()
    ax.bar(list(grouped), [s["final_val_acc_mean"] * 100 for s in summary],
           yerr=[s["final_val_acc_std"] * 100 for s in summary], capsize=4)
    ax.set(xlabel="Condition", ylabel="Final validation accuracy (%)",
           title="Seed mean ± population std")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(output / "final_val_accuracy.png", dpi=150)
    fig, ax = figure()
    for key, runs in grouped.items():
        ax.scatter([r["initial_statistics"]["position_patch_std_ratio"] for r in runs],
                   [r["final_val_acc"] * 100 for r in runs], label=f"{key}: {LABELS[key]}")
    ax.set(xlabel="Initial scaled position / patch std", ylabel="Final validation accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output / "initial_ratio_vs_accuracy.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "stripes"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/position_ablation"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="First of consecutive seeds")
    parser.add_argument("--seeds", type=int, nargs="+", help="Explicit seed list overrides num-seeds")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-train-samples", type=int, help="Optional smoke-test subset")
    parser.add_argument("--max-val-samples", type=int, help="Optional smoke-test subset")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot_results(args.output_dir)
        return
    if any(v <= 0 for v in (args.epochs, args.batch_size, args.num_seeds, args.lr,
                            args.d_model, args.num_heads, args.num_layers, args.d_ff)):
        parser.error("Training sizes, seed count and learning rate must be positive")
    if not math.isfinite(args.lr) or not 0 <= args.dropout <= 1:
        parser.error("lr must be finite and dropout must be in [0, 1]")
    if args.d_model % args.num_heads:
        parser.error("d-model must be divisible by num-heads")
    image_size = 32 if args.dataset == "cifar10" else 16
    if args.patch_size <= 0 or image_size % args.patch_size:
        parser.error("patch-size must be a positive divisor of image size")
    for limit in (args.max_train_samples, args.max_val_samples):
        if limit is not None and limit <= 0:
            parser.error("Sample limits must be positive")
    seeds = args.seeds if args.seeds is not None else list(range(args.seed, args.seed + args.num_seeds))
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**32 for s in seeds):
        parser.error("Seeds must be unique integers in [0, 2**32)")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    # Avoid CUDA algorithm selection and request deterministic kernels when available.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    config = {"image_size": image_size, "patch_size": args.patch_size,
                  "in_channels": 3 if args.dataset == "cifar10" else 1,
                  "num_classes": 10 if args.dataset == "cifar10" else 2,
                  "d_model": args.d_model, "num_heads": args.num_heads, "d_ff": args.d_ff,
                  "num_layers": args.num_layers, "dropout": args.dropout}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse accidental mixing or overwriting of prior experimental records.
    if (args.output_dir / "config.json").exists():
        parser.error("Output already contains a run; choose a new output-dir or use --plot-only")
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(model_config=config, seeds=seeds, resolved_device=str(device),
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    optimizer={"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.999],
                               "eps": 1e-8, "weight_decay": 0.01},
                    scheduler=None, augmentation=None, transform="ToTensor (CIFAR-10)",
                    conditions=conditions(args.d_model), std_convention="population (ddof=0)",
                    final_metric="last epoch; no test-set evaluation",
                    determinism="Best effort; deterministic algorithms warn_only; workers=0")
    write_json(args.output_dir / "config.json", metadata)
    results = []
    for seed in seeds:
        seed_everything(seed)
        if args.dataset == "cifar10":
            from vision_transformer.cifar10 import load_cifar10
            train, validation, _ = load_cifar10(args.data_dir, seed, not args.no_download)
        else:
            train = make_stripe_dataset(256, seed)
            validation = make_stripe_dataset(128, seed + 1)
        if args.max_train_samples:
            train = Subset(train, range(min(len(train), args.max_train_samples)))
        if args.max_val_samples:
            validation = Subset(validation, range(min(len(validation), args.max_val_samples)))
        # Separate loader and RNG so this diagnostic does not consume training shuffle/RNG.
        images, _ = next(iter(DataLoader(train, batch_size=args.batch_size,
                            generator=torch.Generator().manual_seed(seed))))
        images = images.to(device)
        for condition, position_config in conditions(args.d_model).items():
            run_dir = args.output_dir / f"{condition}_seed_{seed}"
            run_dir.mkdir()
            seed_everything(seed)
            model_config = {**config, **position_config}
            model = VisionTransformer(**model_config).to(device)
            # normal_ consumes identical standard normal draws for each condition.
            stats = initial_statistics(model, images)
            train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                      generator=torch.Generator().manual_seed(seed))
            val_loader = DataLoader(validation, batch_size=args.batch_size,
                                    generator=torch.Generator().manual_seed(seed))
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            record = {"condition": condition, "seed": seed, "model_config": model_config,
                      "train_samples": len(train), "validation_samples": len(validation),
                      "initial_statistics": stats, "statistics_samples": images.shape[0],
                      "history": []}
            write_json(run_dir / "run.json", record)
            history_path = run_dir / "history.csv"
            fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
            with history_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for epoch in range(1, args.epochs + 1):
                    lr = optimizer.param_groups[0]["lr"]
                    training = train_one_epoch(model, train_loader, optimizer, device)
                    val = evaluate(model, val_loader, device)
                    row = {"epoch": epoch, "train_loss": training["loss"], "train_acc": training["accuracy"],
                               "val_loss": val["loss"], "val_acc": val["accuracy"], "lr": lr}
                    record["history"].append(row)
                    writer.writerow(row)
                    file.flush()
                    print(f"{condition} seed={seed} {row}", flush=True)
            record["final_val_acc"] = record["history"][-1]["val_acc"]
            write_json(run_dir / "run.json", record)
            torch.save({"model_config": model_config, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "seed": seed,
                        "condition": condition, "epoch": args.epochs}, run_dir / "final.pt")
            results.append(record)
            write_json(args.output_dir / "results.json", results)
    plot_results(args.output_dir)


if __name__ == "__main__":
    main()
