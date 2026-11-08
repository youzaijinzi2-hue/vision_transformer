"""patch size比較実験の逐次実行と3種類のグラフ出力。"""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_histories(paths: list[Path], output_dir: Path) -> list[Path]:
    """実測したCSVから描画する。実験を再実行せず再描画もできる。"""
    histories = []
    for path in paths:
        with path.open(newline="") as file:
            rows = [{key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(file)]
        if not rows:
            raise ValueError(f"History is empty: {path}")
        histories.append(rows)
    if not histories:
        raise ValueError("At least one history is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = Figure(figsize=(8, 5), layout="constrained")
    time_plot = Figure(figsize=(8, 5), layout="constrained")
    losses = Figure(figsize=(7, 3.5 * len(histories)), layout="constrained")
    for figure in (accuracy, time_plot, losses):
        FigureCanvasAgg(figure)
    accuracy_axis = accuracy.subplots()
    time_axis = time_plot.subplots()
    loss_axes = losses.subplots(len(histories), 1, squeeze=False)
    for rows, axes in zip(histories, loss_axes):
        patch_size = int(rows[0]["patch_size"])
        label = f"P={patch_size} ({(32 // patch_size)**2} patches)"
        epochs = [row["epoch"] for row in rows]
        val_acc = [row["val_acc"] * 100 for row in rows]
        accuracy_axis.plot(epochs, val_acc, marker=".", label=label)
        time_axis.plot([row["cumulative_train_seconds"] for row in rows],
                       val_acc, marker=".", label=label)
        axis = axes[0]
        axis.plot(epochs, [row["train_loss"] for row in rows], label="Training")
        axis.plot(epochs, [row["val_loss"] for row in rows], label="Validation")
        axis.set(title=label, xlabel="Epoch", ylabel="Cross-entropy loss")
        axis.grid(alpha=0.3)
        axis.legend()
    accuracy_axis.set(xlabel="Epoch", ylabel="Validation accuracy (%)",
                      title="Validation accuracy vs Epoch", ylim=(0, 100))
    time_axis.set(xlabel="Cumulative training time (seconds)", ylabel="Validation accuracy (%)",
                  title="Validation accuracy vs Training time", ylim=(0, 100))
    for axis in (accuracy_axis, time_axis):
        axis.grid(alpha=0.3)
        axis.legend()
    outputs = []
    for figure, name in zip((accuracy, losses, time_plot),
                            ("validation_accuracy", "losses", "accuracy_vs_time")):
        path = output_dir / f"{name}.png"
        figure.savefig(path, dpi=150)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-sizes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/patch_comparison"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if any(size <= 0 or 32 % size for size in args.patch_sizes):
        parser.error("patch sizes must be positive divisors of 32")
    if len(set(args.patch_sizes)) != len(args.patch_sizes):
        parser.error("patch sizes must be unique")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    paths = []
    for size in args.patch_sizes:
        history = args.output_dir / f"patch_{size}.csv"
        paths.append(history)
        if args.plot_only:
            continue
        # 別プロセスで逐次実行し、各条件でモデルとshuffleのseedをリセット。
        command = [
            sys.executable, "-m", "vision_transformer.train_cifar10",
            "--patch-size", str(size), "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size), "--lr", str(args.lr),
            "--seed", str(args.seed), "--device", args.device,
            "--data-dir", str(args.data_dir), "--history", str(history),
            "--output", str(args.output_dir / f"patch_{size}.pt"), "--skip-test",
        ]
        if args.no_download:
            command.append("--no-download")
        subprocess.run(command, check=True)
    for path in plot_histories(paths, args.output_dir):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
