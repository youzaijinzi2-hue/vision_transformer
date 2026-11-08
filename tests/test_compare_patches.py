import csv
import json
import sys
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import TensorDataset

from vision_transformer.compare_patches import plot_histories
from vision_transformer.train_cifar10 import main


def test_training_writes_real_metrics_and_monotonic_times(tmp_path):
    dataset = TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    dataset.classes = [str(i) for i in range(10)]
    output = tmp_path / "model.pt"
    argv = ["train", "--epochs", "2", "--patch-size", "16", "--batch-size", "2",
            "--output", str(output), "--skip-test"]
    with patch.object(sys, "argv", argv), patch(
        "vision_transformer.train_cifar10.load_cifar10",
        return_value=(dataset, dataset, dataset),
    ):
        main()
    with output.with_suffix(".csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert all(int(row["patch_size"]) == 16 for row in rows)
    elapsed = [float(row["cumulative_train_seconds"]) for row in rows]
    assert 0 < elapsed[0] < elapsed[1]
    assert elapsed[1] == pytest.approx(sum(float(row["train_seconds"]) for row in rows))
    assert all(0 <= float(row["val_acc"]) <= 1 for row in rows)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["model_config"]["patch_size"] == 16
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["dataset"] == "cifar10"
    outputs = plot_histories([output.with_suffix(".csv")], tmp_path / "plots")
    assert len(outputs) == 3
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in outputs)


def test_plot_multiple_patch_sizes(tmp_path):
    paths = []
    for size in (4, 8):
        path = tmp_path / f"patch_{size}.csv"
        path.write_text(
            "epoch,patch_size,train_loss,val_loss,val_acc,cumulative_train_seconds\n"
            f"1,{size},2.0,2.1,0.2,10\n2,{size},1.5,1.7,0.4,20\n"
        )
        paths.append(path)
    assert all(path.exists() for path in plot_histories(paths, tmp_path / "plots"))


def test_plot_rejects_empty_history(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("epoch,patch_size\n")
    with pytest.raises(ValueError, match="empty"):
        plot_histories([path], tmp_path)
