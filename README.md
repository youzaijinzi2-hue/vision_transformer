# Vision Transformer from scratch

## CIFAR-10で学習

```bash
uv sync
uv run python -m vision_transformer.train_cifar10
```

初回はtorchvisionがCIFAR-10を `data/` にダウンロード・展開します。
学習45,000枚と検証5,000枚に分け、検証損失が最小のモデルを
`checkpoints/cifar10.pt` に保存します。学習後、そのモデルを
学習・モデル選択に使っていないテスト10,000枚で評価します。
画像はToTensorで0〜1に変換し、データ拡張は行わない最初の実装です。

Apple SiliconのGPUを使う場合は `--device mps`、CUDAなら `--device cuda` を指定します。
`--epochs`、`--batch-size`、`--lr`、`--seed`、`--data-dir`、`--output`も指定できます。
ダウンロード済みのデータだけを使う場合は `--no-download` を付けます。
データ、保存した重み、生成画像はGit管理から除外しています。
学習後に、テスト画像の正解と予測を一覧で確認できます。

```bash
uv run python -m vision_transformer.visualize_predictions \
  --checkpoint checkpoints/cifar10.pt \
  --output outputs/cifar10_predictions.png
```

チェックポイントからデータセットを判別し、公式テストデータから16枚を
重複なしで無作為抽出します。`--num-samples` と `--seed` で変更できます。
一覧の正解率は表示した画像だけの値です。

## 学習処理の動作確認

```bash
.venv/bin/python -m vision_transformer.train
```

ノイズ付きの縦線・横線画像を生成し、2クラス分類を学習します。
これは学習経路の動作確認用で、自然画像に対する性能評価ではありません。
学習用256枚と評価用128枚を別々に生成します。
`--epochs`、`--batch-size`、`--lr`、`--seed`、`--device`、`--output`を指定できます。

モデルの設定と重みは `checkpoints/stripes.pt` に保存されます。

```python
import torch
from vision_transformer.model import VisionTransformer

checkpoint = torch.load("checkpoints/stripes.pt", map_location="cpu", weights_only=True)
model = VisionTransformer(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## 予測を画像で確認

学習後に実行します。

```bash
.venv/bin/python -m vision_transformer.visualize_predictions
```

`outputs/predictions.png` に16枚の画像、正解（True）、予測（Pred）、
予測クラスのsoftmax値を保存します。緑は正解、赤は誤分類です。
softmax値は正しさを保証する確率ではありません。
表示される正解率は一覧に載せた画像だけの値です。
`--checkpoint`、`--output`、`--num-samples`、`--seed`で変更できます。
既定のseed=1は、既定の学習実行で評価に使うseedと同じです。

## テスト

```bash
.venv/bin/python -m pytest -q
```
