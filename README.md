# Vision Transformer from scratch

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
