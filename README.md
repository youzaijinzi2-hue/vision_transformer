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

## Patch sizeの比較実験

```bash
uv run python -m vision_transformer.compare_patches \
  --patch-sizes 4 8 16 --epochs 30 --device mps
```

CPUなら `--device cpu`、CUDAなら `--device cuda` を指定します。
3条件を順番に学習するので、合計90エポックになります。
出力先は `outputs/patch_comparison/` です。

- `validation_accuracy.png`: epochに対する検証正解率を重ねた図
- `losses.png`: patch sizeごとの訓練・検証損失
- `accuracy_vs_time.png`: 累積訓練時間に対する検証正解率
- `patch_*.csv`: 各epochの精度・損失・訓練時間・検証時間
- `patch_*.json`: モデル設定、seed、deviceなどの実験条件
- `patch_*.pt`: 検証損失が最小のモデル

データ分割、shuffleのseed、batch size、学習率、層数、埋め込み次元を揃えます。
patch sizeにより系列長とEmbeddingのパラメータ数は変わります。
累積訓練時間にはデータ読み込みとパラメータ更新を含み、検証・保存・初回ダウンロードを除外します。
GPUの非同期処理を同期して計測します。同じdeviceで他の負荷を避けて比較してください。
この実験ではテスト評価を行わず、検証データで比較します。
単一seedの結果なので、優劣を確かめる際はseedを変えて別フォルダに保存してください。
訓練損失はDropout有効、検証損失は無効で測るため、その差だけで過学習とは判断できません。

CSVからの再描画（学習は実行しません）:

```bash
uv run python -m vision_transformer.compare_patches --plot-only --patch-sizes 4 8 16
```

単一条件の既存学習でも `--patch-size 8` を指定できます。
CSVとJSONはチェックポイントと同名で保存されます。
同じ出力先で再学習すると上書きされるため、別の実験には `--output-dir` を変更してください。

## 人工画像での学習処理の動作確認

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

### Position Embedding ablation

`VisionTransformerEmbedding` / `VisionTransformer` accept
`position_init_std=0.02` and `position_scale=1.0`. The defaults preserve the
previous behavior; the scale is a fixed scalar, while positions remain trainable.

Run all six conditions A–F with three seeds (0, 1, 2):

```bash
uv run python -m vision_transformer.ablate_positions \
  --dataset cifar10 --epochs 10 --num-seeds 3 --device auto \
  --output-dir outputs/position_ablation
```

Defaults follow `train_cifar10`: patch size 4, D=64, 3 encoder layers, 4 heads,
FFN=256, dropout=0.1, batch size 64, AdamW lr=0.001 (weight decay=0.01), no
scheduler, no augmentation, ToTensor, and a seeded 45,000/5,000 train/validation
split. Test accuracy is not used. Use `--seeds 10 20 30` for explicit seeds,
`--no-download` for cached CIFAR-10, or `--device cpu|cuda|mps`. Model dimensions,
dropout, batch size and learning rate are configurable and shared by every run.

Download-free smoke test (synthetic stripes, not a CIFAR-10 result):

```bash
uv run python -m vision_transformer.ablate_positions \
  --dataset stripes --epochs 1 --num-seeds 2 --batch-size 8 \
  --max-train-samples 16 --max-val-samples 8 --device cpu \
  --output-dir outputs/position_ablation_smoke
```

Each output directory stores `config.json` (settings, versions, seeds, optimizer),
`results.json` (all runs), `summary.json` (final validation mean/std), and five
PNG plots. Each `A_seed_0/` etc. contains `run.json` (initial statistics and full
history), `history.csv` (epoch, train/validation loss/accuracy, learning rate),
and `final.pt` (last-epoch model, configuration and optimizer state). Final
accuracy always means the last epoch, not the best validation epoch. Replot with:

```bash
uv run python -m vision_transformer.ablate_positions \
  --plot-only --output-dir outputs/position_ablation
```

Curves show seed means with ±1 population standard deviation; bars use the same
population standard deviation (`ddof=0`, also defined for one seed). The scatter
shows each seed separately. Initial statistics use the first unshuffled real
training batch before updates, exclude CLS in both embeddings, and measure
population std across all elements and mean L2 norm across patch tokens. Position
statistics cover its unique patch positions, without batch duplication. This
measurement does not consume the training loader or training random stream.

For a given seed, each condition resets Python/NumPy/PyTorch RNGs, creates the
same architecture in the same order, and uses an independently seeded loader.
This shares standard normal initialization draws, common parameter initial
values, dataset split and shuffle order. DataLoader workers are zero; deterministic
algorithms are requested with warnings for unsupported kernels. Reproducibility
across devices or PyTorch versions is not guaranteed.

B and C have the same effective initial position distribution (and matching
samples up to floating-point roundoff with the shared draws). Their trainable
parameters differ: `dL/dP = alpha * dL/d(alpha P)`. AdamW moments, epsilon and
parameter updates depend on that parameterization; an update in P is multiplied
by alpha at the encoder input. Equal initial position/patch ratios therefore do
not imply equal training dynamics. Compare paired seeds and learning curves as
well as final means/std; the synthetic smoke test is only an execution check.
Use a fresh output directory for each experiment to avoid mixing settings.
