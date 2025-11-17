# Analysis Common Utilities

analysisディレクトリ配下のスクリプトで共通して使用するユーティリティモジュールです。

## モジュール構成

### 1. `model_utils.py` - モデル関連ユーティリティ

事前学習済みモデルの読み込み、特徴量抽出など。

```python
from analysis.common import load_pretrained_model, extract_features, get_model_window_size

# モデル読み込み（ウィンドウサイズ自動検出）
encoder, window_size = load_pretrained_model(
    model_path="experiments/pretrain/run_*/exp_0/models/checkpoint.pth",
    device='cuda'
)

# 特徴量抽出
X = np.random.randn(100, 3, 150)  # (N, channels, time_steps)
features = extract_features(encoder, X, batch_size=256, device='cuda')
# -> (N, feature_dim)
```

**主な関数:**
- `load_pretrained_model(model_path, window_size=None, device='cuda')` - モデル読み込み
- `extract_features(encoder, X, batch_size=256, device='cuda')` - 特徴量抽出
- `get_model_window_size(model_path, default=150)` - ウィンドウサイズ自動検出

### 2. `data_utils.py` - データ関連ユーティリティ

センサーデータの読み込み、ウィンドウクリッピング、身体部位カテゴリ化など。

```python
from analysis.common import load_sensor_data, find_dataset_location_pairs, categorize_body_part

# 利用可能なデータセット・部位のペアを検出
pairs = find_dataset_location_pairs(
    dataset_filter=['dsads', 'mhealth'],  # 特定データセットのみ
    location_filter=['Torso', 'Wrist']     # 特定部位のみ
)
# -> [('dsads', 'Torso'), ('mhealth', 'Chest'), ...]

# センサーデータ読み込み
X, y, metadata = load_sensor_data(
    dataset_name='dsads',
    location='Torso',
    window_size=60,                    # クリップ後のサイズ
    max_samples_per_class=100,         # クラスごとの最大サンプル数
    max_users=20                       # 大規模データセット用
)

# 身体部位のカテゴリ化
category = categorize_body_part('RightWrist')  # -> 'Wrist'
category = categorize_body_part('LeftAnkle')  # -> 'Ankle'
```

**主な関数:**
- `find_dataset_location_pairs(dataset_filter, location_filter)` - データセット・部位ペア検出
- `load_sensor_data(dataset_name, location, window_size, max_samples_per_class)` - データ読み込み
- `clip_windows(X, window_size, strategy='center')` - ウィンドウクリッピング
- `get_label_dict(dataset_name)` - ラベル辞書取得
- `categorize_body_part(location)` - 身体部位カテゴリ化

### 3. `viz_utils.py` - 可視化ユーティリティ

次元削減、プロット設定、カラーパレットなど。

```python
from analysis.common import reduce_dimensions, setup_plotting_style, get_color_palette

# プロット設定
setup_plotting_style(style='whitegrid', context='paper', font_scale=1.0)

# 次元削減
features = np.random.randn(1000, 512)  # (N, D)
embedded = reduce_dimensions(features, method='umap', n_components=2)
# -> (N, 2)

# カラーパレット取得
colors = get_color_palette(n_colors=10, palette_name='auto')
# palette_name: 'auto', 'Set1', 'Set3', 'tab20', 'hsv' など

# 身体部位カテゴリごとの色
body_part_colors = get_body_part_colors()
# -> {'Wrist': '#ff6b6b', 'Ankle': '#4ecdc4', ...}
```

**主な関数:**
- `reduce_dimensions(features, method='umap', n_components=2)` - 次元削減
- `setup_plotting_style(style, context, font_scale)` - プロット設定
- `get_color_palette(n_colors, palette_name='auto')` - カラーパレット
- `get_body_part_colors()` - 身体部位カテゴリの色マッピング
- `get_dataset_colors()` - データセットごとの色マッピング
- `save_figure(fig, output_path, dpi=150)` - 図の保存

## 使用例

### 例1: 特徴量抽出と可視化

```python
from analysis.common import (
    load_pretrained_model,
    extract_features,
    load_sensor_data,
    reduce_dimensions,
    setup_plotting_style,
)
import matplotlib.pyplot as plt

# スタイル設定
setup_plotting_style()

# モデル読み込み
encoder, window_size = load_pretrained_model('path/to/model.pth')

# データ読み込み
X, y, _ = load_sensor_data('dsads', 'Torso', window_size=window_size)

# 特徴抽出
features = extract_features(encoder, X)

# 次元削減
embedded = reduce_dimensions(features, method='umap')

# プロット
plt.scatter(embedded[:, 0], embedded[:, 1], c=y, cmap='tab10')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('UMAP Embeddings')
plt.show()
```

### 例2: 複数データセットの比較

```python
from analysis.common import (
    find_dataset_location_pairs,
    load_sensor_data,
    categorize_body_part,
)

# 全データセット・部位のペアを取得
pairs = find_dataset_location_pairs()

for dataset, location in pairs:
    category = categorize_body_part(location)
    print(f"{dataset}/{location} -> {category}")

    X, y, metadata = load_sensor_data(
        dataset, location,
        max_samples_per_class=50,
        max_users=10
    )
    print(f"  Samples: {len(X)}, Classes: {len(set(y))}")
```

## 依存関係

- **PyTorch** - モデル読み込み・特徴抽出
- **NumPy** - データ処理
- **UMAP / scikit-learn** - 次元削減
- **matplotlib / seaborn** - 可視化
- **plotly** - インタラクティブ可視化

## リファクタリング前後の比較

### リファクタリング前

各スクリプトで重複したコードが存在：

```python
# visualize_embeddings.py
def load_pretrained_model(model_path, device='cuda'):
    # 100行以上の重複コード
    ...

# embedding_explorer/extract_features.py
def load_pretrained_model(model_path, device='cuda'):
    # 同じコードが再度実装されている
    ...
```

### リファクタリング後

共通モジュールを使用：

```python
# visualize_embeddings.py
from analysis.common import load_pretrained_model

encoder, window_size = load_pretrained_model(model_path)
```

**メリット:**
- コードの重複を削減（約300行削減）
- バグ修正が一箇所で済む
- 保守性・可読性の向上
- テストが容易

## 今後の拡張

以下の機能を追加予定：

1. **キャッシング機構** - 特徴量の自動キャッシュ
2. **並列処理** - マルチプロセスによる高速化
3. **データ拡張** - 共通の拡張関数
4. **メトリクス計算** - 共通の評価指標計算

## トラブルシューティング

### Q: `ModuleNotFoundError: No module named 'analysis.common'`

A: プロジェクトルートからスクリプトを実行してください。

```bash
# プロジェクトルートから
python analysis/visualize_embeddings.py

# サブディレクトリからの実行はNG
cd analysis && python visualize_embeddings.py  # NG
```

### Q: ウィンドウサイズが正しく検出されない

A: `--window-size`オプションで明示的に指定してください。

```bash
python analysis/visualize_embeddings.py \\
    --model path/to/model.pth \\
    --window-size 60
```

### Q: メモリ不足エラーが発生

A: `max_samples_per_class`や`max_users`を減らしてください。

```python
X, y, _ = load_sensor_data(
    dataset, location,
    max_samples_per_class=50,  # デフォルト100 -> 50に削減
    max_users=10                # デフォルト20 -> 10に削減
)
```

## 貢献

新しいユーティリティ関数を追加する場合：

1. 適切なモジュールに追加（`model_utils.py`, `data_utils.py`, `viz_utils.py`）
2. docstringを記述（Args, Returns, 使用例）
3. `__init__.py`の`__all__`にエクスポート
4. このREADMEに使用例を追記
