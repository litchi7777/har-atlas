# Analysis Scripts

HARFoundationプロジェクトの分析スクリプト集です。

## ディレクトリ構造

```
analysis/
├── common/                    # ★共通ユーティリティモジュール
│   ├── __init__.py
│   ├── model_utils.py         # モデル読み込み・特徴抽出
│   ├── data_utils.py          # データ読み込み・処理
│   ├── viz_utils.py           # 可視化ユーティリティ
│   └── README.md              # 詳細な使用方法
│
├── embedding_explorer/        # インタラクティブ埋め込み可視化
│   ├── extract_features.py   # 特徴量抽出
│   ├── extract_all_features.py  # 全ウィンドウサイズの特徴抽出
│   ├── server.py              # 可視化サーバー
│   ├── templates/             # HTMLテンプレート
│   └── data/                  # 抽出済み特徴量（.gitignore対象）
│
├── visualize_embeddings.py   # ★埋め込み可視化（メイン）
├── visualize_finetune_comparison.py  # ファインチューニング比較
├── report_f1_comparison.py   # F1スコア比較レポート
│
├── .archive/                  # 使用頻度の低いスクリプト（.gitignore対象）
│   ├── README.md              # アーカイブの説明
│   ├── data_quality.py        # データ品質チェック
│   ├── dataset_distribution.py  # データセット分布分析
│   ├── feature_analysis.py    # 特徴量分析
│   ├── model_performance.py   # モデル性能分析
│   ├── utils.py               # 旧ユーティリティ
│   └── analyze.py             # 統合分析インターフェース
│
└── figures/                   # 生成された図（.gitignore対象）
```

## リファクタリング概要

### 変更内容（2024-11-17整理）

1. **共通モジュールの作成** (`analysis/common/`)
   - 重複コードを削減（約300行削減）
   - モデル読み込み、データ処理、可視化の共通化
   - テスト・保守が容易に

2. **主要スクリプトのリファクタリング**
   - `visualize_embeddings.py` - 共通モジュールを使用
   - `embedding_explorer/extract_features.py` - 共通モジュールを使用

3. **使用頻度の低いスクリプトをアーカイブ** (`analysis/.archive/`)
   - 6つの古いスクリプトを移動
   - 必要に応じて復元可能
   - メインディレクトリがスッキリ

### よく使うスクリプト

日常的に使う分析スクリプトは3つだけ：

1. **visualize_embeddings.py** - 埋め込み可視化（UMAP/t-SNE）
2. **report_f1_comparison.py** - F1スコア比較レポート
3. **visualize_finetune_comparison.py** - ファインチューニング結果比較

## 主要スクリプト

### 1. 埋め込み可視化 (`visualize_embeddings.py`)

事前学習済みモデルの特徴表現をUMAP/t-SNEで可視化。

```bash
# 基本的な使用
python analysis/visualize_embeddings.py \\
    --model experiments/pretrain/run_*/exp_0/models/checkpoint.pth \\
    --method umap \\
    --color-by body_part

# 特定のデータセット・部位のみ
python analysis/visualize_embeddings.py \\
    --model experiments/pretrain/run_*/exp_0/models/checkpoint.pth \\
    --datasets dsads mhealth pamap2 \\
    --locations Torso Wrist

# 複数モデルの比較
python analysis/visualize_embeddings.py \\
    --models model1.pth model2.pth model3.pth \\
    --method umap \\
    --color-by dataset
```

**出力:** `analysis/figures/embeddings_*.png` (静的画像)、`*.html` (インタラクティブ)

### 2. 特徴量抽出 (`embedding_explorer/extract_features.py`)

全データセットから特徴量を抽出してファイルに保存。

```bash
python analysis/embedding_explorer/extract_features.py \\
    --model experiments/pretrain/run_*/exp_0/models/checkpoint.pth \\
    --window-size 60 \\
    --max-samples 100 \\
    --output-dir analysis/embedding_explorer/data \\
    --compute-umap
```

**出力:**
- `features_2.0s.npz` - 特徴量（NumPy配列）
- `metadata_2.0s.json` - メタデータ（データセット名、ラベルなど）
- `umap_2.0s.npz` - UMAP埋め込み（オプション）

### 3. インタラクティブ可視化サーバー (`embedding_explorer/server.py`)

ブラウザでインタラクティブに埋め込みを探索。

```bash
python analysis/embedding_explorer/server.py
# ブラウザで http://localhost:5001 を開く
```

### 4. ファインチューニング比較 (`visualize_finetune_comparison.py`)

複数のファインチューニング実験のF1スコアを比較。

```bash
python analysis/visualize_finetune_comparison.py \\
    --runs run_20251112_* \\
    --output-dir analysis/figures
```

### 5. F1スコア比較レポート (`report_f1_comparison.py`)

実験結果のF1スコアを表形式で出力。

```bash
python analysis/report_f1_comparison.py \\
    --finetune-runs run_20251112_*
```

## 共通モジュールの使用方法

詳細は [`analysis/common/README.md`](common/README.md) を参照。

### クイックスタート

```python
from analysis.common import (
    load_pretrained_model,
    extract_features,
    load_sensor_data,
    reduce_dimensions,
)

# モデル読み込み（ウィンドウサイズ自動検出）
encoder, window_size = load_pretrained_model('path/to/model.pth')

# データ読み込み
X, y, _ = load_sensor_data('dsads', 'Torso', window_size=window_size)

# 特徴抽出
features = extract_features(encoder, X)

# 次元削減
embedded = reduce_dimensions(features, method='umap')
```

## データセット情報

使用可能なデータセット：
- DSADS, MHEALTH, PAMAP2, HARTH, RealDisp, USCHAD
- FORTHTRACE, HAR70PLUS, LARA, MEX, PAAL, SELFBACK
- NHANES（巨大データセット、サンプリング推奨）

身体部位カテゴリ：
- **Wrist** - 手首
- **Ankle** - 足首
- **Arm** - 腕（上腕、前腕）
- **Leg** - 脚（太もも、すね）
- **Front** - 胴体前面（胸、腰）
- **Back** - 背中
- **Head** - 頭部
- **Phone** - ポケット（スマートフォン）

## トラブルシューティング

### メモリ不足

```bash
# サンプル数を減らす
--max-samples 50  # デフォルト: 100

# ユーザー数を減らす（大規模データセット用）
# スクリプト内で max_users=10 を指定
```

### ウィンドウサイズが合わない

```bash
# 明示的にウィンドウサイズを指定
--window-size 60
```

### CUDAメモリ不足

```bash
# CPUを使用
--device cpu

# バッチサイズを減らす（スクリプト内で batch_size=128）
```

## 開発者向け

### 新しい分析スクリプトの追加

1. `analysis/common/` の共通モジュールを活用
2. Docstringを記述
3. `--help`オプションで使い方を表示
4. このREADMEに使用例を追記

### コーディング規約

- 共通化できるコードは `analysis/common/` に移動
- 各スクリプトは独立して実行可能に
- 出力は `analysis/figures/` に保存
- 大きなデータは `.gitignore` に追加

## 参考リンク

- [プロジェクトREADME](../README.md)
- [共通モジュール詳細](common/README.md)
- [データセット情報](../har-unified-dataset/README.md)
