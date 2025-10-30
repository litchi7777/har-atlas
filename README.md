# HAR Foundation

Human Activity Recognition (HAR) プロジェクト - Self-Supervised Learning (SSL) を用いた事前学習とファインチューニングによる行動クラス分類

## プロジェクト概要

このプロジェクトは、2段階の学習アプローチで行動クラス分類を実現します：

1. **Pre-training Phase**: ラベルなしデータを使用したSSL（自己教師あり学習）による事前学習
2. **Fine-tuning Phase**: ラベル付きデータを使用した教師あり学習によるファインチューニング

## プロジェクト構造

```
har-foundation/
├── train.py                  # トレーニング実行スクリプト
├── configs/                  # 設定ファイル
│   ├── pretrain.yaml        # Pre-training設定（グリッドサーチ対応）
│   └── finetune.yaml        # Fine-tuning設定（グリッドサーチ対応）
├── har-unified-dataset/      # データセット前処理・可視化（サブモジュール）
│   ├── preprocess.py        # データ前処理スクリプト
│   ├── visualize_server.py  # データ可視化Webサーバー
│   ├── src/
│   │   ├── dataset_info.py  # データセットメタデータ
│   │   ├── preprocessors/   # データセット別前処理ロジック
│   │   └── visualization/   # 可視化ツール
│   ├── data/                # データディレクトリ
│   │   ├── raw/             # 生データ
│   │   └── processed/       # 前処理済みデータ
│   └── outputs/             # 可視化結果（HTML）
├── models/                   # モデル保存ディレクトリ
│   ├── pretrained/          # 事前学習モデル
│   ├── finetuned/           # ファインチューニング済みモデル
│   └── checkpoints/         # チェックポイント
├── src/                      # ソースコード
│   ├── data/                # データ処理（学習用）
│   │   ├── dataset.py       # Dataset定義
│   │   ├── sensor_dataset.py # Sensor Dataset
│   │   └── augmentations.py # データ拡張
│   ├── models/              # モデル定義
│   │   └── model.py         # SSLモデル、分類モデル
│   ├── training/            # トレーニングスクリプト
│   │   ├── pretrain.py      # Pre-trainingスクリプト
│   │   ├── finetune.py      # Fine-tuningスクリプト
│   │   └── run_experiments.py  # 複数実験実行スクリプト
│   └── utils/               # ユーティリティ
│       ├── logger.py        # ロギング
│       └── metrics.py       # 評価メトリクス
├── logs/                     # ログファイル
├── experiments/              # 実験結果
├── __test__/                 # テストコード
├── requirements.txt          # 依存関係
├── .gitignore
├── .gitmodules              # サブモジュール設定
├── CLAUDE.md                # Claude Code用ガイドライン
└── README.md                # このファイル
```

**注意**: `har-unified-dataset`はGitサブモジュールです。初回クローン時は以下のコマンドでサブモジュールも取得してください：

```bash
git clone --recursive git@github.com:litchi7777/har-foundation.git

# または既にクローン済みの場合
git submodule update --init --recursive
```

## セットアップ

### 1. 環境構築

```bash
# 仮想環境作成
python -m venv venv

# 仮想環境有効化
# Linux/Mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt
```

### 2. Weights & Biases セットアップ（オプション）

```bash
# W&Bにログイン
wandb login

# または環境変数で設定
export WANDB_API_KEY=your_api_key
```

設定ファイル（`configs/pretrain.yaml`, `configs/finetune.yaml`）の`wandb`セクションを編集：

```yaml
wandb:
  enabled: true
  project: "har-foundation"
  entity: "your-username"  # あなたのW&Bユーザー名
```

## データ前処理

データ前処理機能は`har-unified-dataset`サブモジュールに移動しました。

### データセットのダウンロードと前処理

```bash
# サブモジュールに移動
cd har-unified-dataset

# DSADSデータセットをダウンロードして前処理（ワンコマンド）
python preprocess.py --dataset dsads --download

# 前処理のみ（データが既にある場合）
python preprocess.py --dataset dsads
```

処理済みデータは以下の形式で保存されます：

```
data/processed/dsads/
├── USER00001/
│   ├── Torso_ACC/
│   │   ├── X.npy      # (num_windows, 3, 125)
│   │   └── Y.npy      # (num_windows,)
│   ├── Torso_GYRO/
│   ├── Torso_MAG/
│   ├── LeftArm_ACC/
│   └── ... (5センサー × 3モダリティ = 15ディレクトリ)
└── USER00002/
    └── ...
```

### データの確認・可視化

データ可視化機能は`har-unified-dataset`サブモジュールに移動しました。

```bash
# Webサーバーで可視化（推奨）
cd har-unified-dataset
python visualize_server.py

# ブラウザで http://localhost:5000 にアクセス
# インタラクティブにデータセット、ユーザー、センサーを選択して可視化
```

詳細は [`har-unified-dataset/README.md`](har-unified-dataset/README.md) を参照してください。

---

## 使用方法

### トレーニングの実行

**推奨コマンド（これだけで完結）**：

```bash
# Pre-training（事前学習）
python train.py pretrain

# Fine-tuning（ファインチューニング）
python train.py finetune
```

それだけです！以下のことが自動的に行われます：

- **グリッドサーチの自動検出**: 設定ファイルの`grid_search`セクションを読み取り、単一実験かグリッドサーチかを自動判定
- **並列実行**: 利用可能なGPUを自動検出し、複数実験を並列実行（`settings.parallel: true`の場合）
- **実験管理**: 各実験の設定、ログ、チェックポイントを自動的に保存

**現在のpretrain設定例**（`configs/pretrain.yaml`）：
- データセット: NHANES
- GPU: cuda:0
- グリッドサーチ: 3つのSSLタスク組み合わせを検証
  1. `[binary_permute, binary_reverse, binary_timewarp]`
  2. `[binary_permute, binary_reverse, binary_timewarp, masking_time_channel]`
  3. `[masking_time_channel]`
- 並列実行: 有効（`parallel: true`）

実行すると、3つの実験が自動的に並列実行され、結果は`experiments/pretrain/run_YYYYMMDD_HHMMSS/`に保存されます。

### グリッドサーチの設定

#### 単一実験（パラメータ1つだけ）

```yaml
# configs/finetune.yaml
grid_search:
  training:
    learning_rate: [0.0001]  # 1つだけ = 単一実験
```

実行：
```bash
python train.py finetune  # 1実験のみ実行
```

#### 複数実験（グリッドサーチ）

```yaml
# configs/finetune.yaml
grid_search:
  training:
    learning_rate: [0.0001, 0.001, 0.01]  # 3パターン
    batch_size: [32, 64, 128]              # 3パターン
    optimizer: ["adam", "sgd"]             # 2パターン
# → 3 × 3 × 2 = 18実験が自動生成される
```

実行：
```bash
python train.py finetune  # 18実験が自動的に実行される
```

### 設定のカスタマイズ

設定ファイル（`configs/finetune.yaml`, `configs/pretrain.yaml`）で以下を調整可能：
- モデルアーキテクチャ
- データパス
- データ拡張
- ハイパーパラメータ
- グリッドサーチするパラメータ
- ロギング設定

## ワークフロー例

### 1. データ準備

```bash
# データを配置
# Pre-training用（ラベル不要）
data/raw/pretrain/
  ├── sample1.png
  ├── sample2.png
  └── ...

# Fine-tuning用（クラス別フォルダ）
data/processed/train/
  ├── class1/
  │   ├── sample1.png
  │   └── sample2.png
  ├── class2/
  └── ...
```

### 2. 事前学習

```bash
# configs/pretrain.yamlを編集してから実行
python train.py pretrain
```

### 3. ファインチューニング

```bash
# configs/finetune.yamlで事前学習済みモデルのパスを指定してから実行
python train.py finetune
```

### 4. ハイパーパラメータ探索（グリッドサーチ）

```bash
# configs/finetune.yamlのgrid_searchセクションに複数値を指定
# 例:
# grid_search:
#   training:
#     learning_rate: [0.0001, 0.001, 0.01]
#     batch_size: [32, 64]

# そのまま実行すれば自動的にグリッドサーチ
python train.py finetune
```

## 実験追跡

### TensorBoard

```bash
# TensorBoardを起動
tensorboard --logdir logs/

# ブラウザでアクセス
# http://localhost:6006
```

### Weights & Biases

設定ファイルで`wandb.enabled: true`にすると、W&Bダッシュボードで実験を追跡できます：

- リアルタイムメトリクス可視化
- ハイパーパラメータ比較
- モデル性能の比較
- 実験の再現

## 開発

### テスト実行

```bash
pytest __test__/
```

### コードフォーマット

```bash
# Black
black src/

# isort
isort src/

# Flake8
flake8 src/
```

## トラブルシューティング

### CUDA Out of Memory

バッチサイズを減らす、または`mixed_precision: true`を設定してください。

### W&Bログインエラー

```bash
wandb login
# または
export WANDB_API_KEY=your_api_key
```

### データセットが見つからない

`configs/*.yaml`でデータパスが正しく設定されているか確認してください。

## 参考

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Self-Supervised Learning Overview](https://arxiv.org/abs/2304.12210)

## ライセンス

TODO: ライセンスを追加

## 貢献

TODO: 貢献ガイドラインを追加
