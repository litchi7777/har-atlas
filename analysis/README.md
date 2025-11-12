# Analysis - データ分析・可視化フレームワーク

このディレクトリには、HARプロジェクトのための包括的な分析ツール群が含まれています。

## 📁 ディレクトリ構造

```
analysis/
├── utils.py                    # 共通ユーティリティ関数
├── dataset_distribution.py     # データセット分布の可視化
├── visualize_embeddings.py     # 特徴空間の可視化（t-SNE/UMAP）
├── model_performance.py        # モデル性能の詳細分析
├── feature_analysis.py         # 特徴量の詳細分析
├── data_quality.py             # データ品質の評価
└── figures/                    # 出力画像
    ├── performance/
    ├── features/
    ├── data_quality/
    └── embeddings/
```

## 🎯 分析スクリプトの概要

### 1. **共通ユーティリティ** (`utils.py`)

全スクリプトで共有される関数群：
- データセット・モデルの読み込み
- 特徴抽出
- プロジェクト構造のナビゲーション
- 出力フォーマット

### 2. **データセット分布分析** (`dataset_distribution.py`)

各データセット・身体部位のセンサーデータ分布を可視化し、データセット間の特性の違いを分析します。

#### 主な機能

1. **個別データセット分析**
   - 各軸の振幅分布（ヒストグラム）
   - 統計量の箱ひげ図
   - 活動クラス別の時系列サンプル
   - クラス分布（サンプル数）

2. **データセット間比較**
   - 信号の変動性（標準偏差）
   - 活動クラス数
   - データセットサイズ
   - 振幅分布の比較（バイオリンプロット）

#### 使用方法

```bash
# 個別データセット・部位の分析
python analysis/dataset_distribution.py --dataset dsads --location Torso

# 全データセット・部位を分析（個別の可視化のみ）
python analysis/dataset_distribution.py --all

# 全データセット・部位を分析 + クロスデータセット比較
python analysis/dataset_distribution.py --all --compare

# 比較のみ生成
python analysis/dataset_distribution.py --compare
```

#### 出力例

```
analysis/figures/
├── dsads_Torso_distribution.png          # 個別データセット分析
├── mhealth_Chest_distribution.png
├── pamap2_hand_distribution.png
├── ...
└── cross_dataset_comparison.png          # データセット間比較
```

#### 可視化の内容

**個別データセット分析図（例: `dsads_Torso_distribution.png`）:**
- 左上: 各軸（X/Y/Z）の振幅分布ヒストグラム
- 右上: 各軸の統計量（箱ひげ図）
- 左下: 活動クラス別サンプル時系列（X軸のみ）
- 右下: 活動クラス分布（サンプル数の棒グラフ）

**クロスデータセット比較図（`cross_dataset_comparison.png`）:**
- 左上: データセット別の信号変動性（標準偏差）
- 右上: データセット別の活動クラス数
- 左下: データセット別のサンプル数
- 右下: 振幅分布の比較（バイオリンプロット、最大8データセット）

## 📈 統計情報の出力

スクリプトを実行すると、ターミナルに以下の統計情報が出力されます：

```
================================================================================
Dataset: DSADS - Torso
================================================================================
Data shape: (9120, 3, 125) (N_samples, N_channels, N_timesteps)
Label shape: (9120,)
Number of samples: 9120
Time steps: 125

Number of classes: 19
Class distribution:
  Sitting                       :    480 samples ( 5.26%)
  Standing                      :    480 samples ( 5.26%)
  Lying(Back)                   :    480 samples ( 5.26%)
  ...

Sensor data statistics (all axes):
  X-axis:
    Mean:     0.0234 G
    Std:      0.4567 G
    Min:     -2.3456 G
    Max:      3.1234 G
    Range:    5.4690 G
  Y-axis:
    ...
  Z-axis:
    ...
```

## 🎯 分析の目的

1. **データセット特性の理解**
   - 各データセットのセンサーデータ分布を把握
   - 活動クラスのバランスを確認
   - データ品質の評価

2. **データセット間の違いの可視化**
   - Fine-tuning性能の差（0.58～0.87）の原因を探る
   - データ収集方法、ノイズ特性の違いを視覚的に確認
   - データセット選択の根拠を提供

3. **事前学習データの選択**
   - 多様性の評価
   - 類似性の定量化
   - データ統合の戦略検討

## 📝 依存関係

スクリプトは以下のライブラリを使用します：

```python
numpy
matplotlib
seaborn
```

プロジェクトの`requirements.txt`に含まれているため、追加のインストールは不要です。

## 🔧 カスタマイズ

スクリプトを編集して、以下のカスタマイズが可能です：

- **図のスタイル**: `sns.set_style()`, `sns.set_context()`
- **カラーマップ**: `plt.cm.*`
- **サンプリング数**: `n_samples_per_class`, ランダムサンプリング数
- **出力形式**: `.png`, `.pdf`, `.svg`

### 3. **特徴空間の可視化** (`visualize_embeddings.py`)

事前学習済みモデルの特徴表現を次元削減して可視化します。

**機能**:
- t-SNE/UMAPによる2次元可視化
- データセット別・身体部位別・アクティビティ別の色分け
- 複数モデルの比較

**使用例**:
```bash
# 単一モデルの可視化
python analysis/visualize_embeddings.py \
  --model experiments/pretrain/run_*/exp_0/models/best_model.pth \
  --method umap --color-by body_part

# 複数モデルの比較
python analysis/visualize_embeddings.py \
  --models exp_0/models/best_model.pth exp_1/models/best_model.pth \
  --compare
```

### 4. **モデル性能分析** (`model_performance.py`)

学習済みモデルの性能を多角的に分析します。

**機能**:
- 学習曲線（損失・精度の推移）
- 混同行列とクラス別メトリクス
- データセット別・身体部位別の性能比較
- 複数実験の比較

**使用例**:
```bash
# 単一実験の分析
python analysis/model_performance.py \
  --experiment experiments/finetune/run_*/exp_0

# 複数実験の比較
python analysis/model_performance.py \
  --experiments exp_0 exp_1 exp_2 --compare
```

### 5. **特徴量分析** (`feature_analysis.py`)

エンコーダーが学習した特徴表現を詳細に分析します。

**機能**:
- 特徴量の活性化パターン
- 特徴量の重要度分析（Fisher判別比）
- レイヤー別の特徴分布
- クラス別・データセット別の特徴統計

**使用例**:
```bash
# 事前学習モデルの特徴分析
python analysis/feature_analysis.py \
  --model experiments/pretrain/run_*/exp_0/models/best_model.pth

# 複数モデルの比較
python analysis/feature_analysis.py \
  --models model1.pth model2.pth --compare
```

### 6. **データ品質分析** (`data_quality.py`)

データセットの品質を多角的に評価します。

**機能**:
- 欠損値・異常値の検出
- クラスバランスの評価
- 信号品質の評価（SNR、周波数特性）
- データセット間の品質比較

**使用例**:
```bash
# 単一データセットの分析
python analysis/data_quality.py --dataset dsads --location Torso

# 全データセットの比較
python analysis/data_quality.py --all --compare
```

---

## 🚀 統一エントリーポイント

すべての分析を1つのコマンドで実行できる統一インターフェース：

```bash
# メインの分析スクリプト
python analysis/analyze.py <analysis_type> [options]
```

**利用可能な分析タイプ**:
- `data`: データセット分布と品質分析
- `embeddings`: 特徴空間の可視化
- `performance`: モデル性能分析
- `features`: 特徴量の詳細分析
- `all`: 全分析を実行

**使用例**:
```bash
# データセット分析
python analysis/analyze.py data --dataset dsads --location Torso

# 特徴空間の可視化
python analysis/analyze.py embeddings \
  --model experiments/pretrain/run_*/exp_0/models/best_model.pth

# モデル性能分析
python analysis/analyze.py performance \
  --experiment experiments/finetune/run_*/exp_0

# 全分析を実行
python analysis/analyze.py all \
  --model experiments/pretrain/run_*/exp_0/models/best_model.pth \
  --experiment experiments/finetune/run_*/exp_0
```

---

## 📈 出力

すべての分析結果は `analysis/figures/` 以下に保存されます：

```
analysis/figures/
├── dataset_distribution/    # データセット分布の図
├── data_quality/           # データ品質レポート
├── embeddings/             # 特徴空間の可視化
├── performance/            # モデル性能分析
└── features/               # 特徴量分析
```

---

## 💡 今後の拡張案

- [ ] インタラクティブな可視化（Plotly）
- [ ] 自動レポート生成（PDF/HTML）
- [ ] A/Bテスト機能（モデル比較の統計的検定）
- [ ] リアルタイムモニタリング（学習中の可視化）
- [ ] カスタム分析スクリプトのプラグイン機構
