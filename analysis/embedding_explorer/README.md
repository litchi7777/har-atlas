# Embedding Explorer

インタラクティブな特徴空間分析ツール

## 概要

事前学習済みモデルから抽出した特徴ベクトルをインタラクティブに可視化・分析するWebアプリケーション。

## 機能

- 📊 **データセット選択**: 複数データセットの選択・比較
- 🎯 **アクティビティフィルタ**: 特定の行動クラスのみ表示
- 📍 **Body Partフィルタ**: 手/腕、胴体、脚などを選択
- 🎨 **色分け方法**: dataset / activity / location で切り替え
- 🔧 **次元削減手法**: UMAP / t-SNE / PCA から選択
- ⏱️ **ウィンドウサイズ**: 0.5s / 1.0s / 2.0s / 5.0s
- 🔍 **インタラクティブ**: ズーム・パン・ホバーで詳細情報

## セットアップ

### 1. 依存関係インストール

```bash
pip install flask umap-learn scikit-learn plotly
```

### 2. 特徴ベクトル抽出

**方法1: 一括抽出（推奨）**

全ウィンドウサイズの特徴ベクトルを一度に抽出：

```bash
# Pythonスクリプト版（推奨）
python analysis/embedding_explorer/extract_all_features.py

# または、最新モデルを自動検出
python analysis/embedding_explorer/extract_all_features.py --auto-detect

# または、シェルスクリプト版
bash analysis/embedding_explorer/extract_all_features.sh
```

**方法2: 個別抽出**

各ウィンドウサイズで個別に実行する場合：

```bash
# 5.0s (150 samples)
python analysis/embedding_explorer/extract_features.py \
    --model experiments/pretrain/run_20251111_171703/exp_2/models/checkpoint_epoch_45.pth \
    --max-samples 100

# 2.0s (60 samples)
python analysis/embedding_explorer/extract_features.py \
    --model experiments/pretrain/run_20251112_192545/exp_0/models/checkpoint_epoch_40.pth \
    --max-samples 100

# 1.0s (30 samples)
python analysis/embedding_explorer/extract_features.py \
    --model experiments/pretrain/run_20251112_192545/exp_1/models/checkpoint_epoch_40.pth \
    --max-samples 100

# 0.5s (15 samples)
python analysis/embedding_explorer/extract_features.py \
    --model experiments/pretrain/run_20251112_192545/exp_2/models/checkpoint_epoch_39.pth \
    --max-samples 100
```

**オプション（extract_all_features.py）：**
- `--auto-detect`: 最新のモデルチェックポイントを自動検出
- `--model-5-0s PATH`: 5.0sモデルのパス指定
- `--model-2-0s PATH`: 2.0sモデルのパス指定
- `--model-1-0s PATH`: 1.0sモデルのパス指定
- `--model-0-5s PATH`: 0.5sモデルのパス指定
- `--max-samples N`: 各dataset-locationから最大N個をサンプリング（デフォルト: 100）
- `--max-users N`: 大規模データセット（>100ユーザー）の最大ユーザー数（デフォルト: 20）
- `--output-dir PATH`: 出力ディレクトリ（デフォルト: analysis/embedding_explorer/data）
- `--skip-existing`: 既存ファイルがあればスキップ
- `--device cuda/cpu`: 使用デバイス（デフォルト: cuda）

**処理時間目安：**
- 各ウィンドウサイズで約5-10分
- 全体で約20-40分
- 合計サンプル数: 約10,000-20,000サンプル/ウィンドウサイズ

### 3. Webサーバー起動

```bash
python analysis/embedding_explorer/server.py --port 5000
```

### 4. ブラウザでアクセス

```
http://localhost:5000
```

## 使い方

### 基本設定

1. **Window Size**: 分析したいウィンドウサイズを選択
2. **Reduction Method**: 次元削減手法を選択
   - **UMAP**: バランス良好、グローバル構造も保持（推奨）
   - **t-SNE**: 局所構造に優れる、時間がかかる
   - **PCA**: 高速、シンプル
3. **Color By**: 色分け基準を選択
   - **Dataset**: データセット別
   - **Activity**: 行動クラス別
   - **Location**: 身体部位別

### フィルター

各セクションでチェックボックスを使って表示データを絞り込み：

- **Dataset Filter**: 分析したいデータセットのみ選択
- **Activity Filter**: 特定の行動クラスに絞り込み
- **Location Filter**: 特定の身体部位のみ表示

**Tip**: "Select All" / "Deselect All" ボタンで一括選択/解除

### 可視化生成

1. フィルターとパラメータを設定
2. 🚀 **Generate Visualization** ボタンをクリック
3. 処理完了まで待機（10秒〜数分）
4. インタラクティブな可視化が表示される

### インタラクティブ操作

- **ズーム**: マウスホイールまたはドラッグで選択
- **パン**: ドラッグで移動
- **ホバー**: ポイントにカーソルを合わせると詳細情報表示
- **レジェンド**: クリックで表示/非表示切り替え

## ディレクトリ構造

```
embedding_explorer/
├── extract_features.py      # 特徴ベクトル抽出スクリプト
├── server.py                 # Flaskサーバー
├── templates/
│   └── index.html           # WebUI
├── data/                     # 抽出した特徴ベクトル
│   ├── features_5.0s.npz
│   ├── metadata_5.0s.json
│   ├── features_2.0s.npz
│   ├── metadata_2.0s.json
│   ├── features_1.0s.npz
│   ├── metadata_1.0s.json
│   ├── features_0.5s.npz
│   └── metadata_0.5s.json
└── README.md                 # このファイル
```

## トラブルシューティング

### エラー: "Features file not found"

特徴ベクトルが抽出されていません。`extract_features.py`を実行してください。

### エラー: "No data matches the selected filters"

フィルターが厳しすぎます。より多くのデータセット/アクティビティ/locationを選択してください。

### 処理が遅い

- サンプル数を減らす：`--max-samples 50`
- より速い次元削減手法を使用：PCA
- 選択するデータを減らす（フィルター活用）

## 注意事項

- **メモリ使用量**: 全データで約2-4GB
- **初回起動**: データ読み込みに数秒かかります
- **キャッシュ**: ウィンドウサイズ切り替え時、データは自動キャッシュされます

## 今後の拡張

- [ ] クラスター分析機能
- [ ] 特定領域の詳細分析
- [ ] エクスポート機能（PNG/HTML）
- [ ] 複数ウィンドウサイズの比較表示
- [ ] 統計情報の表示

## ライセンス

このツールはhar-foundationプロジェクトの一部です。
