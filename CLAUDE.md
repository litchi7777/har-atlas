# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**har-foundation** - Human Activity Recognition (HAR) プロジェクト

**目的**: Self-Supervised Learning (SSL) を用いた事前学習とファインチューニングによる行動クラス分類

**技術スタック**:
- PyTorch 2.0+
- Weights & Biases (実験追跡)
- TensorBoard (ログ可視化)

Repository: `git@github.com:litchi7777/har-foundation.git`

---

# AI駆動開発 共通ガイドライン

思考は英語で行い、最終的な出力は必ず日本語で提供してください。

## 開発の基本理念
- 動くコードを書くだけでなく、品質・保守性・安全性を常に意識する
- プロジェクトの段階（プロトタイプ、MVP、本番環境）に応じて適切なバランスを取る
- 問題を見つけたら放置せず、必ず対処または明示的に記録する
- ボーイスカウトルール：エラーを見つけた時よりも良い状態で残す

## エラーハンドリングの原則
- 関連が薄く見えるエラーでも必ず解決する
- エラーの抑制（@ts-ignore、try-catch で握りつぶす等）ではなく、根本原因を修正
- 早期にエラーを検出し、明確なエラーメッセージを提供
- エラーケースも必ずテストでカバーする
- 外部APIやネットワーク通信は必ず失敗する可能性を考慮

## コード品質の基準
- DRY原則：重複を避け、単一の信頼できる情報源を維持
- 意味のある変数名・関数名で意図を明確に伝える
- プロジェクト全体で一貫したコーディングスタイルを維持
- 小さな問題も放置せず、発見次第修正（Broken Windows理論）
- コメントは「なぜ」を説明し、「何を」はコードで表現

## テスト規律
- テストをスキップせず、問題があれば修正する
- 実装詳細ではなく振る舞いをテスト
- テスト間の依存を避け、任意の順序で実行可能に
- テストは高速で、常に同じ結果を返すように
- カバレッジは指標であり、質の高いテストを重視

## 保守性とリファクタリング
- 機能追加と同時に既存コードの改善を検討
- 大規模な変更は小さなステップに分割
- 使用されていないコードは積極的に削除
- 依存関係は定期的に更新（セキュリティと互換性のため）
- 技術的負債は明示的にコメントやドキュメントに記録

## セキュリティの考え方
- APIキー、パスワード等は環境変数で管理（ハードコード禁止）
- すべての外部入力を検証
- 必要最小限の権限で動作（最小権限の原則）
- 不要な依存関係を避ける
- セキュリティ監査ツールを定期的に実行

## パフォーマンスの意識
- 推測ではなく計測に基づいて最適化
- 初期段階から拡張性を考慮
- 必要になるまでリソースの読み込みを遅延
- キャッシュの有効期限と無効化戦略を明確に
- N+1問題やオーバーフェッチを避ける

## 信頼性の確保
- タイムアウト処理を適切に設定
- リトライ機構の実装（指数バックオフを考慮）
- サーキットブレーカーパターンの活用
- 一時的な障害に対する耐性を持たせる
- 適切なログとメトリクスで可観測性を確保

## プロジェクトコンテキストの理解
- ビジネス要件と技術要件のバランスを取る
- 現在のフェーズで本当に必要な品質レベルを判断
- 時間制約がある場合でも、最低限の品質基準を維持
- チーム全体の技術レベルに合わせた実装選択

## トレードオフの認識
- すべてを完璧にすることは不可能（銀の弾丸は存在しない）
- 制約の中で最適なバランスを見つける
- プロトタイプなら簡潔さを、本番なら堅牢性を優先
- 妥協点とその理由を明確にドキュメント化

## ファイル管理とディレクトリ構造
- **一時的なテストコード**: `.claude/`ディレクトリに配置
  - デバッグ用スクリプト、検証コード、実験的なコードなど
  - プロジェクトの正式なコードベースには含めない一時的なファイル
  - 例: `.claude/test_*.py`, `.claude/debug_*.py`, `.claude/scratch.py`
- **正式なテスト**: `__test__/`ディレクトリに配置（pytest実行対象）
- `.claude/`ディレクトリは`.gitignore`に追加し、バージョン管理から除外

## Git運用の基本
- コンベンショナルコミット形式を使用（feat:, fix:, docs:, test:, refactor:, chore:）
- コミットは原子的で、単一の変更に焦点を当てる
- 明確で説明的なコミットメッセージを英語で記述
- main/masterブランチへの直接コミットは避ける

## コードレビューの姿勢
- レビューコメントは建設的な改善提案として受け取る
- 個人ではなくコードに焦点を当てる
- 変更の理由と影響を明確に説明
- フィードバックを学習機会として歓迎

## デバッグのベストプラクティス
- 問題を確実に再現できる手順を確立
- 二分探索で問題の範囲を絞り込む
- 最近の変更から調査を開始
- デバッガー、プロファイラー等の適切なツールを活用
- 調査結果と解決策を記録し、知識を共有

## 依存関係の管理
- 本当に必要な依存関係のみを追加
- package-lock.json等のロックファイルを必ずコミット
- 新しい依存関係追加前にライセンス、サイズ、メンテナンス状況を確認
- セキュリティパッチとバグ修正のため定期的に更新

## ドキュメントの基準
- READMEにプロジェクトの概要、セットアップ、使用方法を明確に記載
- ドキュメントをコードと同期して更新
- 実例を示すことを優先
- 重要な設計判断はADR (Architecture Decision Records)で記録

## 継続的な改善
- 学んだことを次のプロジェクトに活かす
- 定期的に振り返りを行い、プロセスを改善
- 新しいツールや手法を適切に評価して取り入れる
- チームや将来の開発者のために知識を文書化

---

## Development Commands

### 環境セットアップ

```bash
# 仮想環境作成と依存関係インストール
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# W&Bセットアップ（オプション）
wandb login
```

### トレーニング

**重要**: トレーニングは必ず以下のコマンドを使用してください：

```bash
# Fine-tuning（ファインチューニング）
python train.py finetune

# Pre-training（事前学習）
python train.py pretrain
```

**動作の詳細**:
- 設定ファイルに`grid_search`セクションがある場合、**パラメータが1つでも複数でも**、必ず`run_experiments.py`を使用して実験を実行します
- これにより、`grid_search`で指定したパラメータが確実に適用されます
- `grid_search`セクションがない場合のみ、直接トレーニングスクリプトを実行します

**パラメータ適用の仕組み**:
```yaml
# 基本設定
model:
  pretrained_path: null  # ベースはnull

# grid_searchセクションでオーバーライド
grid_search:
  model:
    pretrained_path: ["path/to/checkpoint.pth"]  # 1つでもrun_experiments.pyが使用される

# → run_experiments.pyが設定を生成し、pretrained_pathが確実に適用される
```

**禁止事項**:
- ~~`python src/training/pretrain.py`~~ (直接実行しない)
- ~~`python src/training/finetune.py`~~ (直接実行しない)
- ~~`python src/training/run_experiments.py`~~ (直接実行しない)

**理由**: `train.py`経由で実行することで、grid_searchセクションの有無を自動検出し、適切なスクリプトを選択します。

### ログと可視化

```bash
# TensorBoard起動
tensorboard --logdir logs/

# 実験結果の確認（複数実験実行後）
cat experiments/run_*/summary.json
```

### テストとコード品質

```bash
# テスト実行
pytest __test__/

# コードフォーマット
black src/
isort src/

# リンター
flake8 src/
```

### 単一テストの実行

```bash
# 特定のテストファイル
pytest __test__/test_models.py

# 特定のテスト関数
pytest __test__/test_models.py::test_ssl_model_forward
```

## Architecture

### 2段階学習アプローチ

このプロジェクトは以下の2段階で学習を行います：

1. **Pre-training Phase（事前学習）**
   - ラベルなしデータで自己教師あり学習（SSL）
   - SimCLR、MoCo、BYOL等のSSL手法を使用
   - エンコーダーを学習し、一般的な特徴表現を獲得
   - 出力: 事前学習済みエンコーダー（`experiments/pretrain/run_*/checkpoints/best_model.pth`）

2. **Fine-tuning Phase（ファインチューニング）**
   - ラベル付きデータで教師あり学習
   - 事前学習済みエンコーダーをロード
   - クラス分類ヘッドを追加してファインチューニング
   - 出力: 分類モデル（`experiments/finetune/run_*/checkpoints/best_model.pth`）

### コードベース構造

#### `src/models/model.py`

- **SSLModel**: 事前学習用モデル
  - エンコーダー（ResNet18/50等）
  - プロジェクションヘッド（対照学習用）

- **ClassificationModel**: ファインチューニング用モデル
  - 事前学習済みエンコーダー
  - 分類ヘッド
  - バックボーンの凍結オプション

#### `src/data/dataset.py`

- **PretrainDataset**: SSL用データセット
  - 同じサンプルの2つの拡張ビューを返す
  - ラベル不要

- **FinetuneDataset**: 分類用データセット
  - クラスラベル付きデータ
  - クラス別フォルダ構造を想定

#### `src/training/`

- **pretrain.py**: Pre-trainingスクリプト
  - SSLトレーニングループ
  - W&B統合
  - チェックポイント保存

- **finetune.py**: Fine-tuningスクリプト
  - 教師ありトレーニングループ
  - 評価メトリクス計算
  - Early stopping

- **run_experiments.py**: グリッドサーチによる複数実験管理
  - パラメータの全組み合わせを自動生成
  - 実験結果の自動集約

### 設定ファイル管理

設定ファイル（YAML）で全てのハイパーパラメータを管理：

- **configs/pretrain.yaml**: Pre-training設定（グリッドサーチ対応）
- **configs/finetune.yaml**: Fine-tuning設定（グリッドサーチ対応）

各ファイルには以下が含まれる：
- 基本設定（モデル、データ、トレーニングパラメータ等）
- `grid_search`セクション：グリッドサーチするパラメータ
- `settings`セクション：実験管理設定

グリッドサーチの仕組み：
```python
# grid_searchセクションで指定したパラメータが基本設定を上書き
base_config + grid_search_parameters = experiments
# 例: learning_rate [0.001, 0.01] × batch_size [32, 64] = 4実験
# 例: learning_rate [0.001] = 1実験（単一実験）
```

### データフロー

```
生データ (data/raw/)
    ↓
前処理 (ユーザー実装)
  - スケーリング適用（加速度のみ: m/s² → G）
  - float16変換
    ↓
処理済みデータ (data/processed/)
    ↓
PretrainDataset → Pre-training → 事前学習済みモデル (experiments/pretrain/run_*)
    ↓
FinetuneDataset + 事前学習済みモデル → Fine-tuning → 分類モデル (experiments/finetune/run_*)
```

### データ前処理の詳細

#### 加速度センサーの単位統一

すべての加速度センサーデータをG（重力加速度）単位に統一：

- **目的**: 異なるデータセット間での加速度スケールの統一
- **対象データセット**: DSADS、MHEALTH（デフォルトでm/s²単位）
- **変換係数**: `scale_factor = 9.8`（m/s² → G）
- **適用対象**: ACCモダリティのみ（GYRO、MAG、ECGは対象外）
- **実装場所**:
  - 設定: [src/data/dataset_info.py](src/data/dataset_info.py) の`DATASETS`辞書
  - 適用: 各データセットのpreprocessor（`extract_features`メソッド内）

```python
# 例: DSADSとMHEALTHの設定
DATASETS = {
    "DSADS": {
        "scale_factor": 9.8,  # m/s^2 -> G に変換
        ...
    },
    "MHEALTH": {
        "scale_factor": 9.8,  # m/s^2 -> G に変換
        ...
    }
}
```

#### データ型の最適化

- **保存形式**: float16（メモリ効率化）
- **精度**: 約3桁の有効数字（センサーデータには十分）
- **メリット**: ストレージ容量とメモリ使用量を約50%削減

### 実験追跡

- **ローカル**: TensorBoard（`logs/`）
- **クラウド**: Weights & Biases
- **複数実験**: JSON summary（`experiments/run_*/summary.json`）

すべてのメトリクス、ハイパーパラメータ、モデルチェックポイントが追跡可能。

### 重要な設計判断

1. **設定ファイルベースのアプローチ**: コード変更なしでハイパーパラメータ調整可能
2. **モジュラー設計**: データセット、モデル、トレーニングループが独立
3. **W&B統合**: オプショナル（無効化しても動作）
4. **2段階学習**: Pre-trainingとFine-tuningを完全に分離
5. **実験再現性**: シード固定、設定ファイル保存、チェックポイント管理
