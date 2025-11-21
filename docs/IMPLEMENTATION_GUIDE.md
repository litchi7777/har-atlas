# Granularity & Location-Aware HAR Foundation Model 実装ガイド

**作成日**: 2025-11-17
**対象**: Activity Atlas + Granularity-Aware Contrastive Learningの実装
**研究目標**: 30データセットで学習したHAR Foundation Modelによる汎用Motion Primitive発見

---

## 📋 目次

1. [研究概要](#研究概要)
2. [Activity Atlas構築](#activity-atlas構築)
3. [Granularity-Aware Contrastive Learning](#granularity-aware-contrastive-learning)
4. [Location-Aware Learning](#location-aware-learning)
5. [実装の全体像](#実装の全体像)
6. [学習パイプライン](#学習パイプライン)
7. [評価プロトコル](#評価プロトコル)
8. [実装チェックリスト](#実装チェックリスト)

---

## 🎯 研究概要

### 核心アイデア

**従来のHAR**: 単一データセット、固定センサー位置、均一な粒度のラベル
**本研究**: 30データセット統合、任意センサー位置対応、階層的粒度を考慮したFoundation Model

### 3つの鍵となる技術

1. **Activity Atlas**: LLMで構築する階層的ラベルマップ + センサー位置情報
2. **Granularity-Aware Learning**: ラベル粒度の違いを考慮した対比学習
3. **Location-Aware Learning**: センサー装着位置の違いを特徴空間に反映

### 期待される効果

| 評価指標 | ベースライン | 提案手法 | 改善幅 |
|---------|-------------|---------|--------|
| LODO精度 | 45-50% | 55-60% | +10-15% |
| Cross-Location精度 | 30-35% | 55-65% | +20-30% |
| LLM Atlas品質 | N/A | >75% | 人間評価 |

---

## 🗺️ Activity Atlas構築

### 概要

**Activity Atlas**は、30データセットの全ラベルをLLM（GPT-4）で階層化し、センサー位置情報を統合したナレッジベースです。

### データ構造

```python
{
  "locomotion": {                           # Level 0: Super-class
    "walking": {                            # Level 1: Activity
      "labels": ["walking", "walk", "Walking"],  # データセット間の表記ゆれ
      "locations": {
        "wrist": {
          "primitives": [                   # Level 2: Location-specific primitives
            "arm_swing_natural",
            "arm_forward_backward"
          ],
          "description": "手首位置では腕の振り動作が顕著"
        },
        "hip": {
          "primitives": [
            "leg_forward_step",
            "hip_lateral_movement"
          ],
          "description": "腰位置では脚の前後運動が顕著"
        },
        "pocket": {
          "primitives": [
            "slight_vertical_oscillation",
            "periodic_forward_backward"
          ],
          "description": "ポケット内では振動が抑制され周期的な動きのみ"
        }
      }
    },
    "running": { ... },
    "stairs_up": { ... }
  },
  "daily_living": { ... },
  "sports": { ... }
}
```

### 実装: ActivityAtlasConstructor

```python
# src/models/activity_atlas.py

import json
import openai
from typing import Dict, List, Tuple, Optional


class ActivityAtlasConstructor:
    """
    LLMを使ってActivity Atlasを構築
    """

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.atlas = {}

    def construct_hierarchy(self, datasets_info: Dict) -> Dict:
        """
        全データセットのラベルから階層構造を構築

        Args:
            datasets_info: {
                'DSADS': {
                    'labels': ['sitting', 'standing', 'walking', ...],
                    'sensor_location': 'wrist'
                },
                'MHEALTH': { ... },
                ...
            }

        Returns:
            階層的Activity Atlas
        """
        # Step 1: 全ラベルを収集
        all_labels = []
        label_to_datasets = {}  # ラベルがどのデータセットに出現するか

        for ds_name, info in datasets_info.items():
            for label in info['labels']:
                all_labels.append({
                    'label': label,
                    'dataset': ds_name,
                    'location': info['sensor_location']
                })

                if label not in label_to_datasets:
                    label_to_datasets[label] = []
                label_to_datasets[label].append(ds_name)

        # Step 2: LLMで階層化
        hierarchy = self._llm_hierarchical_clustering(all_labels)

        # Step 3: センサー位置別のprimitiveを追加
        hierarchy = self._add_location_primitives(hierarchy, datasets_info)

        self.atlas = hierarchy
        return hierarchy

    def _llm_hierarchical_clustering(self, labels: List[Dict]) -> Dict:
        """
        LLMでラベルを階層的にクラスタリング
        """
        prompt = f"""
あなたはHuman Activity Recognition (HAR)の専門家です。
以下のセンサーデータから認識される行動ラベルを、階層的に整理してください。

ラベルリスト（JSON形式）:
{json.dumps(labels, indent=2)}

以下の3階層で整理してください:
- Level 0 (Super-class): 大分類 (例: locomotion, daily_living, sports)
- Level 1 (Activity): 中分類 (例: walking, running, sitting)
- Level 2 (Sub-activity): 細分類 (例: walk_slow, walk_fast, walk_upstairs)

出力形式:
{{
  "super_class_name": {{
    "activity_name": {{
      "labels": ["表記ゆれを含む全ラベル"],
      "sub_activities": ["細分化されたラベル"]
    }}
  }}
}}

JSON形式で返してください。
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in human activity recognition and taxonomy construction."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # 安定した出力のため低めに設定
        )

        hierarchy = json.loads(response.choices[0].message.content)
        return hierarchy

    def _add_location_primitives(self, hierarchy: Dict, datasets_info: Dict) -> Dict:
        """
        各活動について、センサー位置別のmotion primitiveを追加
        """
        for super_class, activities in hierarchy.items():
            for activity_name, activity_data in activities.items():

                # この活動が出現するセンサー位置を収集
                locations = set()
                for label in activity_data['labels']:
                    for ds_name, info in datasets_info.items():
                        if label in info['labels']:
                            locations.add(info['sensor_location'])

                # 各位置でのprimitiveをLLMに生成させる
                activity_data['locations'] = {}
                for location in locations:
                    primitives = self._generate_location_primitives(
                        activity_name, location
                    )
                    activity_data['locations'][location] = primitives

        return hierarchy

    def _generate_location_primitives(self, activity: str, location: str) -> Dict:
        """
        特定の活動×センサー位置でのmotion primitiveを生成
        """
        prompt = f"""
活動「{activity}」をセンサー位置「{location}」で観測した場合、
どのようなmotion primitive（基本動作要素）が観測されますか？

以下の形式で3-5個のprimitiveを列挙してください:
{{
  "primitives": ["primitive_1", "primitive_2", ...],
  "description": "この位置での観測の特徴"
}}

例:
活動「walking」、位置「wrist」の場合:
{{
  "primitives": ["arm_swing_forward", "arm_swing_backward", "periodic_vertical_movement"],
  "description": "手首では腕の前後振り動作が周期的に観測される"
}}

JSON形式で返してください。
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in biomechanics and sensor-based activity recognition."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.5
        )

        result = json.loads(response.choices[0].message.content)
        return result

    def get_relation(
        self,
        label1: str,
        label2: str,
        location1: str,
        location2: str
    ) -> Dict:
        """
        2つのラベル間の関係性を判定

        Returns:
            {
                'type': 'same_activity_same_location' | 'parent_child' |
                        'sibling' | 'distant' | 'different_location_same_activity',
                'distance': int (階層的距離, 0=same, 1=parent-child, 2=sibling, 3+=distant)
            }
        """
        # ラベルが階層のどこに位置するか検索
        path1 = self._find_label_path(label1)
        path2 = self._find_label_path(label2)

        if path1 is None or path2 is None:
            return {'type': 'unknown', 'distance': 999}

        # 階層的距離を計算
        common_depth = 0
        for i in range(min(len(path1), len(path2))):
            if path1[i] == path2[i]:
                common_depth = i + 1
            else:
                break

        # 関係性を判定
        if label1 == label2:
            if location1 == location2:
                return {'type': 'same_activity_same_location', 'distance': 0}
            else:
                return {'type': 'different_location_same_activity', 'distance': 0}

        elif common_depth == len(path1) - 1 or common_depth == len(path2) - 1:
            # 片方がもう片方の親
            return {'type': 'parent_child', 'distance': 1}

        elif common_depth == len(path1) - 2 and len(path1) == len(path2):
            # 同じ親を持つ兄弟
            return {'type': 'sibling', 'distance': 2}

        else:
            # 離れた関係
            distance = len(path1) + len(path2) - 2 * common_depth
            return {'type': 'distant', 'distance': distance}

    def _find_label_path(self, label: str) -> Optional[List[str]]:
        """
        ラベルの階層パスを取得
        例: 'walk_slow' -> ['locomotion', 'walking', 'walk_slow']
        """
        for super_class, activities in self.atlas.items():
            for activity_name, activity_data in activities.items():
                if label in activity_data['labels']:
                    return [super_class, activity_name, label]
                if 'sub_activities' in activity_data:
                    if label in activity_data['sub_activities']:
                        return [super_class, activity_name, label]
        return None

    def save(self, path: str):
        """Activity Atlasを保存"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.atlas, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        """Activity Atlasを読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            self.atlas = json.load(f)
```

---

## 🔬 Granularity-Aware Contrastive Learning

### 核心アイデア

**課題**: 異なるデータセットのラベル粒度が異なる
- Dataset A: "locomotion" (粗い)
- Dataset B: "walking" (中間)
- Dataset C: "walk_slow", "walk_fast" (細かい)

**従来の対比学習**: 全て異なるラベルとして扱う → 本来同じ活動なのに遠ざけてしまう

**Granularity-Aware Learning**: 階層的関係を考慮した重み付き対比学習

### 関係性に基づく損失重み

| 関係性 | 例 | Positive/Negative | 重み |
|-------|---|------------------|------|
| Same activity + Same location | walking@wrist vs walking@wrist | Strong Positive | 2.0 |
| Parent-child | locomotion vs walking | Weak Positive | 0.7 |
| Sibling | walking vs running | Hard Negative | 1.5 |
| Different location, Same activity | walking@wrist vs walking@hip | Medium Positive | 0.5 |
| Distant | locomotion vs sitting | Easy Negative | 0.3 |

### 実装: GranularityLocationAwareContrastiveLoss

```python
# src/losses/granularity_aware_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class GranularityLocationAwareContrastiveLoss(nn.Module):
    """
    Granularity & Location-Aware Supervised Contrastive Loss

    特徴:
    - ラベルの階層的関係を考慮
    - センサー位置の違いを考慮
    - 関係性ベースの重み付き損失
    """

    def __init__(
        self,
        atlas,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        relation_weights: Dict[str, float] = None
    ):
        super().__init__()
        self.atlas = atlas
        self.temperature = temperature
        self.base_temperature = base_temperature

        # デフォルトの関係性重み
        if relation_weights is None:
            self.relation_weights = {
                'same_activity_same_location': 2.0,
                'different_location_same_activity': 0.5,
                'parent_child': 0.7,
                'sibling': 1.5,
                'distant': 0.3,
                'unknown': 0.1
            }
        else:
            self.relation_weights = relation_weights

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        locations: torch.Tensor,
        dataset_ids: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            features: [batch_size, feature_dim] L2正規化済み
            labels: [batch_size] ラベルID (文字列をencodingしたもの)
            locations: [batch_size] センサー位置ID
            dataset_ids: [batch_size] データセットID (オプション)

        Returns:
            loss: scalar
            metrics: dict of loss breakdown
        """
        device = features.device
        batch_size = features.shape[0]

        # 関係性マスクと重み行列を構築
        positive_mask = torch.zeros((batch_size, batch_size), device=device)
        negative_mask = torch.zeros((batch_size, batch_size), device=device)
        weight_matrix = torch.zeros((batch_size, batch_size), device=device)

        # 各ペアの関係性を判定
        relation_counts = {k: 0 for k in self.relation_weights.keys()}

        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue

                # Activity Atlasで関係性を取得
                relation = self.atlas.get_relation(
                    labels[i].item(),
                    labels[j].item(),
                    locations[i].item(),
                    locations[j].item()
                )

                relation_type = relation['type']
                weight = self.relation_weights.get(relation_type, 0.1)

                # Positive/Negative判定
                if relation_type in ['same_activity_same_location',
                                     'parent_child',
                                     'different_location_same_activity']:
                    positive_mask[i, j] = 1.0
                else:
                    negative_mask[i, j] = 1.0

                weight_matrix[i, j] = weight
                relation_counts[relation_type] += 1

        # 類似度行列計算
        # features: [B, D], features.T: [D, B]
        # similarity_matrix: [B, B]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Log-sum-exp安定化
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # exp(logits)計算（対角成分除外）
        logits_mask = torch.ones_like(similarity_matrix)
        logits_mask.fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logits_mask

        # Log probability
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Positive pairに対する損失（重み付き平均）
        weighted_log_prob_pos = (positive_mask * weight_matrix * log_prob).sum(1)
        weighted_positive_count = (positive_mask * weight_matrix).sum(1)

        # ゼロ除算回避
        mean_log_prob_pos = weighted_log_prob_pos / (weighted_positive_count + 1e-12)

        # 損失（負の対数尤度）
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[weighted_positive_count > 0].mean()

        # Hard negative mining（sibling関係を強調）
        sibling_mask = torch.zeros_like(negative_mask)
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                relation = self.atlas.get_relation(
                    labels[i].item(), labels[j].item(),
                    locations[i].item(), locations[j].item()
                )
                if relation['type'] == 'sibling':
                    sibling_mask[i, j] = 1.0

        if sibling_mask.sum() > 0:
            # Sibling間の距離を大きくするペナルティ
            sibling_similarity = (sibling_mask * similarity_matrix).sum() / (sibling_mask.sum() + 1e-12)
            hard_negative_penalty = torch.relu(sibling_similarity + 0.2)  # マージン0.2
            loss = loss + 0.5 * hard_negative_penalty

        # メトリクス
        metrics = {
            'loss': loss.item(),
            'num_positives': positive_mask.sum().item() / 2,  # 対称性のため2で割る
            'num_negatives': negative_mask.sum().item() / 2,
            'relation_counts': relation_counts
        }

        return loss, metrics
```

---

## 📍 Location-Aware Learning

### 概要

同じ活動でも、センサー位置が異なれば観測される信号が大きく異なります。

**例: "walking"**
- **Wrist（手首）**: 腕の振り動作が顕著 → 周期的な前後加速度
- **Hip（腰）**: 脚の前後運動 → 上下+前後の複合パターン
- **Pocket（ポケット）**: 振動が抑制 → 小振幅の周期信号

### Location Embeddingの導入

```python
# src/models/location_aware_encoder.py

import torch
import torch.nn as nn


class LocationAwareEncoder(nn.Module):
    """
    センサー位置を考慮したエンコーダー

    アプローチ:
    - センサー位置をembeddingして特徴量に統合
    - Location-specific Attention
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        num_locations: int,
        location_embed_dim: int = 32,
        feature_dim: int = 512
    ):
        super().__init__()

        self.base_encoder = base_encoder

        # Location embedding
        self.location_embedding = nn.Embedding(num_locations, location_embed_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + location_embed_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, time_steps]
            locations: [batch_size] センサー位置ID

        Returns:
            features: [batch_size, feature_dim] Location-aware features
        """
        # Base features
        base_features = self.base_encoder(x)  # [B, feature_dim]

        # Location embedding
        loc_embed = self.location_embedding(locations)  # [B, location_embed_dim]

        # Concatenate and fuse
        combined = torch.cat([base_features, loc_embed], dim=1)
        fused_features = self.fusion(combined)

        # Residual connection
        features = base_features + fused_features

        return features
```

### Location-Specific Data Augmentation

```python
# src/data/augmentations.py

class LocationAwareAugmentation:
    """
    センサー位置に応じたデータ拡張
    """

    def __init__(self):
        self.location_configs = {
            'wrist': {
                'jitter_std': 0.05,      # 手首は動きが大きいので強めのjitter
                'scaling_range': (0.8, 1.2),
                'rotation_prob': 0.5     # 手首の回転は頻繁
            },
            'hip': {
                'jitter_std': 0.03,      # 腰は安定しているので弱め
                'scaling_range': (0.9, 1.1),
                'rotation_prob': 0.2
            },
            'pocket': {
                'jitter_std': 0.02,      # ポケット内は最も安定
                'scaling_range': (0.95, 1.05),
                'rotation_prob': 0.1
            },
            'chest': {
                'jitter_std': 0.03,
                'scaling_range': (0.9, 1.1),
                'rotation_prob': 0.3
            }
        }

    def __call__(self, x: torch.Tensor, location: str) -> torch.Tensor:
        """
        Args:
            x: [channels, time_steps]
            location: センサー位置名

        Returns:
            augmented: [channels, time_steps]
        """
        config = self.location_configs.get(location, self.location_configs['wrist'])

        # Jittering
        if config['jitter_std'] > 0:
            noise = torch.randn_like(x) * config['jitter_std']
            x = x + noise

        # Scaling
        scale = torch.FloatTensor(1).uniform_(*config['scaling_range']).item()
        x = x * scale

        # Rotation (加速度センサーのみ)
        if torch.rand(1).item() < config['rotation_prob']:
            x = self._random_rotation(x)

        return x

    def _random_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """3軸加速度にランダム回転を適用"""
        # 簡易的な2D回転（XY平面）
        theta = torch.rand(1).item() * 2 * 3.14159
        rotation_matrix = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ], dtype=x.dtype, device=x.device)

        # x: [3, T] -> [T, 3] -> rotate -> [T, 3] -> [3, T]
        x_rotated = torch.matmul(x.T, rotation_matrix.T).T
        return x_rotated
```

---

## 🏗️ 実装の全体像

### ディレクトリ構造

```
src/
├── models/
│   ├── activity_atlas.py          # Activity Atlas構築・管理
│   ├── location_aware_encoder.py  # Location-aware encoder
│   ├── foundation_model.py        # 統合モデル
│   └── backbones.py               # 既存バックボーン
├── losses/
│   └── granularity_aware_loss.py  # Granularity-aware loss
├── data/
│   ├── augmentations.py           # Location-aware augmentation
│   ├── dataset.py                 # Dataset loaders
│   └── dataset_info.py            # 30データセット情報
└── training/
    ├── pretrain_granularity.py    # 新しい事前学習スクリプト
    └── finetune.py                # ファインチューニング（既存）
```

### 統合モデル

```python
# src/models/foundation_model.py

import torch
import torch.nn as nn
from .location_aware_encoder import LocationAwareEncoder
from .backbones import ResNet1D
from .activity_atlas import ActivityAtlasConstructor


class GranularityLocationAwareFoundationModel(nn.Module):
    """
    HAR Foundation Model with Granularity & Location Awareness
    """

    def __init__(
        self,
        backbone_type: str = 'resnet18',
        in_channels: int = 3,
        base_feature_dim: int = 512,
        num_locations: int = 8,
        location_embed_dim: int = 32,
        atlas_path: str = None
    ):
        super().__init__()

        # Backbone encoder
        if backbone_type == 'resnet18':
            self.backbone = ResNet1D(
                in_channels=in_channels,
                base_filters=64,
                num_classes=base_feature_dim,
                kernel_size=3,
                stride=2,
                groups=1,
                n_block=2  # ResNet18
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        # Location-aware wrapper
        self.encoder = LocationAwareEncoder(
            base_encoder=self.backbone,
            num_locations=num_locations,
            location_embed_dim=location_embed_dim,
            feature_dim=base_feature_dim
        )

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(base_feature_dim, base_feature_dim),
            nn.ReLU(),
            nn.Linear(base_feature_dim, 128)
        )

        # Activity Atlas
        self.atlas = ActivityAtlasConstructor(api_key="dummy")
        if atlas_path:
            self.atlas.load(atlas_path)

    def forward(
        self,
        x: torch.Tensor,
        locations: torch.Tensor = None,
        return_projection: bool = True
    ):
        """
        Args:
            x: [batch_size, channels, time_steps]
            locations: [batch_size] センサー位置ID
            return_projection: Trueなら投影ベクトルも返す

        Returns:
            features: [batch_size, feature_dim]
            projections: [batch_size, proj_dim] (if return_projection=True)
        """
        # Encode
        features = self.encoder(x, locations)

        if return_projection:
            projections = F.normalize(self.projection_head(features), dim=1)
            return features, projections
        else:
            return features
```

---

## 🚀 学習パイプライン

### Phase 1: Activity Atlas構築（Week 1）

```python
# scripts/construct_atlas.py

import sys
sys.path.insert(0, '/mnt/home/har-foundation')

from src.models.activity_atlas import ActivityAtlasConstructor
from src.data.dataset_info import DATASETS
import json
from pathlib import Path


def main():
    # Step 1: データセット情報収集
    datasets_info = {}
    base_path = Path("har-unified-dataset/data/processed")

    for ds_name, config in DATASETS.items():
        metadata_path = base_path / ds_name / "metadata.json"
        if not metadata_path.exists():
            continue

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        datasets_info[ds_name] = {
            'labels': metadata['class_names'],
            'sensor_location': config.get('sensor_location', 'unknown')
        }

    # Step 2: Activity Atlas構築
    constructor = ActivityAtlasConstructor(
        api_key="YOUR_OPENAI_API_KEY"
    )

    atlas = constructor.construct_hierarchy(datasets_info)

    # Step 3: 保存
    output_path = "data/activity_atlas.json"
    constructor.save(output_path)

    print(f"Activity Atlas saved to {output_path}")
    print(f"Total super-classes: {len(atlas)}")

    # 統計表示
    total_activities = sum(len(activities) for activities in atlas.values())
    print(f"Total activities: {total_activities}")


if __name__ == "__main__":
    main()
```

### Phase 2: Granularity-Aware Pretraining（Week 2-4）

```python
# src/training/pretrain_granularity.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

from src.models.foundation_model import GranularityLocationAwareFoundationModel
from src.losses.granularity_aware_loss import GranularityLocationAwareContrastiveLoss
from src.data.dataset import MultiDatasetLoader
from src.data.augmentations import LocationAwareAugmentation


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    relation_stats = {}

    for batch_idx, batch in enumerate(dataloader):
        x = batch['data'].to(device)          # [B, C, T]
        labels = batch['labels'].to(device)    # [B]
        locations = batch['locations'].to(device)  # [B]
        dataset_ids = batch['dataset_ids'].to(device)  # [B]

        # Forward
        features, projections = model(x, locations, return_projection=True)

        # Loss
        loss, metrics = criterion(projections, labels, locations, dataset_ids)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Pos: {metrics['num_positives']:.0f} "
                  f"Neg: {metrics['num_negatives']:.0f}")

            # W&B logging
            wandb.log({
                'train/loss': loss.item(),
                'train/num_positives': metrics['num_positives'],
                'train/num_negatives': metrics['num_negatives'],
                'train/step': epoch * len(dataloader) + batch_idx
            })

    return total_loss / len(dataloader)


def main():
    # Config
    config = {
        'backbone': 'resnet18',
        'batch_size': 256,
        'learning_rate': 1e-3,
        'epochs': 100,
        'temperature': 0.07,
        'atlas_path': 'data/activity_atlas.json',
        'num_locations': 8
    }

    # W&B init
    wandb.init(project='har-foundation-granularity', config=config)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GranularityLocationAwareFoundationModel(
        backbone_type=config['backbone'],
        num_locations=config['num_locations'],
        atlas_path=config['atlas_path']
    ).to(device)

    # Loss
    criterion = GranularityLocationAwareContrastiveLoss(
        atlas=model.atlas,
        temperature=config['temperature']
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Data
    train_loader = MultiDatasetLoader(
        datasets=list(range(30)),  # 全30データセット
        batch_size=config['batch_size'],
        augmentation=LocationAwareAugmentation()
    )

    # Training loop
    for epoch in range(config['epochs']):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = Path(f"checkpoints/epoch_{epoch+1}.pth")
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, save_path)


if __name__ == "__main__":
    main()
```

---

## 📊 評価プロトコル

### 1. LODO (Leave-One-Dataset-Out)

```python
# scripts/evaluate_lodo.py

def evaluate_lodo(model, target_dataset, all_datasets):
    """
    1つのデータセットをホールドアウトして評価

    Args:
        model: 学習済みモデル
        target_dataset: 評価対象データセット
        all_datasets: 全データセットリスト
    """
    # 訓練データ: target以外の29データセット
    train_datasets = [ds for ds in all_datasets if ds != target_dataset]

    # Fine-tuning on 29 datasets
    model_finetuned = finetune(model, train_datasets, epochs=50)

    # Evaluate on target dataset
    accuracy = evaluate(model_finetuned, target_dataset)

    return accuracy
```

### 2. Cross-Location Transfer

```python
# scripts/evaluate_cross_location.py

def evaluate_cross_location(model, dataset, train_location, test_location):
    """
    異なるセンサー位置でのゼロショット転移評価

    Args:
        dataset: データセット名
        train_location: 訓練時のセンサー位置
        test_location: テスト時のセンサー位置
    """
    # Train on specific location
    train_data = load_data(dataset, location=train_location)
    model_trained = train(model, train_data)

    # Test on different location (zero-shot)
    test_data = load_data(dataset, location=test_location)
    accuracy = evaluate(model_trained, test_data)

    return accuracy
```

### 3. LLM Atlas品質評価（人間評価）

```python
# scripts/evaluate_atlas_quality.py

def human_evaluation_atlas(atlas, num_samples=100):
    """
    Activity Atlasの品質を人間評価

    評価項目:
    1. 階層構造の妥当性（parent-child関係が正しいか）
    2. Sibling関係の妥当性（同じ親の子が適切か）
    3. Location-specific primitiveの妥当性

    Returns:
        agreement_rate: 人間評価者との一致率
    """
    # ランダムサンプリング
    sampled_pairs = sample_label_pairs(atlas, n=num_samples)

    # 評価用UIで人間にラベリングしてもらう
    human_labels = collect_human_labels(sampled_pairs)

    # LLM生成ラベルとの一致率
    llm_labels = [atlas.get_relation(pair[0], pair[1]) for pair in sampled_pairs]
    agreement = compute_agreement(human_labels, llm_labels)

    return agreement
```

---

## ✅ 実装チェックリスト

### Week 1: Activity Atlas構築
- [ ] `ActivityAtlasConstructor`実装
- [ ] 30データセットのラベル+位置情報収集スクリプト
- [ ] LLM APIで階層構造生成（小規模テスト: 5データセット）
- [ ] 人間評価（50サンプル、>75%一致目標）
- [ ] 全30データセットでAtlas構築
- [ ] Atlas保存・可視化

### Week 2-4: Granularity-Aware Pretraining
- [ ] `GranularityLocationAwareContrastiveLoss`実装
- [ ] `LocationAwareEncoder`実装
- [ ] `GranularityLocationAwareFoundationModel`実装
- [ ] `MultiDatasetLoader`実装（30データセット対応）
- [ ] `LocationAwareAugmentation`実装
- [ ] Pretrain実験実行（100 epochs）
- [ ] Loss曲線確認、収束検証

### Week 5-6: Evaluation
- [ ] LODO評価スクリプト実装
- [ ] Cross-Location評価スクリプト実装
- [ ] ベースライン（Atlas無し）との比較
- [ ] Ablation study（Granularity効果、Location効果）

### Week 7: Analysis & Visualization
- [ ] UMAP可視化（特徴空間の階層構造）
- [ ] Confusion matrix（LODO, Cross-Location）
- [ ] Relation-wise performance breakdown
- [ ] Failure case analysis

### Week 8: Paper Writing
- [ ] 実験結果整理
- [ ] 図表作成
- [ ] 原稿執筆
- [ ] IMWUT投稿

---

## 📚 重要な技術決定の記録

### 1. なぜGranularity-Awareか？

**課題**: 30データセットのラベル粒度が統一されていない
- DSADS: 19クラス（細かい）
- PAMAP2: 12クラス（中程度）
- HARTH: 12クラス（中程度）

**従来手法の限界**: 全て異なるラベルとして扱う → "walking"と"locomotion"を遠ざけてしまう

**解決策**: 階層的関係を考慮した重み付き対比学習

### 2. なぜLocation-Awareか？

**課題**: 同じ活動でもセンサー位置で信号が全く異なる

**実験的証拠**: PAALデータセット（wrist/hip/chest/pocket 4位置）
- Wrist→Hip転移: 30-35%精度
- 同一位置: 70-75%精度

**解決策**: Location embeddingで位置情報を特徴量に統合

### 3. なぜLLMでAtlas構築か？

**代替案1**: 手動でラベル階層を作る → 30データセット×平均15クラス=450ラベル、現実的でない

**代替案2**: クラスタリングで自動構築 → 表記ゆれに弱い（"walk", "walking", "Walking"を別物として扱う）

**LLMの利点**:
- 表記ゆれを吸収
- ドメイン知識を活用（"stairs_up"と"stairs_down"がsibling関係だと理解）
- 位置別のprimitiveも生成可能

---

## 🔗 関連ドキュメント

- [RESEARCH_STATUS.md](RESEARCH_STATUS.md): 研究全体の進捗・タイムライン
- [RELATED_WORK_SUMMARY.md](archive/RELATED_WORK_SUMMARY.md): 関連研究（アーカイブ）
- [CLAUDE.md](../CLAUDE.md): プロジェクト全体の指針

---

**最終更新**: 2025-11-17
**次のステップ**: Week 1タスクの実行（Activity Atlas構築開始）
