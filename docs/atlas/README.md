# Activity Atlas

Human Activity Recognition用の階層的Activity定義システム。

## ファイル構成

| ファイル | 説明 |
|---------|------|
| `atomic_motions.json` | 69種のAtomic Motion定義（Body Part別） |
| `activity_mapping.json` | 24データセット × Activity → Atomic Motion |
| `body_part_taxonomy.json` | センサー位置 → Body Partカテゴリ |
| `evaluation_plan.md` | 人間評価計画 |

## 階層構造

```
Level 0: Complex Activity (baseball, cooking, ...)
  └→ 複数のSimple Activityを含む

Level 1: Simple Activity (walking, running, ...)
  └→ 直接Atomic Motionに対応

Level 2: Atomic Motion × Body Part (69種)
  └→ head (7), wrist (18), hip (16), chest (11), leg (17)
```

## 設計原則

- **Motion-only**: 検出可能な「動き」のみ（姿勢・向きは含まない）
- **Sensor-agnostic**: センサー座標系に依存しない
- **周波数・振幅で区別**: swing_slow (1-2Hz) vs swing_fast (2-4Hz)

## 使用方法

```python
import json

# Atomic Motion定義の読み込み
with open('atomic_motions.json') as f:
    atlas = json.load(f)

# wristのAtomic Motionを取得
wrist_motions = atlas['atomic_motions']['wrist']

# Activity Mappingの読み込み
with open('activity_mapping.json') as f:
    mapping = json.load(f)

# DSADSのwalking ActivityのAtomic Motionsを取得
walking = mapping['dsads']['activities']['walking']
```

## 注意事項

- 静的Activity（sitting, lying, standing）はゼロショット評価対象外
- センサー座標系の違いにより、姿勢の区別は不可能
