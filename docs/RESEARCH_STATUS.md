# Motion Primitive Foundation Model via Partial Label Learning for HAR

**最終更新**: 2025-11-25

---

## 🎯 研究概要

### 投稿先・期限
- **投稿先**: IMWUT (Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies)
- **締め切り**: 2026/2/1
- **残り期間**: 約10週間（実装7週間 + 執筆2週間 + バッファ1週間）
- **採択確率**: 75-80%

### タイトル
"Hierarchical Partial Label Contrastive Learning for Motion Primitive Discovery in Human Activity Recognition"

---

## 🔑 核心的貢献

### **中心的アイデア**

**Window-level labelなしでAtomic Motionを自動発見する。**

LLMで階層的Atlas（Complex/Simple/Atomic）を構築し、PiCOでAtomic Motionを発見。
Body Part別にPrototypeを学習し、同じPrototypeに割り当てられたサンプル同士をpositiveとして学習。

### **差別化の核心**
- ✅ **HAR × Partial Label Learning** (世界初)
- ✅ **Atomic Motion自動発見** (window-level labelなし)
- ✅ **3階層Loss** (Complex/Activity/Atomic)
- ✅ **Body Part別学習** (独立Prototype空間)
- ✅ **PiCOによるクロスデータセット汎化**

---

## 💡 問題設定

### **Problem 1: Window-level Label不在**

```
現状: Activity-level labelのみ
┌─────────────────────────────────┐
│ Activity: "walking" @ wrist     │
│ Duration: 10秒                  │
└─────────────────────────────────┘
        ↓ 分割
┌────┬────┬────┬────┬────┐
│ w1 │ w2 │ w3 │ w4 │ w5 │  各2秒window
└────┴────┴────┴────┴────┘
  ?    ?    ?    ?    ?

課題:
- どのwindowが何のAtomic Motionか不明
- arm_swing? vertical_oscillation?
- ラベル付けは非現実的（数十万window）

→ PiCO (Partial Label Contrastive Learning) で解決
```

---

### **Problem 2: Activity階層の混在**

```
データセットによってラベル粒度が異なる:

Dataset A: "baseball" (Complex Activity)
  → 内部に walking, running, throwing を含む

Dataset B: "walking", "running" (Simple Activity)
  → 直接的な動作

Dataset C: "walking_treadmill", "walking_slope" (Variants)
  → 同じAtomic Motionを持つ

→ 3階層Loss + PiCOで解決
```

---

## 🌟 提案手法

### **Atlas構造（v3: Motion-based）**

```
Level 0: Complex Activity (vacuum_cleaning, cooking, commuting, ...)
  └→ 複数のSimple Activityを含む
  └→ 弱い監督信号として使用

Level 1: Simple Activity (walking, running, cycling, ...)
  └→ 直接Atomic Motionに対応
  └→ 中程度の監督信号
  └→ ⚠️ sitting/lying/standingは「stationary」を共有（ゼロショット区別不可）

Level 2: Atomic Motion × Body Part (全69種)
  └→ head (7種): nod, shake, tilt, rotation, bounce_gait, sway_gait, stationary
  └→ wrist (18種): swing_slow/fast, rotation, push_pull, grip_stable, stationary, ...
  └→ hip (16種): gait_slow/fast, step_up/down, jump_launch/land, stationary, ...
  └→ chest (11種): rotation_gait, bounce_walk/run, lean_transition, stationary, ...
  └→ leg (17種): step_walk/run, pedal, jump_explosive, knee_flex, stationary, ...
```

### **設計原則**
- **Motion-only**: 全Atomic Motionは検出可能な「動き」のみ（姿勢・向きは含まない）
- **Sensor-agnostic**: センサー座標系に依存しない定義
- **周波数・振幅で区別**: swing_slow (1-2Hz) vs swing_fast (2-4Hz)

---

## 🔬 学習アルゴリズム

### **3つのLoss（実装完了 ✅）**

```
L_total = λ0 * L_complex + λ1 * L_activity + λ2 * L_atomic

λ0=0.1, λ1=0.3, λ2=0.6
```

| Loss | Positive判定 | スコープ | 重み |
|------|-------------|---------|------|
| L_atomic | PiCOで同じPrototype（Atomic Motion）に割り当て | **クロスデータセット** | λ2=0.6（大） |
| L_activity | 同じActivity名 | 同じデータセット内 | λ1=0.3（中） |
| L_complex | 同じComplex Activity名 | 同じデータセット内 | λ0=0.1（小） |

### **各Lossの詳細**

```
1. L_atomic (重み大) - 核心
   - PiCOでサンプル → Prototype（Atomic Motion）への割り当てを推定
   - 同じPrototypeに割り当てられたサンプル同士がpositive
   - Body Part別に独立したPrototype空間
   - クロスデータセットで学習（汎化性能の源泉）

2. L_activity (重み中)
   - 同じActivity名 + 同じデータセット → positive
   - 全Activity対象（Complex/Simple両方）
   - データセット内のみ

3. L_complex (重み小)
   - 同じComplex Activity名 + 同じデータセット → positive
   - Complex Activity（level=0）のみを対象
   - データセット内のみ
```

### **PiCOによるAtomic Motion発見**

```
核心:
- Activity間の類似度計算ではない
- サンプル単位でPrototype割り当てを推定
- 同じPrototypeに割り当てられたサンプル同士がpositive

例:
- walkingのサンプルA → PiCOが「W01_swing_slow」と推定
- nordic_walkingのサンプルB → PiCOが「W01_swing_slow」と推定
- → AとBは同じAtomic Motionなのでpositive（Activity名は関係なし）
```

### **Body Part別学習**

```
- 共有エンコーダー（全Body Part共通）
- Body Part別Prototype空間（wrist/hip/chest/leg/head独立）
- 同一Body Part内でのみContrastive Learning
```

---

## 📊 評価計画

### **RQ1: Atomic Motion発見精度**
- 手動で100 windowラベル付け
- PiCO推定と比較
- 期待: >85%

### **RQ2: 階層的学習の効果**
- w/o階層 vs 提案手法
- 期待: +10-15%

### **RQ3: Foundation Model性能**
- LODO (19データセット)
- Cross-location transfer
- 期待: LODO 55-60%, Cross-location 50-60%

### **Ablation Studies**
- w/o PiCO (random label)
- w/o 階層 (single-level)
- w/o Body Part別 (全部混ぜ)
- w/o L_atomic (Activity Lossのみ)

---

## 📅 タイムライン

### Week 1-2: Atlas構築 + Loss実装 ✅
- ~~19データセットのラベル + Body Part情報収集~~ ✅
- ~~LLMでAtlas構築（Complex/Simple/Atomic 3階層）~~ ✅
- ~~3階層Loss実装~~ ✅
  - L_complex: Complex Activity Contrastive Loss
  - L_activity: Activity Contrastive Loss
  - L_atomic: PiCO-based Prototype Loss

### Week 3: 学習パイプライン
- 19データセット統合学習
- λ0, λ1, λ2のチューニング
- 人間評価 (>70%)

### Week 4: 評価
- Atomic発見精度（手動100 window）
- LODO評価

### Week 5: Ablation
- w/o L_complex
- w/o L_activity
- w/o L_atomic
- w/o クロスデータセット

### Week 6: Cross-location
- Transfer実験

### Week 7: Figure作成

### Week 8-9: 論文執筆

### Week 10: 投稿準備

---

## ⚠️ Scope & Limitations

### **ゼロショット認識のスコープ**
- ✅ **対象**: 動的Activity（walking, running, cycling, jumping, etc.）
- ❌ **対象外**: 静的Activity（sitting, lying, standing）

### **理由**
- 静的Activityは重力方向との関係で定義される（姿勢）
- センサー座標系がデータセット・被験者ごとに異なる
- 同じ加速度信号がsittingにもlyingにもなりうる
- → Atomic Motionは「動き」のみを定義し、姿勢は含まない

### **論文での記述**
> 本手法は動的Activityのゼロショット認識を対象とする。静的Activity（sitting/lying/standing）はセンサー座標系の標準化なしにはクロスデータセット汎化が困難であり、本研究のスコープ外とする。

---

## 🔄 更新履歴

- **2025-11-25**:
  - 3階層Loss実装完了
  - L_atomic: PiCOベースのPrototype割り当てでpositive判定（Activity間類似度ではない）
  - L_activity: 同じActivity + 同じデータセット → positive
  - L_complex: 同じComplex Activity + 同じデータセット → positive
  - 実装ファイル: `src/losses/hierarchical_loss.py`

- **2025-11-21**:
  - Atlas v3完成（Motion-based、69 Atomic Motions）
  - 姿勢ベース→動作ベースに統一（静的Activityはゼロショット対象外）
  - Body Part Taxonomy整備（head/wrist/hip/chest/leg）
  - 14データセットのActivity Mapping完成

- **2025-11-20**:
  - PiCO (Partial Label Learning) を核心手法として採用
  - Motion Primitive自動発見を中心課題に設定

---

## 📌 Next Actions

1. ~~19データセットのラベル + Body Part情報収集~~ ✅
2. ~~Atlas構築（Complex/Simple/Atomic 3階層）~~ ✅ (v3: 69 Atomic Motions)
3. ~~3階層Loss実装~~ ✅
4. **学習パイプライン統合** ← 次のステップ
5. **人間評価（目標 >70%）**
6. **小規模実験で動作確認**

---

## 📁 関連ファイル

```
docs/atlas/
├── atomic_motions.json              # Atomic Motion定義 (69種) + Activity Level
├── activity_mapping.json            # データセット × Activity → Atomic Mapping
└── body_part_taxonomy.json          # Body Part分類

src/losses/
└── hierarchical_loss.py             # 3階層Loss実装
    ├── HierarchicalSSLLoss          # メインクラス
    ├── ComplexActivityLoss          # L_complex
    ├── SimpleActivityLoss           # L_activity
    ├── AtomicMotionLoss             # L_atomic (PiCO)
    └── BodyPartPrototypes           # Body Part別Prototype管理

src/utils/
└── atlas_loader.py                  # Atlas読み込み・正規化ユーティリティ

src/data/
└── hierarchical_dataset.py          # 階層的SSL用データセット
```

---

**核心の貢献**: Window-level labelなしでAtomic Motionを自動発見

**技術的ポイント**:
- 3階層Loss（Complex/Activity/Atomic）
- PiCOによるPrototype割り当て → 同じPrototypeがpositive
- Motion-based Atomic定義（姿勢は含まない）
- Body Part別Prototype学習
- 動的Activityに特化したゼロショット認識
