# Motion Primitive Foundation Model via Partial Label Learning for HAR

**最終更新**: 2025-11-21

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
Body Part別にPrototypeを学習し、Atomic共有でActivity間の類似性を自動判定。

### **差別化の核心**
- ✅ **HAR × Partial Label Learning** (世界初)
- ✅ **Atomic Motion自動発見** (window-level labelなし)
- ✅ **3階層Atlas** (Complex/Simple/Atomic)
- ✅ **Body Part別学習** (独立Prototype空間)
- ✅ **Atomic共有によるsoft positive** (variants問題を自動解決)

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

→ 3階層Atlas + Atomic共有で解決
```

---

## 🌟 提案手法

### **Atlas構造（v3: Motion-based）**

```
Level 0: Complex Activity (baseball, cooking, commuting, ...)
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

### **Atlas JSON構造**

```json
{
  "activities": {
    "baseball": {
      "level": 0,
      "children": ["walking", "running", "throwing", "catching"]
    },
    "walking": {
      "level": 1,
      "atomic_motions": {
        "wrist": ["arm_swing", "periodic_swing"],
        "hip": ["vertical_oscillation", "lateral_sway"],
        "chest": ["torso_rotation"]
      }
    },
    "walking_treadmill": {
      "level": 1,
      "atomic_motions": {
        "wrist": ["arm_swing", "periodic_swing"],
        "hip": ["vertical_oscillation", "lateral_sway"],
        "chest": ["torso_rotation"]
      }
    },
    "running": {
      "level": 1,
      "atomic_motions": {
        "wrist": ["arm_swing", "high_frequency_swing"],
        "hip": ["vertical_oscillation", "high_impact"],
        "chest": ["torso_rotation"]
      }
    }
  }
}
```

### **ポイント**
- `walking_treadmill`と`walking`は同じatomic_motionsを持つ → **自動的にsoft positive**
- `walking`と`running`は一部共有（arm_swing, vertical_oscillation）→ **弱いsoft positive**
- variants問題は**Atlasの階層構造ではなくAtomic共有で解決**

---

## 🔬 学習アルゴリズム

### **3つのLoss**

```
L_total = λ0 * L_complex + λ1 * L_simple + λ2 * L_atomic

λ0 < λ1 < λ2 (階層が深いほど重視)
例: λ0=0.1, λ1=0.3, λ2=0.6
```

| Loss | Scope | Positive | Negative |
|------|-------|----------|----------|
| L_complex | データセット内 | 同じComplex Activity | 違うComplex Activity |
| L_simple | データセット内 | 同じSimple Activity | 違うSimple Activity |
| L_atomic | **全データセット横断** | Atomic共有度で連続重み | Atomic共有なし |

```
1. Complex Activity Loss (Level 0)
   - Scope: データセット内
   - 通常のContrastive Loss（hard label）
   - 重み λ0: 弱（内部に多様なSimple Activityを含むため）

2. Simple Activity Loss (Level 1)
   - Scope: データセット内
   - 通常のContrastive Loss（hard label）
   - 重み λ1: 中

3. Atomic Motion Loss (Level 2) - 核心
   - Scope: 全データセット横断（Foundation Modelの汎化性能の源泉）
   - Body Part別にPiCOで学習（wrist同士、hip同士で比較）
   - Atomic共有度でsoft positive（連続重み 0〜1）
   - 重み λ2: 強（最も細かい粒度）
```

### **Soft Positive判定（Atomic共有）**

```
Activity A: atomic_motions = [arm_swing, periodic_swing]
Activity B: atomic_motions = [arm_swing, wrist_rotation]

共有: [arm_swing] → 1個

Soft positive weight = 共有数 / max(|A|, |B|) = 1/2 = 0.5
```

### **Body Part別学習**

```
- 共有エンコーダー（全Body Part共通）
- Body Part別Prototype空間（wrist/hip/chest独立）
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
- w/o Soft positive (hard only)

---

## 📅 タイムライン

### Week 1: Atlas構築
- 19データセットのラベル + Body Part情報収集
- LLMでAtlas構築（Complex/Simple/Atomic 3階層）
- 人間評価 (>70%)

### Week 2: 3階層Loss実装
- L_complex, L_simple: データセット内Contrastive Loss
- L_atomic: クロスデータセットPiCO Loss
- Atomic共有度によるsoft positive重み計算

### Week 3: 学習パイプライン
- Body Part別Prototype空間の実装
- 19データセット統合学習
- λ0, λ1, λ2のチューニング

### Week 4: 評価
- Atomic発見精度（手動100 window）
- LODO評価

### Week 5: Ablation
- w/o L_complex
- w/o L_simple
- w/o クロスデータセット（L_atomicをデータセット内のみ）
- w/o Soft positive (hard only)

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

- **2025-11-21**:
  - Atlas v3完成（Motion-based、69 Atomic Motions）
  - 姿勢ベース→動作ベースに統一（静的Activityはゼロショット対象外）
  - Body Part Taxonomy整備（head/wrist/hip/chest/leg + forearm/thigh/calf/ankle）
  - 14データセットのActivity Mapping完成
  - 3階層Loss設計を確定:
    - L_complex, L_simple: データセット内Contrastive
    - L_atomic: 全データセット横断PiCO

- **2025-11-20**:
  - PiCO (Partial Label Learning) を核心手法として採用
  - Motion Primitive自動発見を中心課題に設定

---

## 📌 Next Actions

1. ~~19データセットのラベル + Body Part情報収集~~ ✅
2. ~~Atlas構築（Complex/Simple/Atomic 3階層）~~ ✅ (v3: 69 Atomic Motions)
3. **人間評価（目標 >70%）** ← 次のステップ
4. **3階層Loss実装（L_complex, L_simple, L_atomic）**

---

## 📁 Atlas関連ファイル

```
docs/atlas/
├── atlas_v3.json                    # Atomic Motion定義 (69種)
├── dataset_activity_mapping_v3.json # 14データセット × Activity → Atomic
└── body_part_taxonomy.json          # Body Part分類 (8カテゴリ)
```

---

**核心の貢献**: Window-level labelなしでAtomic Motionを自動発見

**技術的ポイント**:
- 3階層Atlas（Complex/Simple/Atomic）
- Motion-based Atomic定義（姿勢は含まない）
- Body Part別Prototype学習
- Atomic共有によるsoft positive（variants自動解決）
- 動的Activityに特化したゼロショット認識
