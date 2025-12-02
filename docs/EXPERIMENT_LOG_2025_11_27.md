# Experiment Log: ATLAS v2 Zero-shot Evaluation

**Date**: 2025-11-27
**Phase**: Week 3 - Zero-shot評価の検証

---

## 概要

ATLAS v2を用いたMulti-body-part Zero-shot Activity Recognitionの評価を実施。
HARGPTベースラインとの比較準備も完了。

---

## 評価手法

### 1. Ours (ATLAS v2 + Multi-body-part)

**アプローチ**:
1. 学習済みモデルで各Body PartのAtomic Motionを予測
2. 複数Body PartのAtomic Motionを組み合わせてActivity推論
3. 推論方法: LLM (Gemini) または Rule-based

**推論バリエーション**:
- **LLM Multi-device**: 複数Body PartのAtomic → LLMでActivity推論
- **Rule Multi-device**: 複数Body PartのAtomic → ルールベースでActivity推論
- **Rule Single-device**: 単一Body PartのAtomic → ルールベースでActivity推論

### 2. HARGPT Baseline

**アプローチ** (Ji et al., 2024):
- 生IMUデータを10Hzにダウンサンプル
- LLM (Gemini) に直接入力
- Chain-of-Thought promptingでActivity推論

---

## 評価結果

### DSADS (50 samples)

| 手法 | 全体Acc | Dynamic Acc |
|------|---------|-------------|
| LLM Multi-device | 56.0% | 47.5% |
| Rule Multi-device | 58.0% | 47.5% |
| Rule Single-device | 52.0% | 40.0% |

**Multi-device vs Single-device**: +6-7.5pp の改善

### Activity別精度 (DSADS, Rule Multi-device)

| Activity | Acc | N |
|----------|-----|---|
| walking | 100% | 7 |
| running | 100% | 3 |
| jumping | 100% | 4 |
| descending_stairs | 100% | 2 |
| static | 100% | 10 |
| moving_in_elevator | 50% | 2 |
| playing_basketball | 33% | 3 |
| cycling | 17% | 6 |
| ascending_stairs | 0% | 2 |
| rowing | 0% | 3 |
| exercising_on_cross_trainer | 0% | 4 |
| exercising_on_stepper | 0% | 4 |

### 複数データセット・ロケーション結果

| Dataset | Location | Body Part | Total Acc | Dynamic Acc |
|---------|----------|-----------|-----------|-------------|
| dsads | LeftLeg | leg | 60% | 47% |
| dsads | Torso | chest | 56% | 35% |
| mhealth | LeftAnkle | leg | 66% | 45% |
| mhealth | Chest | chest | 62% | 39% |
| pamap2 | ankle | leg | 36% | 34% |
| pamap2 | chest | chest | 46% | 44% |

---

## 分析

### 成功パターン
- **基本的な動作** (walking, running, jumping): 高精度
- **Static activities**: ほぼ完璧に認識

### 課題パターン

1. **階段の昇降混同**
   - ascending_stairs → descending_stairsに誤認識
   - Atomic Motion: L05 (step_ascend) vs L06 (step_descend) の区別が困難
   - 原因: 加速度パターンが類似

2. **Cyclingの認識困難**
   - 精度: 17%
   - leg: L11_knee_flex_deep または L17_stationary に割り当て
   - chest: C11_stationary
   - 問題: 上半身が静的なため、legのみで判断が必要だが精度不足

3. **複合動作の限界**
   - rowing, cross_trainer, stepper: 0%
   - ATLASに専用マッピングがない
   - 既存のAtomic Motionの組み合わせでは表現困難

4. **データセット間のばらつき**
   - pamap2が他より低精度 (36-46%)
   - センサー配置・Activity定義の違いが影響か

---

## 技術的知見

### Multi-device の優位性
- Single-device比で +6-7.5pp
- 複数Body Partの情報統合が有効
- 特にelevator（振動検出）で効果大

### Atomic Motion予測の課題
- 同じActivityでもwindowごとにAtomic Motionが変動
- ascending_stairs でも L06 (descend) が予測されるケースあり
- → Prototype学習の改善が必要

---

## 次のステップ

### 短期 (今週)
1. [ ] HARGPTベースラインの実行・結果取得
2. [ ] 同一サンプルでの公平比較
3. [ ] 結果テーブルの完成

### 中期 (来週)
1. [ ] ATLASマッピングの改善
   - 階段昇降の区別強化
   - cycling用のAtomic Motion追加検討
2. [ ] より多くのデータセットでの評価
3. [ ] Ablation study開始

### 長期
1. [ ] 論文用Figure作成
2. [ ] IMWUT投稿準備

---

## 関連ファイル

```
.claude/
├── analyze_zeroshot_multidevice.py  # Multi-device評価スクリプト
├── hargpt_baseline.py               # HARGPTベースライン
├── compare_methods.py               # 比較評価スクリプト
├── ours_evaluation_results.json     # 評価結果
└── results/
    └── dsads_multillm_50.json       # 詳細結果

docs/atlas/
├── activity_mapping_v2.json         # ATLAS v2マッピング
└── atomic_motions.json              # Atomic Motion定義
```

---

## メモ

- Dynamic Activityに絞ると精度は低下 (47.5%)
- 静的Activityを含めると見かけ上の精度が上がる
- 論文では両方報告するのが妥当
- HARGPTとの比較は同一条件（同じサンプル、同じActivity正規化）で行う必要あり
