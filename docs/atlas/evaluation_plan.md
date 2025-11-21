# Atlas v3 評価計画

## 1. ゼロショット評価のスコープ

### 対象Activity（動的）
- locomotion: walking, running, jogging, shuffling, cycling
- stairs: ascending_stairs, descending_stairs, walking_upstairs, walking_downstairs
- exercise: jumping, rope_jumping, exercising_stepper, rowing
- sports: playing_basketball, badminton, football, etc.
- household: vacuum_cleaning, ironing, sweeping, brushing_teeth

### 除外Activity（静的）
```
sitting, sit, sitting_relaxing
standing, stand, standing_still
lying, lying_down, lying_on_back, lying_on_right, lying_side
sleeping, sleep
still
```

### 除外理由
- 静的Activityは姿勢（重力方向との関係）で定義される
- センサー座標系がデータセットごとに異なる
- Atomic Motion（動き）では区別不可能

---

## 2. 人間評価計画

### 目的
Atlas v3のAtomic Motion割り当ての妥当性を検証

### 評価方法

#### Phase 1: Atomic Motion定義の妥当性（目標 >80%）
1. 評価者: 3名（HAR研究者 or 運動学の知識がある人）
2. タスク: 各Atomic Motionの説明が理解可能か評価
3. 基準:
   - 「この説明を読んで、どのような動きか想像できるか？」
   - 5段階評価（1: 全く分からない ～ 5: 明確に分かる）
4. 合格基準: 平均4.0以上

#### Phase 2: Activity-Atomic マッピングの妥当性（目標 >70%）
1. 評価者: 同上3名
2. タスク: 各ActivityのAtomic Motion割り当てが適切か評価
3. 方法:
   - Activity名とBody Partを提示
   - 「このActivityでこのBody Partはどのような動きをするか？」
   - 自由記述 → Atlas割り当てと比較
4. 評価指標:
   - Precision: 評価者が挙げた動きがAtlasに含まれる割合
   - Recall: Atlasの動きを評価者がカバーする割合
5. 合格基準: F1 > 0.7

#### Phase 3: 動画ベース検証（オプション）
1. 各Activityの動画を視聴
2. 観察された動きとAtlas定義を比較
3. 目視での一致度確認

### サンプリング
- 全Activity（約100種）からランダムに30個選択
- 各Body Partから均等にサンプリング

### 評価シート
```
Activity: walking
Body Part: wrist
Atlas定義: W01_swing_slow (Rhythmic arm swing, 1-2Hz)

Q1: この定義は理解できますか？ [1-5]
Q2: walkingでwristはどのような動きをしますか？（自由記述）
Q3: Atlas定義は適切ですか？ [1-5]
Q4: 不足している動きはありますか？（自由記述）
```

---

## 3. データセットマッピング状況

### マッピング完了（14データセット）
- dsads, mhealth, pamap2, uschad, harth, realdisp
- realworld, selfback, paal, hhar, har70plus
- lara, imsb, forthtrace

### 未マッピング（13データセット）
- capture24, chad, hmp, imwsha, mex
- motionsense, nhanes, openpack, opportunity
- sbrhapt, tmd, ward, wisdm

### 優先度
1. **高**: opportunity, motionsense, wisdm（広く使われる）
2. **中**: capture24, chad, hmp, mex
3. **低**: nhanes, openpack（特殊なラベル体系）

---

## 4. 評価スケジュール

### Week 1
- [ ] 残り13データセットのActivity情報収集
- [ ] マッピング追加（優先度高から）
- [ ] 評価シート作成

### Week 2
- [ ] Phase 1実施（Atomic Motion定義の妥当性）
- [ ] フィードバック反映

### Week 3
- [ ] Phase 2実施（Activity-Atomicマッピング）
- [ ] 結果集計・分析

---

## 5. 成功基準

| 評価項目 | 目標 | 最低ライン |
|---------|------|----------|
| Atomic Motion定義の理解度 | >4.0/5.0 | >3.5/5.0 |
| マッピングF1スコア | >0.70 | >0.60 |
| 評価者間一致度(Kappa) | >0.60 | >0.40 |
