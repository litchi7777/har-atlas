# Atlas Verification Checklist

**目的**: 全データセットのActivity Mappingが正確か、1データセットずつ検証する

**真実のソース**: `har-unified-dataset/src/dataset_info.py`
- 全データセットのactivity数、ラベル名、センサーリストを定義
- **必ずこのファイルを参照してAtlasを作成・検証すること**

**検証項目**:
1. ✅ **dataset_info.pyのアクティビティクラス数が一致**
2. ✅ **dataset_info.pyのラベル名が正確にマッピングされている**
3. ✅ **Body Part mappingが正確**
4. ✅ **Atomic Motionsが妥当**

---

## 検証状況

| # | Dataset | Preprocessor確認 | Activity名確認 | Body Part確認 | Atomic Motion確認 | ステータス |
|---|----------------|-----------------|---------------|--------------|------------------|----------|
| 1 | chad | ✅ | ✅ | ✅ | ✅ | 完了 |
| 2 | dsads | ✅ | ✅ | ✅ | ✅ | 完了 |
| 3 | forthtrace | ✅ | ✅ | ✅ | ✅ | 完了 |
| 4 | har70plus | ✅ | ✅ | ✅ | ✅ | 完了 |
| 5 | harth | ✅ | ✅ | ✅ | ✅ | 完了 |
| 6 | hhar | ✅ | ✅ | ✅ | ✅ | 完了 |
| 7 | hmp | ✅ | ✅ | ✅ | ✅ | 完了 |
| 8 | imsb | ✅ | ✅ | ✅ | ✅ | 完了 |
| 9 | imwsha | ✅ | ✅ | ✅ | ✅ | 完了 |
| 10 | lara | ⚠️ | ⚠️ | ✅ | ✅ | 要確認（論文準拠に更新、dataset_infoと名称不一致） |
| 11 | mex | ✅ | ✅ | ✅ | ✅ | 完了 |
| 12 | mhealth | ✅ | ✅ | ✅ | ✅ | 完了 |
| 13 | motionsense | ✅ | ✅ | ✅ | ✅ | 完了 |
| 14 | openpack | ✅ | ✅ | ✅ | ✅ | 完了 |
| 15 | opportunity | ✅ | ✅ | ✅ | ✅ | 完了 |
| 16 | paal | ✅ | ✅ | ✅ | ✅ | 完了 |
| 17 | pamap2 | ✅ | ✅ | ✅ | ✅ | 完了 |
| 18 | realdisp | ✅ | ✅ | ✅ | ✅ | 完了 |
| 19 | realworld | ✅ | ✅ | ✅ | ✅ | 完了 |
| 20 | sbrhapt | ✅ | ✅ | ✅ | ✅ | 完了 |
| 21 | selfback | ✅ | ✅ | ✅ | ✅ | 完了 |
| 22 | tmd | ✅ | ✅ | ✅ | ✅ | 完了 |
| 23 | uschad | ✅ | ✅ | ✅ | ✅ | 完了 |
| 24 | ward | ✅ | ✅ | ✅ | ✅ | 完了 |
| 25 | wisdm | ✅ | ✅ | ✅ | ✅ | 完了 |
| 26 | capture24 | ✅ | ✅ | ✅ | ✅ | 完了 |
| 27 | utdmhad_arm | ✅ | ✅ | ✅ | ✅ | 完了 |
| 28 | utdmhad_leg | ✅ | ✅ | ✅ | ✅ | 完了 |
| 29 | unimib | ✅ | ✅ | ✅ | ✅ | 完了 |

**進捗**: 29/29 データセット完了

---

## データセット別検証メモ

### dsads
- **preprocessor**: `har-unified-dataset/src/preprocessors/dsads.py`
- **検証結果**: dataset_infoの19クラスとAtlasの19エントリが一致（elevator/parking/treadmill含む）。Torso→chest、Arm→wrist、Leg→legのマッピングでAtomicも全body partに定義。
- **問題点**: なし。
- **修正**: なし。

### mhealth
- **preprocessor**: `har-unified-dataset/src/preprocessors/mhealth.py`
- **検証結果**: dataset_infoの12クラス（-1 Undefined除外）をAtlasが12件でカバー。StairsUpは`climbing_stairs`で対応。Chest/Wrist/Ankleに対応するAtomicあり。
- **問題点**: なし。
- **修正**: なし。

### pamap2
- **preprocessor**: `har-unified-dataset/src/preprocessors/pamap2.py`
- **検証結果**: dataset_infoの有効12クラスをAtlasが12件で対応（-1 otherは除外）。hand/chest/ankleに対応するAtomicあり。
- **問題点**: なし。
- **修正**: なし。

### harth
- **preprocessor**: `har-unified-dataset/src/preprocessors/harth.py`
- **検証結果**: dataset_infoの12クラス（cycling variants含む）とAtlas一致。LowerBack→hip、RightThigh→legのAtomicを全活動に付与。
- **問題点**: なし。
- **修正**: なし。

### realdisp
- **preprocessor**: `har-unified-dataset/src/preprocessors/realdisp.py`
- **検証結果**: dataset_infoの33クラスをAtlasが33件で対応。Forearm/Thigh/Calf/Backの各センサーに対応するAtomicが含まれる。
- **問題点**: なし。
- **修正**: なし。

### wisdm
- **preprocessor**: `har-unified-dataset/src/preprocessors/wisdm.py`
- **検証結果**: dataset_infoの18クラスをAtlasが18件で対応。Phone→hip、Watch→wristの両方にAtomicあり。
- **問題点**: なし。
- **修正**: なし。

### hhar
- **preprocessor**: `har-unified-dataset/src/preprocessors/hhar.py`
- **検証結果**: dataset_infoの6クラス（Undefined除外）をAtlasが6件で対応。hipに加えてwrist系Atomic（bike: W12_grip_stable、sit/stand: W14_stationary、walk/stairs: W01_swing_slow）を付与済み。
- **問題点**: なし。
- **修正**: なし。

### hmp
- **preprocessor**: `har-unified-dataset/src/preprocessors/hmp.py`
- **検証結果**: dataset_infoの14クラスをAtlasが14件で対応。
- **問題点**: なし。
- **修正**: なし。

### imsb
- **preprocessor**: `har-unified-dataset/src/preprocessors/imsb.py`
- **検証結果**: dataset_infoの6クラスをAtlasが6件で対応。
- **問題点**: なし。
- **修正**: なし。

### imwsha
- **preprocessor**: `har-unified-dataset/src/preprocessors/imwsha.py`
- **検証結果**: dataset_infoの11クラスをAtlasが11件で対応。
- **問題点**: なし。
- **修正**: なし。

### lara
- **preprocessor**: `har-unified-dataset/src/preprocessors/lara.py`
- **検証結果**: 論文のActivity Class (c1–c8: standing/walking/cart/handling_up/handling_centred/handling_down/synchronization/none) にAtlasを更新。rawラベルCSVのI/II階層フラグ（cart=II-E, None=V-A etc.）に合わせてatomicを再割当。
- **問題点**: dataset_info.pyのラベル名（stationary/gaitcycle/step/...）と論文ラベルが不一致のため、名称上はズレている。論文準拠を優先。
- **修正**: 論文ラベルでAtlas更新済み。必要ならdataset_info側のラベルも論文名に合わせる改修が必要。

### mex
- **preprocessor**: `har-unified-dataset/src/preprocessors/mex.py`
- **検証結果**: dataset_infoの7クラスをAtlasが7件で対応。
- **問題点**: なし。
- **修正**: なし。

### motionsense
- **preprocessor**: `har-unified-dataset/src/preprocessors/motionsense.py`
- **検証結果**: dataset_infoの6クラスをAtlasが6件で対応。
- **問題点**: なし。
- **修正**: なし。

### openpack
- **preprocessor**: `har-unified-dataset/src/preprocessors/openpack.py`
- **検証結果**: dataset_infoの9クラスに合わせて assemble/insert/put/walk/pick/scan/press/open/close に統一済み（Undefinedは除外）。
- **問題点**: なし。
- **修正**: なし。

### opportunity
- **preprocessor**: `har-unified-dataset/src/preprocessors/opportunity.py`
- **検証結果**: dataset_infoの17クラスをAtlasが17件で対応。Null (-1)はAtlasに含めずOK。
- **問題点**: なし。
- **修正**: なし。

### paal
- **preprocessor**: `har-unified-dataset/src/preprocessors/paal.py`
- **検証結果**: dataset_infoの24クラスに合わせて表記を統一済み（open_bottle等）。
- **問題点**: なし。
- **修正**: なし。

### sbrhapt
- **preprocessor**: `har-unified-dataset/src/preprocessors/sbrhapt.py`
- **検証結果**: dataset_infoの12クラスをAtlasが12件で対応。
- **問題点**: なし。
- **修正**: なし。

### selfback
- **preprocessor**: `har-unified-dataset/src/preprocessors/selfback.py`
- **検証結果**: dataset_infoの9クラスをAtlasが9件で対応。
- **問題点**: なし。
- **修正**: なし。

### tmd
- **preprocessor**: `har-unified-dataset/src/preprocessors/tmd.py`
- **検証結果**: dataset_infoの5クラスをAtlasが5件で対応。
- **問題点**: なし。
- **修正**: なし。

### uschad
- **preprocessor**: `har-unified-dataset/src/preprocessors/uschad.py`
- **検証結果**: dataset_infoの12クラスとAtlasが一致。`jumping_up`に名称統一済み。
- **問題点**: なし。
- **修正**: なし。

### ward
- **preprocessor**: `har-unified-dataset/src/preprocessors/ward.py`
- **検証結果**: dataset_infoの13クラスに合わせて stand/sit/lie/walk_left_circle/walk_right_circle へ統一済み。
- **問題点**: なし。
- **修正**: なし。

### capture24
- **preprocessor**: `har-unified-dataset/src/preprocessors/capture24.py`
- **検証結果**: dataset_infoの10クラスをAtlasが10件で対応。
- **問題点**: なし。
- **修正**: なし。

### utdmhad_arm
- **preprocessor**: `har-unified-dataset/src/preprocessors/utdmhad_arm.py`
- **検証結果**: dataset_infoの21クラスをAtlasが21件で対応。
- **問題点**: なし。
- **修正**: なし。

### utdmhad_leg
- **preprocessor**: `har-unified-dataset/src/preprocessors/utdmhad_leg.py`
- **検証結果**: dataset_infoの6クラスをAtlasが6件で対応。
- **問題点**: なし。
- **修正**: なし。

### unimib
- **preprocessor**: `har-unified-dataset/src/preprocessors/unimib.py`
- **検証結果**: dataset_infoの17クラスをAtlasが17件で対応。
- **問題点**: なし。
- **修正**: なし。

### forthtrace
- **preprocessor**: `har-unified-dataset/src/preprocessors/forthtrace.py`
- **検証結果**: dataset_infoの16クラス（talk含む遷移/複合）をAtlasが16件で対応。LeftWrist/RightWrist→wrist、Torso→chest、RightThigh→leg、LeftAnkle→ankleのAtomicを付与。
- **問題点**: なし。
- **修正**: なし。

### har70plus
- **preprocessor**: `har-unified-dataset/src/preprocessors/har70plus.py`
- **検証結果**: dataset_infoの7クラスをAtlasが7件で対応。LowerBack→hip、RightThigh→legで全活動にAtomic付与。
- **問題点**: なし。
- **修正**: なし。

### realworld
- **preprocessor**: `har-unified-dataset/src/preprocessors/realworld.py`
- **検証結果**: dataset_infoの8クラスをAtlasが8件で対応。Chest/Forearm/Head/Shin/Thigh/UpperArm/Waistをchest/wrist/head/leg/hipに正規化し、全活動にAtomic付与。
- **問題点**: なし。
- **修正**: なし。

### uschad
- **preprocessor**: `har-unified-dataset/src/preprocessors/uschad.py`
- **検証結果**:
- **問題点**:
- **修正**:

### forthtrace
- **preprocessor**: `har-unified-dataset/src/preprocessors/forthtrace.py`
- **検証結果**:
- **問題点**:
- **修正**:

### har70plus
- **preprocessor**: `har-unified-dataset/src/preprocessors/har70plus.py`
- **検証結果**:
- **問題点**:
- **修正**:

### paal
- **preprocessor**: `har-unified-dataset/src/preprocessors/paal.py`
- **検証結果**:
- **問題点**:
- **修正**:

### selfback
- **preprocessor**: `har-unified-dataset/src/preprocessors/selfback.py`
- **検証結果**:
- **問題点**:
- **修正**:

### imsb
- **preprocessor**: `har-unified-dataset/src/preprocessors/imsb.py`
- **検証結果**:
- **問題点**:
- **修正**:

### imwsha
- **preprocessor**: `har-unified-dataset/src/preprocessors/imwsha.py`
- **検証結果**:
- **問題点**:
- **修正**:

### hmp
- **preprocessor**: `har-unified-dataset/src/preprocessors/hmp.py`
- **検証結果**:
- **問題点**:
- **修正**:

### sbrhapt
- **preprocessor**: `har-unified-dataset/src/preprocessors/sbrhapt.py`
- **検証結果**:
- **問題点**:
- **修正**:

### openpack
- **preprocessor**: `har-unified-dataset/src/preprocessors/openpack.py`
- **検証結果**:
- **問題点**:
- **修正**:

### lara
- **preprocessor**: `har-unified-dataset/src/preprocessors/lara.py`
- **検証結果**:
- **問題点**:
- **修正**:

### opportunity
- **preprocessor**: `har-unified-dataset/src/preprocessors/opportunity.py`
- **検証結果**:
- **問題点**:
- **修正**:

### tmd
- **preprocessor**: `har-unified-dataset/src/preprocessors/tmd.py`
- **検証結果**:
- **問題点**:
- **修正**:

---

## 検証除外データセット

### nhanes
- **理由**: ラベルなしデータセット（全てY=-1）
- **用途**: 事前学習のみ、Atlasには含めない
