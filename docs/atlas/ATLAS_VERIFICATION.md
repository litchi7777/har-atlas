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
|---|---------|-----------------|---------------|--------------|------------------|----------|
| 1 | chad | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 2 | dsads | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 3 | forthtrace | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 4 | har70plus | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 5 | harth | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 6 | hhar | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 7 | hmp | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 8 | imsb | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 9 | imwsha | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 10 | lara | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 11 | mex | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 12 | mhealth | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 13 | motionsense | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 14 | openpack | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 15 | opportunity | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 16 | paal | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 17 | pamap2 | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 18 | realdisp | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 19 | realworld | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 20 | sbrhapt | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 21 | selfback | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 22 | tmd | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 23 | uschad | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 24 | ward | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |
| 25 | wisdm | ⬜ | ⬜ | ⬜ | ⬜ | 未検証 |

**進捗**: 0/25 データセット完了

---

## データセット別検証メモ

### dsads
- **preprocessor**: `har-unified-dataset/src/preprocessors/dsads.py`
- **検証結果**:
- **問題点**:
- **修正**:

### mhealth
- **preprocessor**: `har-unified-dataset/src/preprocessors/mhealth.py`
- **検証結果**:
- **問題点**:
- **修正**:

### pamap2
- **preprocessor**: `har-unified-dataset/src/preprocessors/pamap2.py`
- **検証結果**:
- **問題点**:
- **修正**:

### harth
- **preprocessor**: `har-unified-dataset/src/preprocessors/harth.py`
- **検証結果**:
- **問題点**:
- **修正**:

### realdisp
- **preprocessor**: `har-unified-dataset/src/preprocessors/realdisp.py`
- **検証結果**:
- **問題点**:
- **修正**:

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
