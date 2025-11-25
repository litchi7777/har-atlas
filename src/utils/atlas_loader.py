"""
Atlas Loader for Hierarchical Activity Recognition

Atlasファイル群を読み込み、以下の情報を提供する：
- activity_mapping.json: Activity → Atomic Motion マッピング
- atomic_motions.json: Atomic Motionの定義とBody Part別分類
- body_part_taxonomy.json: センサー位置の正規化マッピング

Usage:
    atlas = AtlasLoader("docs/atlas/activity_mapping.json")

    # データセットのActivity情報取得
    activities = atlas.get_activities("dsads")

    # 特定ActivityのAtomic Motions取得
    atomics = atlas.get_atomic_motions("dsads", "walking")
    # => {"wrist": ["W01_swing_slow"], "chest": ["C01_rotation_gait", ...], ...}

    # Body Part別の候補Atomic Motions取得
    wrist_atomics = atlas.get_atomic_motions_by_body_part("dsads", "walking", "wrist")
    # => ["W01_swing_slow"]

    # Body Part別のPrototype数（= Atomic Motion数）を取得
    prototype_counts = atlas.get_prototype_counts()
    # => {"wrist": 18, "hip": 16, "chest": 11, "leg": 17, "head": 7}

    # センサー位置の正規化
    normalized = atlas.normalize_body_part("dsads", "RightArm")
    # => "wrist"

    # Activity名の正規化（データセット間で統一）
    canonical = atlas.get_canonical_activity_name("dsads", "sitting")
    # => "sitting"
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class AtlasLoader:
    """Activity Atlas のローダーと管理クラス"""

    # Body Part カテゴリ（Atomic Motionのプレフィックスに対応）
    BODY_PART_PREFIXES = {
        "wrist": "W",
        "forearm": "W",  # wristと同じプレフィックス
        "hip": "H",
        "chest": "C",
        "leg": "L",
        "thigh": "L",    # legと同じプレフィックス
        "calf": "L",     # legと同じプレフィックス
        "ankle": "L",    # legと同じプレフィックス
        "head": "HD",
    }

    # 正規化されたBody Partカテゴリ（学習用）
    NORMALIZED_BODY_PARTS = ["wrist", "hip", "chest", "leg", "head"]

    def __init__(self, atlas_path: str):
        """
        Args:
            atlas_path: activity_mapping.json へのパス
                       （同じディレクトリからatomic_motions.json, body_part_taxonomy.jsonも読み込む）
        """
        self.atlas_path = Path(atlas_path)
        self.atlas_dir = self.atlas_path.parent
        self.atlas = self._load_atlas()

        # 追加のAtlasファイルを読み込み
        self.atomic_motions = self._load_atomic_motions()
        self.body_part_taxonomy = self._load_body_part_taxonomy()

        # キャッシュ
        self._all_atomic_motions: Optional[Dict[str, Set[str]]] = None
        self._activity_to_atomics: Optional[Dict[Tuple[str, str], Dict[str, List[str]]]] = None
        self._prototype_counts: Optional[Dict[str, int]] = None
        self._dataset_sensor_mapping: Optional[Dict[str, Dict[str, str]]] = None

    # メタデータキー（データセットではないトップレベルキー）
    METADATA_KEYS = {"version", "description", "note", "notes"}

    def _load_atlas(self) -> Dict:
        """Atlasファイルを読み込む（メタデータを除外）"""
        if not self.atlas_path.exists():
            raise FileNotFoundError(f"Atlas file not found: {self.atlas_path}")

        with open(self.atlas_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # メタデータを除外してデータセットのみ返す
        return {
            k: v for k, v in raw_data.items()
            if k not in self.METADATA_KEYS and isinstance(v, dict)
        }

    def _load_atomic_motions(self) -> Dict:
        """atomic_motions.jsonを読み込む"""
        atomic_path = self.atlas_dir / "atomic_motions.json"
        if not atomic_path.exists():
            return {}

        with open(atomic_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_body_part_taxonomy(self) -> Dict:
        """body_part_taxonomy.jsonを読み込む"""
        taxonomy_path = self.atlas_dir / "body_part_taxonomy.json"
        if not taxonomy_path.exists():
            return {}

        with open(taxonomy_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_datasets(self) -> List[str]:
        """登録されている全データセット名を取得"""
        return list(self.atlas.keys())

    def get_body_parts(self, dataset: str) -> List[str]:
        """データセットのBody Parts（センサー位置）を取得"""
        dataset_lower = dataset.lower()
        if dataset_lower not in self.atlas:
            raise KeyError(f"Dataset '{dataset}' not found in Atlas")
        return self.atlas[dataset_lower].get("body_parts", [])

    def get_activities(self, dataset: str) -> List[str]:
        """データセットの全Activity名を取得"""
        dataset_lower = dataset.lower()
        if dataset_lower not in self.atlas:
            raise KeyError(f"Dataset '{dataset}' not found in Atlas")
        return list(self.atlas[dataset_lower].get("activities", {}).keys())

    def get_activity_info(self, dataset: str, activity: str) -> Dict:
        """特定ActivityのフルInfo（level, atomic_motions等）を取得"""
        dataset_lower = dataset.lower()
        activity_lower = activity.lower().replace(" ", "_")

        if dataset_lower not in self.atlas:
            raise KeyError(f"Dataset '{dataset}' not found in Atlas")

        activities = self.atlas[dataset_lower].get("activities", {})
        if activity_lower not in activities:
            raise KeyError(f"Activity '{activity}' not found in dataset '{dataset}'")

        return activities[activity_lower]

    def get_atomic_motions(self, dataset: str, activity: str) -> Dict[str, List[str]]:
        """
        ActivityのAtomic Motionsを取得（Body Part別）

        Returns:
            {"wrist": ["W01_swing_slow", ...], "chest": ["C01_rotation_gait", ...], ...}
        """
        info = self.get_activity_info(dataset, activity)
        return info.get("atomic_motions", {})

    def get_atomic_motions_by_body_part(
        self,
        dataset: str,
        activity: str,
        body_part: str
    ) -> List[str]:
        """特定Body PartのAtomic Motionsを取得"""
        atomics = self.get_atomic_motions(dataset, activity)

        # Body Partの正規化（thigh -> leg, forearm -> wrist等）
        normalized_bp = self._normalize_body_part(body_part)

        # 直接マッチを試す
        if body_part in atomics:
            return atomics[body_part]

        # 正規化されたBody Partでマッチを試す
        if normalized_bp in atomics:
            return atomics[normalized_bp]

        return []

    def _normalize_body_part(self, body_part: str) -> str:
        """Body Partを正規化されたカテゴリに変換"""
        body_part_lower = body_part.lower()

        # 直接マッチ
        if body_part_lower in self.NORMALIZED_BODY_PARTS:
            return body_part_lower

        # マッピング（様々なセンサー位置名を統一カテゴリに）
        mapping = {
            # Leg variants
            "thigh": "leg",
            "calf": "leg",
            "ankle": "leg",
            "rightleg": "leg",
            "leftleg": "leg",
            "rightthigh": "leg",
            "leftthigh": "leg",
            "lowerback": "hip",  # Lower back is close to hip
            # Wrist/Arm variants
            "forearm": "wrist",
            "rightarm": "wrist",
            "leftarm": "wrist",
            "rightwrist": "wrist",
            "leftwrist": "wrist",
            "leftankle": "leg",
            "hand": "wrist",
            "watch": "wrist",
            # Chest/Torso variants
            "torso": "chest",
            "back": "chest",
            "waist": "hip",
            # Head
            "neck": "head",
            # Phone typically in pocket = hip
            "phone": "hip",
        }
        return mapping.get(body_part_lower, body_part_lower)

    def get_all_atomic_motions(self) -> Dict[str, Set[str]]:
        """
        全データセット・全Activityから全Atomic Motionsを収集

        Returns:
            {"wrist": {"W01_swing_slow", "W02_swing_fast", ...}, ...}
        """
        if self._all_atomic_motions is not None:
            return self._all_atomic_motions

        all_atomics: Dict[str, Set[str]] = defaultdict(set)

        for dataset in self.get_datasets():
            for activity in self.get_activities(dataset):
                atomics = self.get_atomic_motions(dataset, activity)
                for body_part, motion_list in atomics.items():
                    normalized_bp = self._normalize_body_part(body_part)
                    all_atomics[normalized_bp].update(motion_list)

        self._all_atomic_motions = dict(all_atomics)
        return self._all_atomic_motions

    def get_atomic_motion_to_id(self) -> Dict[str, Dict[str, int]]:
        """
        Atomic Motion名をID（整数）に変換するマッピングを生成

        atomic_motions.jsonの定義に基づいてIDを割り当て。
        ファイルが存在しない場合はactivity_mapping.jsonから集計。

        Returns:
            {"wrist": {"W01_swing_slow": 0, "W02_swing_fast": 1, ...}, ...}
        """
        # atomic_motions.jsonから取得（存在する場合）
        if self.atomic_motions and "atomic_motions" in self.atomic_motions:
            mapping = {}
            atomic_defs = self.atomic_motions["atomic_motions"]

            for body_part, motions in atomic_defs.items():
                # body_partを正規化（head, wrist, hip, chest, leg）
                normalized_bp = self._normalize_body_part(body_part)

                if normalized_bp not in mapping:
                    mapping[normalized_bp] = {}

                # 各motionにIDを割り当て
                sorted_motions = sorted(motions.keys())
                for idx, motion in enumerate(sorted_motions):
                    mapping[normalized_bp][motion] = idx

            return mapping

        # フォールバック: activity_mapping.jsonから集計
        all_atomics = self.get_all_atomic_motions()
        mapping = {}

        for body_part, motions in all_atomics.items():
            sorted_motions = sorted(motions)
            mapping[body_part] = {motion: idx for idx, motion in enumerate(sorted_motions)}

        return mapping

    def get_candidate_atomic_ids(
        self,
        dataset: str,
        activity: str,
        body_part: str,
        atomic_to_id: Optional[Dict[str, Dict[str, int]]] = None
    ) -> List[int]:
        """
        特定のActivity + Body Partに対する候補Atomic Motion IDのリストを取得
        （PiCOのPartial Label用）

        Args:
            dataset: データセット名
            activity: Activity名
            body_part: Body Part名
            atomic_to_id: Atomic Motion → ID マッピング（Noneなら自動生成）

        Returns:
            候補Atomic Motion IDのリスト
        """
        if atomic_to_id is None:
            atomic_to_id = self.get_atomic_motion_to_id()

        normalized_bp = self._normalize_body_part(body_part)
        atomics = self.get_atomic_motions_by_body_part(dataset, activity, body_part)

        if normalized_bp not in atomic_to_id:
            return []

        bp_mapping = atomic_to_id[normalized_bp]
        return [bp_mapping[a] for a in atomics if a in bp_mapping]

    def get_num_atomic_motions(self, body_part: str) -> int:
        """特定Body PartのAtomic Motion総数を取得"""
        all_atomics = self.get_all_atomic_motions()
        normalized_bp = self._normalize_body_part(body_part)
        return len(all_atomics.get(normalized_bp, set()))

    def summary(self) -> str:
        """Atlasの要約情報を文字列で返す"""
        lines = ["=== Atlas Summary ==="]
        lines.append(f"Total datasets: {len(self.get_datasets())}")

        all_atomics = self.get_all_atomic_motions()
        lines.append(f"Body Parts: {list(all_atomics.keys())}")

        for bp, motions in all_atomics.items():
            lines.append(f"  {bp}: {len(motions)} atomic motions")

        total_activities = sum(
            len(self.get_activities(ds)) for ds in self.get_datasets()
        )
        lines.append(f"Total activities: {total_activities}")

        return "\n".join(lines)

    # ========================================
    # 新規メソッド: Prototype数、Body Part正規化、Activity名正規化
    # ========================================

    def get_prototype_counts(self) -> Dict[str, int]:
        """
        Body Part別のPrototype数（= Atomic Motion数）を取得

        atomic_motions.jsonから直接読み込む。
        ファイルが存在しない場合はactivity_mapping.jsonから集計。

        Returns:
            {"wrist": 18, "hip": 16, "chest": 11, "leg": 17, "head": 7}
        """
        if self._prototype_counts is not None:
            return self._prototype_counts

        # atomic_motions.jsonから取得（存在する場合）
        if self.atomic_motions and "atomic_motions" in self.atomic_motions:
            counts = {}
            atomic_defs = self.atomic_motions["atomic_motions"]

            for body_part, motions in atomic_defs.items():
                # body_partを正規化（head, wrist, hip, chest, leg）
                normalized_bp = self._normalize_body_part(body_part)
                if normalized_bp not in counts:
                    counts[normalized_bp] = 0
                counts[normalized_bp] += len(motions)

            self._prototype_counts = counts
            return self._prototype_counts

        # フォールバック: activity_mapping.jsonから集計
        all_atomics = self.get_all_atomic_motions()
        self._prototype_counts = {bp: len(motions) for bp, motions in all_atomics.items()}
        return self._prototype_counts

    def get_dataset_sensor_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        データセット別のセンサー位置マッピングを取得

        body_part_taxonomy.jsonから読み込む。

        Returns:
            {
                "dsads": {"Torso": "chest", "RightArm": "wrist", ...},
                "mhealth": {"Chest": "chest", "LeftAnkle": "ankle", ...},
                ...
            }
        """
        if self._dataset_sensor_mapping is not None:
            return self._dataset_sensor_mapping

        if self.body_part_taxonomy and "dataset_sensor_mapping" in self.body_part_taxonomy:
            self._dataset_sensor_mapping = self.body_part_taxonomy["dataset_sensor_mapping"]
        else:
            self._dataset_sensor_mapping = {}

        return self._dataset_sensor_mapping

    # 細かいBody Partカテゴリを5つの学習用カテゴリに統合
    BODY_PART_TO_LEARNING_CATEGORY = {
        "wrist": "wrist",
        "forearm": "wrist",  # forearm → wrist
        "hip": "hip",
        "chest": "chest",
        "head": "head",
        "thigh": "leg",
        "calf": "leg",
        "ankle": "leg",
        "leg": "leg",
    }

    def normalize_body_part(self, dataset: str, sensor_location: str) -> str:
        """
        データセット固有のセンサー位置を正規化されたBody Partカテゴリに変換

        2段階で正規化:
        1. dataset_sensor_mappingでデータセット固有の名前を中間カテゴリに変換
        2. BODY_PART_TO_LEARNING_CATEGORYで5つの学習用カテゴリに統合

        Args:
            dataset: データセット名 (例: "dsads", "mhealth")
            sensor_location: センサー位置名 (例: "RightArm", "LeftAnkle")

        Returns:
            正規化されたBody Part名 (wrist, hip, chest, leg, headのいずれか)
        """
        dataset_lower = dataset.lower()
        sensor_mapping = self.get_dataset_sensor_mapping()

        # Step 1: データセット固有のマッピングで中間カテゴリを取得
        intermediate = None
        if dataset_lower in sensor_mapping:
            ds_mapping = sensor_mapping[dataset_lower]
            if sensor_location in ds_mapping:
                intermediate = ds_mapping[sensor_location]

        # フォールバック: 汎用的な正規化
        if intermediate is None:
            intermediate = self._normalize_body_part(sensor_location)

        # Step 2: 中間カテゴリを5つの学習用カテゴリに統合
        return self.BODY_PART_TO_LEARNING_CATEGORY.get(intermediate, intermediate)

    def get_canonical_activity_name(self, dataset: str, activity: str) -> str:
        """
        データセット固有のActivity名をAtlasの正規名に変換

        activity_mapping.jsonに定義されているActivity名をそのまま返す。
        見つからない場合は入力をsnake_case化して返す。

        Args:
            dataset: データセット名
            activity: Activity名（ラベルIDではなく文字列）

        Returns:
            正規化されたActivity名
        """
        dataset_lower = dataset.lower()
        activity_lower = activity.lower().replace(" ", "_").replace("-", "_")

        # Atlasに登録されているActivity一覧を取得
        try:
            registered_activities = self.get_activities(dataset_lower)

            # 完全一致
            if activity_lower in registered_activities:
                return activity_lower

            # 大文字小文字を無視してマッチ
            for reg_act in registered_activities:
                if reg_act.lower() == activity_lower:
                    return reg_act

        except KeyError:
            pass

        # マッチしない場合は正規化した入力を返す
        return activity_lower

    def get_activity_name_by_label(self, dataset: str, label_id: int) -> str:
        """
        ラベルIDからActivity名を取得

        各データセットのactivitiesの順序に基づいてマッピング。

        Args:
            dataset: データセット名
            label_id: ラベルID（整数）

        Returns:
            Activity名（見つからない場合は "unknown_{label_id}"）
        """
        dataset_lower = dataset.lower()

        try:
            activities = self.get_activities(dataset_lower)
            if 0 <= label_id < len(activities):
                return activities[label_id]
        except KeyError:
            pass

        return f"unknown_{label_id}"

    def get_all_canonical_activities(self) -> Set[str]:
        """
        全データセットの全Activity名（正規化済み）を取得

        Returns:
            Activity名のSet
        """
        all_activities = set()
        for dataset in self.get_datasets():
            all_activities.update(self.get_activities(dataset))
        return all_activities

    def get_activity_to_id(self) -> Dict[str, int]:
        """
        Activity名 → IDのマッピングを生成

        全データセットのActivityをアルファベット順にソートしてIDを割り当て。

        Returns:
            {"cycling": 0, "jumping": 1, "lying": 2, ...}
        """
        all_activities = sorted(self.get_all_canonical_activities())
        return {act: idx for idx, act in enumerate(all_activities)}


# 便利関数
def load_atlas(atlas_path: str = "docs/atlas/activity_mapping.json") -> AtlasLoader:
    """Atlasをロードするヘルパー関数"""
    return AtlasLoader(atlas_path)


if __name__ == "__main__":
    # テスト実行
    import sys

    atlas_path = sys.argv[1] if len(sys.argv) > 1 else "docs/atlas/activity_mapping.json"

    try:
        atlas = AtlasLoader(atlas_path)
        print(atlas.summary())
        print()

        # サンプルクエリ
        print("=== Sample Queries ===")
        dsads_activities = atlas.get_activities('dsads')
        print(f"DSADS activities: {dsads_activities[:5]}...")

        # 最初のActivityでテスト
        first_activity = dsads_activities[0]
        print(f"DSADS '{first_activity}' atomics: {atlas.get_atomic_motions('dsads', first_activity)}")

        # Atomic Motion ID マッピング
        atomic_to_id = atlas.get_atomic_motion_to_id()
        print(f"\nWrist atomic motions: {len(atomic_to_id.get('wrist', {}))} types")
        print(f"Leg atomic motions: {len(atomic_to_id.get('leg', {}))} types")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
