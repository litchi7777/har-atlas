"""
Atlas Loader for Hierarchical Activity Recognition

Atlasファイル（activity_mapping.json）を読み込み、
Activity → Atomic Motion マッピングを提供する。

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
        """
        self.atlas_path = Path(atlas_path)
        self.atlas = self._load_atlas()

        # キャッシュ
        self._all_atomic_motions: Optional[Dict[str, Set[str]]] = None
        self._activity_to_atomics: Optional[Dict[Tuple[str, str], Dict[str, List[str]]]] = None

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

        Returns:
            {"wrist": {"W01_swing_slow": 0, "W02_swing_fast": 1, ...}, ...}
        """
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
