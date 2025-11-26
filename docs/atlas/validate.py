#!/usr/bin/env python3
"""
Atlas Validation Script

atomic_motions.json (マスターデータ) と activity_mapping.json (参照データ) の
整合性をチェックするバリデーションスクリプト。

使用方法:
    python docs/atlas/validate.py           # チェックのみ
    python docs/atlas/validate.py --fix     # 自動修正も実行
    python docs/atlas/validate.py --strict  # 警告もエラーとして扱う
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional


# Body part と motion ID prefix の対応
BODY_PART_PREFIXES = {
    'head': 'HD',
    'wrist': 'W',
    'chest': 'C',
    'hip': 'H',
    'leg': 'L',
}

# Prefix間の対応関係（HとHDはmotion名が同じことがある）
# 例: H07_stationary (hip) と HD07_stationary (head) は同じ動作名だが部位が違う
PREFIX_MOTION_EQUIVALENTS = {
    # Hip -> Head equivalents (same motion name, different body part)
    ('H', 'HD'): {
        'H03_tilt': 'HD03_tilt',
        'H04_rotation': 'HD04_rotation',
        'H05_bounce_gait': 'HD05_bounce_gait',
        'H07_stationary': 'HD07_stationary',
        # MOTION_ID_FIXESで変換後のIDも対応
        'H16_stationary': 'HD07_stationary',  # H07_stationary -> H16_stationary -> HD07_stationary
        'H01_gait_slow': 'HD05_bounce_gait',  # H05_bounce_gait -> H01_gait_slow -> HD05_bounce_gait
        'H04_sway_lateral': 'HD03_tilt',      # H03_tilt -> H04_sway_lateral -> HD03_tilt
        'H14_twist': 'HD04_rotation',         # H04_rotation -> H14_twist -> HD04_rotation
    },
}


# 不正なmotion IDを正規IDにマッピング
# activity_mapping.jsonで使われている非正規IDを、atomic_motions.jsonの正規IDに変換
MOTION_ID_FIXES = {
    # Wrist fixes
    'W03_reach_up': 'W09_lift_lower',           # 上に伸ばす → lift/lower
    'W03_rotation_pronation': 'W05_rotation_discrete',  # 回内回外 → discrete rotation
    'W04_push_forward': 'W08_push_pull',        # 前に押す → push/pull
    'W04_wipe_linear': 'W17_brush_wipe',        # 直線拭き → brush/wipe
    'W05_grip_tight': 'W12_grip_stable',        # 握る → grip stable
    'W05_push_pull': 'W08_push_pull',           # 重複ID → 正規ID
    'W05_wipe_circular': 'W17_brush_wipe',      # 円形拭き → brush/wipe
    'W06_reach_extend': 'W07_reach',            # 伸ばす → reach
    'W07_lift_raise': 'W09_lift_lower',         # 持ち上げ → lift/lower
    'W07_stir_rotate': 'W04_rotation_continuous',  # かき混ぜ → continuous rotation
    'W08_chop': 'W18_cut',                      # 刻む → cut
    'W08_vibration_high': 'W11_vibration',      # 高周波振動 → vibration
    'W09_typing': 'W06_flex_extend',            # タイピング → flex/extend
    'W10_point_click': 'W13_gesture',           # クリック → gesture
    'W10_tap_press': 'W13_gesture',             # タップ → gesture
    'W11_hold_stable': 'W12_grip_stable',       # 安定保持 → grip stable
    'W11_twist': 'W05_rotation_discrete',       # ひねり → discrete rotation

    # Chest fixes
    'C02_bend_forward': 'C05_lean_transition',  # 前屈 → lean transition
    'C04_rotation_turn': 'C02_rotation_large',  # 回転 → large rotation
    'C06_lateral_tilt': 'C05_lean_transition',  # 横傾き → lean transition
    'C09_twist': 'C10_twist',                   # ひねり → twist
    'C10_rotation': 'C02_rotation_large',       # 回転 → large rotation

    # Hip fixes
    'H03_step_up': 'H07_step_up',               # IDずれ修正
    'H03_tilt': 'H04_sway_lateral',             # 傾き → sway
    'H04_rotation': 'H14_twist',                # 回転 → twist
    'H04_step_down': 'H08_step_down',           # IDずれ修正
    'H05_bounce_gait': 'H01_gait_slow',         # バウンス → gait
    'H05_sit_stand': 'H05_sway_stand',          # 立ち座り → sway stand
    'H07_stationary': 'H16_stationary',         # IDずれ修正
    'H07_vertical_osc': 'H01_gait_slow',        # 上下振動 → gait
    'H08_lateral_sway': 'H04_sway_lateral',     # 横揺れ → sway lateral
    'H12_vibration_low': 'H15_vibration_vehicle',  # 低周波振動 → vehicle vibration
    'H13_bump_impact': 'H10_jump_land',         # 衝撃 → jump land
    'H14_sway_smooth': 'H05_sway_stand',        # スムーズ揺れ → sway stand
    'H15_fall_impact': 'H10_jump_land',         # 落下衝撃 → jump land

    # Leg fixes
    'L04_sway_lateral': 'L14_weight_shift',     # 横揺れ → weight shift
    'L05_squat': 'L11_knee_flex_deep',          # スクワット → deep knee flex
    'L06_knee_flex': 'L12_knee_flex_shallow',   # 膝屈曲 → shallow knee flex
    'L06_rotation_pedal': 'L07_pedal',          # ペダル回転 → pedal
    'L09_hip_lift': 'L08_jump_explosive',       # 股関節上げ → jump
    'L10_hip_extend': 'L14_weight_shift',       # 股関節伸展 → weight shift
    'L12_squat': 'L11_knee_flex_deep',          # スクワット（重複）→ deep knee flex
    'L13_rotation': 'L16_pivot',                # 回転 → pivot
    'L15_stand_up': 'L11_knee_flex_deep',       # 立ち上がり → deep knee flex
    'L16_sit_down': 'L11_knee_flex_deep',       # 座り込み → deep knee flex
}


# Body part名の正規化マッピング
# 様々な表記を5つの正規body partに統一
BODY_PART_ALIASES = {
    # head
    'head': 'head',
    'neck': 'head',

    # wrist (腕全体を含む)
    'wrist': 'wrist',
    'forearm': 'wrist',
    'hand': 'wrist',
    'arm': 'wrist',
    'upper_arm': 'wrist',
    'lower_arm': 'wrist',
    'upperarm': 'wrist',
    'lowerarm': 'wrist',

    # chest
    'chest': 'chest',
    'torso': 'chest',

    # hip (体幹下部を含む)
    'hip': 'hip',
    'waist': 'hip',
    'back': 'hip',
    'lower_back': 'hip',
    'lowerback': 'hip',
    'pocket': 'hip',
    'phone': 'hip',
    'pax': 'hip',

    # leg (足全体を含む)
    'leg': 'leg',
    'thigh': 'leg',
    'calf': 'leg',
    'ankle': 'leg',
    'foot': 'leg',
    'toe': 'leg',
    'shin': 'leg',
}

# 正規body part一覧
CANONICAL_BODY_PARTS = {'head', 'wrist', 'chest', 'hip', 'leg'}


def load_json(path: Path) -> dict:
    """JSONファイルを読み込む"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """JSONファイルを保存"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def normalize_body_part(bp: str) -> Optional[str]:
    """Body part名を正規化"""
    bp_lower = bp.lower().replace(' ', '_').replace('-', '_')
    return BODY_PART_ALIASES.get(bp_lower)


def get_canonical_motions(atomic_motions: dict) -> Dict[str, Set[str]]:
    """atomic_motions.jsonから正規のmotion IDを取得"""
    canonical = {}
    for bp, motions in atomic_motions.get('atomic_motions', {}).items():
        canonical[bp] = set(motions.keys())
    return canonical


def validate_motion_id(motion_id: str, canonical: Dict[str, Set[str]]) -> Tuple[bool, str, Optional[str]]:
    """
    Motion IDを検証

    Returns:
        (is_valid, body_part, error_message)
    """
    # Prefix からbody partを判定
    prefix_map = {
        'HD': 'head',
        'W': 'wrist',
        'C': 'chest',
        'H': 'hip',
        'L': 'leg',
    }

    detected_bp = None
    for prefix, bp in prefix_map.items():
        if motion_id.startswith(prefix):
            # HD と H を区別
            if prefix == 'H' and motion_id.startswith('HD'):
                continue
            detected_bp = bp
            break

    if detected_bp is None:
        return False, None, f"Unknown prefix in motion ID: {motion_id}"

    if detected_bp not in canonical:
        return False, detected_bp, f"Body part '{detected_bp}' not in canonical motions"

    if motion_id not in canonical[detected_bp]:
        return False, detected_bp, f"Motion '{motion_id}' not in canonical {detected_bp} motions"

    return True, detected_bp, None


def validate_activity_mapping(
    mapping: dict,
    canonical: Dict[str, Set[str]],
    fix: bool = False
) -> Tuple[List[str], List[str], dict]:
    """
    activity_mapping.jsonを検証

    Returns:
        (errors, warnings, fixed_mapping)
    """
    errors = []
    warnings = []
    fixed_mapping = {}

    for dataset, data in mapping.items():
        if dataset in ['version', 'description', 'note']:
            fixed_mapping[dataset] = data
            continue

        fixed_dataset = {'activities': {}}
        activities = data.get('activities', {})

        for activity_name, activity_info in activities.items():
            if not isinstance(activity_info, dict):
                warnings.append(f"[{dataset}] {activity_name}: Invalid format (not a dict)")
                continue

            atomic_motions = activity_info.get('atomic_motions', {})
            level = activity_info.get('level', 1)

            fixed_activity = {'level': level, 'atomic_motions': {}}

            for bp_key, motions in atomic_motions.items():
                # Body part名を正規化
                normalized_bp = normalize_body_part(bp_key)

                if normalized_bp is None:
                    errors.append(f"[{dataset}] {activity_name}: Unknown body part '{bp_key}'")
                    continue

                if normalized_bp != bp_key:
                    warnings.append(f"[{dataset}] {activity_name}: Body part '{bp_key}' -> '{normalized_bp}'")

                # 各motionを検証
                valid_motions = []
                for motion_id in motions:
                    # 修正が必要な場合は修正後のIDを使用
                    fixed_motion_id = MOTION_ID_FIXES.get(motion_id, motion_id)
                    if fixed_motion_id != motion_id:
                        warnings.append(
                            f"[{dataset}] {activity_name}/{bp_key}: "
                            f"Fixed '{motion_id}' -> '{fixed_motion_id}'"
                        )
                        motion_id = fixed_motion_id

                    is_valid, detected_bp, error_msg = validate_motion_id(motion_id, canonical)

                    # Prefixの不一致をチェックし、同等のmotionがあれば修正
                    if is_valid and detected_bp != normalized_bp:
                        # 同等のmotionを探す
                        detected_prefix = BODY_PART_PREFIXES.get(detected_bp, '')
                        target_prefix = BODY_PART_PREFIXES.get(normalized_bp, '')
                        equiv_key = (detected_prefix, target_prefix)

                        if equiv_key in PREFIX_MOTION_EQUIVALENTS:
                            equiv_motion = PREFIX_MOTION_EQUIVALENTS[equiv_key].get(motion_id)
                            if equiv_motion and equiv_motion in canonical.get(normalized_bp, set()):
                                warnings.append(
                                    f"[{dataset}] {activity_name}/{bp_key}: "
                                    f"Prefix fix '{motion_id}' -> '{equiv_motion}'"
                                )
                                motion_id = equiv_motion
                                detected_bp = normalized_bp
                                is_valid = True

                    if not is_valid:
                        errors.append(f"[{dataset}] {activity_name}/{bp_key}: {error_msg}")
                    elif detected_bp != normalized_bp:
                        errors.append(
                            f"[{dataset}] {activity_name}: Motion '{motion_id}' has prefix for "
                            f"'{detected_bp}' but is listed under '{bp_key}'"
                        )
                    else:
                        valid_motions.append(motion_id)

                if valid_motions:
                    if normalized_bp in fixed_activity['atomic_motions']:
                        # 同じbody partが複数回出現した場合はマージ
                        fixed_activity['atomic_motions'][normalized_bp].extend(valid_motions)
                        fixed_activity['atomic_motions'][normalized_bp] = list(
                            set(fixed_activity['atomic_motions'][normalized_bp])
                        )
                    else:
                        fixed_activity['atomic_motions'][normalized_bp] = valid_motions

            if fixed_activity['atomic_motions']:
                fixed_dataset['activities'][activity_name] = fixed_activity

        if fixed_dataset['activities']:
            fixed_mapping[dataset] = fixed_dataset

    return errors, warnings, fixed_mapping


def print_summary(canonical: Dict[str, Set[str]]) -> None:
    """正規atomic motionsのサマリーを表示"""
    print("\n" + "=" * 60)
    print("Canonical Atomic Motions (from atomic_motions.json)")
    print("=" * 60)

    total = 0
    for bp in ['head', 'wrist', 'chest', 'hip', 'leg']:
        if bp in canonical:
            count = len(canonical[bp])
            total += count
            print(f"\n{bp.upper()} ({count} motions):")
            for motion in sorted(canonical[bp]):
                print(f"  - {motion}")

    print(f"\nTotal: {total} atomic motions")


def main():
    parser = argparse.ArgumentParser(description='Validate Atlas files')
    parser.add_argument('--fix', action='store_true', help='Auto-fix issues and save')
    parser.add_argument('--strict', action='store_true', help='Treat warnings as errors')
    parser.add_argument('--summary', action='store_true', help='Print canonical motions summary')
    args = parser.parse_args()

    # パスを設定
    script_dir = Path(__file__).parent
    atomic_motions_path = script_dir / 'atomic_motions.json'
    activity_mapping_path = script_dir / 'activity_mapping.json'

    # ファイル存在チェック
    if not atomic_motions_path.exists():
        print(f"ERROR: {atomic_motions_path} not found")
        sys.exit(1)

    if not activity_mapping_path.exists():
        print(f"ERROR: {activity_mapping_path} not found")
        sys.exit(1)

    # ファイル読み込み
    print("Loading files...")
    atomic_motions = load_json(atomic_motions_path)
    activity_mapping = load_json(activity_mapping_path)

    # 正規motionを取得
    canonical = get_canonical_motions(atomic_motions)

    if args.summary:
        print_summary(canonical)
        return

    # バリデーション実行
    print("\nValidating activity_mapping.json...")
    errors, warnings, fixed_mapping = validate_activity_mapping(
        activity_mapping, canonical, fix=args.fix
    )

    # 結果表示
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings[:20]:  # 最初の20件のみ
            print(f"  ⚠️  {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more warnings")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:30]:  # 最初の30件のみ
            print(f"  ❌ {e}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more errors")

    # サマリー
    print("\n" + "-" * 60)
    print(f"Errors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    # 修正保存
    if args.fix and (errors or warnings):
        print("\nSaving fixed activity_mapping.json...")
        # バックアップ
        backup_path = activity_mapping_path.with_suffix('.json.bak')
        save_json(backup_path, activity_mapping)
        print(f"  Backup: {backup_path}")

        # 修正版を保存
        save_json(activity_mapping_path, fixed_mapping)
        print("  Fixed version saved!")

    # 終了コード
    if errors or (args.strict and warnings):
        print("\n❌ Validation FAILED")
        sys.exit(1)
    else:
        print("\n✅ Validation PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()
