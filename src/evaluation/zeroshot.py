"""
ゼロショット行動認識評価

学習済みPrototypeを使って、未見のデータセットで行動認識を行う
Prototypeに最も近いAtomic Motionを特定し、AtlasからActivityを推定

評価方法:
1. LODO (Leave-One-Dataset-Out): 特定のデータセットを除外して学習し、そのデータセットで評価
2. Cross-location: 同じデータセット内で異なるセンサー位置でのtransfer
3. All-datasets: 全データセットで評価（学習データも含む）
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.data.hierarchical_dataset import (
    HierarchicalSSLDataset,
    collate_hierarchical,
    normalize_body_part,
)
from src.losses.hierarchical_loss import BodyPartPrototypes
from src.models.backbones import Resnet
from src.utils.atlas_loader import AtlasLoader


class ZeroshotEvaluator:
    """ゼロショット行動認識評価器"""

    def __init__(
        self,
        checkpoint_path: str,
        atlas_path: str = "docs/atlas/activity_mapping.json",
        device: str = "cuda",
    ):
        """
        Args:
            checkpoint_path: 学習済みチェックポイントのパス
            atlas_path: Atlasファイルのパス
            device: 使用デバイス
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.atlas = AtlasLoader(atlas_path)

        # モデルとPrototypeをロード
        self._load_checkpoint(checkpoint_path)

        # Atomic Motion → Activity のマッピングを構築
        self._build_atomic_to_activity_mapping()

    def _load_checkpoint(self, checkpoint_path: str):
        """チェックポイントからモデルとPrototypeをロード"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # バックボーンを作成・ロード
        self.backbone = Resnet(n_channels=3)

        # model_state_dictからバックボーン部分を抽出
        model_state = checkpoint["model_state_dict"]
        backbone_state = {}
        for key, value in model_state.items():
            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "")
                backbone_state[new_key] = value

        self.backbone.load_state_dict(backbone_state)
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # Prototypeをロード
        loss_state = checkpoint["loss_fn_state_dict"]

        # Prototype情報を抽出
        self.prototypes = {}
        self.projections = {}

        for key, value in loss_state.items():
            # hierarchical_loss.prototypes.prototypes.{body_part}
            if "hierarchical_loss.prototypes.prototypes." in key:
                body_part = key.split(".")[-1]
                self.prototypes[body_part] = value.to(self.device)

            # projection layerも必要
            if "hierarchical_loss.prototypes.projections." in key:
                parts = key.split(".")
                body_part = parts[3]
                layer_info = ".".join(parts[4:])

                if body_part not in self.projections:
                    self.projections[body_part] = {}
                self.projections[body_part][layer_info] = value.to(self.device)

        # Projection networkを再構築
        self.projection_nets = {}
        for body_part in self.prototypes.keys():
            proj = torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 128),
            ).to(self.device)

            # 重みをロード
            if body_part in self.projections:
                proj_state = self.projections[body_part]
                proj[0].weight.data = proj_state["0.weight"]
                proj[0].bias.data = proj_state["0.bias"]
                proj[2].weight.data = proj_state["2.weight"]
                proj[2].bias.data = proj_state["2.bias"]

            self.projection_nets[body_part] = proj
            proj.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Prototypes: {', '.join([f'{k}={v.shape[0]}' for k, v in self.prototypes.items()])}")

    def _build_atomic_to_activity_mapping(self):
        """Atomic Motion → Activity の逆マッピングを構築"""
        # AtlasLoaderのget_atomic_motion_to_id()と同じマッピングを使用
        # これによりPrototype index == Atomic Motion IDが保証される
        atomic_to_id = self.atlas.get_atomic_motion_to_id()

        # Atomic Motion ID → 名前（逆マッピング）
        self.atomic_id_to_name = {}
        self.atomic_name_to_idx = atomic_to_id.copy()

        for body_part, motion_to_id in atomic_to_id.items():
            self.atomic_id_to_name[body_part] = {
                idx: motion_name for motion_name, idx in motion_to_id.items()
            }

        # Activity → Atomic Motions のマッピング（Atlasから）
        # 逆マッピング: Atomic Motion → [Activity1, Activity2, ...]
        self.atomic_to_activities = defaultdict(lambda: defaultdict(list))

        for dataset in self.atlas.get_datasets():
            activities = self.atlas.get_activities(dataset)
            for activity in activities:
                atomic_motions = self.atlas.get_atomic_motions(dataset, activity)
                for body_part, motion_ids in atomic_motions.items():
                    for motion_id in motion_ids:
                        # dataset_activity形式で保存
                        self.atomic_to_activities[body_part][motion_id].append(
                            (dataset, activity)
                        )

    def get_embeddings(
        self, data: torch.Tensor, body_part: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        データからembeddingを取得し、Prototypeとの類似度を計算

        Returns:
            (embeddings, similarities): 埋め込みとPrototypeへの類似度
        """
        with torch.no_grad():
            # バックボーンで特徴抽出
            features = self.backbone(data)  # (B, 512, T) or (B, 512)
            if features.dim() == 3:
                features = features.mean(dim=2)  # Global Average Pooling

            # Body Part別のProjection
            if body_part in self.projection_nets:
                embeddings = self.projection_nets[body_part](features)
            else:
                # フォールバック: wristを使用
                embeddings = self.projection_nets["wrist"](features)

            # L2正規化
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Prototypeとの類似度
            prototypes = self.prototypes.get(body_part, self.prototypes["wrist"])
            prototypes = F.normalize(prototypes, p=2, dim=1)

            similarities = torch.mm(embeddings, prototypes.t())  # (B, num_prototypes)

        return embeddings, similarities

    def predict_activity(
        self,
        data: torch.Tensor,
        body_part: str,
        target_dataset: str,
    ) -> List[str]:
        """
        データからActivityを予測

        Args:
            data: 入力データ (B, C, T)
            body_part: Body Part
            target_dataset: 評価対象データセット

        Returns:
            予測されたActivity名のリスト
        """
        _, similarities = self.get_embeddings(data, body_part)

        # 最も類似度の高いPrototype
        best_prototype_idx = similarities.argmax(dim=1)  # (B,)

        predictions = []
        for idx in best_prototype_idx.tolist():
            # Prototype index → Atomic Motion ID
            atomic_id = self.atomic_id_to_name.get(body_part, {}).get(idx)

            if atomic_id is None:
                predictions.append("unknown")
                continue

            # Atomic Motion → Activity候補
            candidates = self.atomic_to_activities[body_part].get(atomic_id, [])

            # target_datasetのActivityを優先
            target_activities = [a for d, a in candidates if d == target_dataset]
            if target_activities:
                # 最初の候補を使用
                predictions.append(target_activities[0])
            elif candidates:
                # 他のデータセットの候補から推測
                # 最も頻出するActivity名を使用
                activity_counts = defaultdict(int)
                for _, activity in candidates:
                    activity_counts[activity] += 1
                predictions.append(max(activity_counts, key=activity_counts.get))
            else:
                predictions.append("unknown")

        return predictions

    def evaluate_dataset(
        self,
        data_root: str,
        dataset: str,
        sensor_location: str,
        samples_per_file: int = 100,
    ) -> Dict[str, Any]:
        """
        特定のデータセット・センサー位置で評価

        Returns:
            精度やconfusion matrixなどの結果
        """
        body_part = normalize_body_part(dataset, sensor_location)

        # データセットをロード
        test_dataset = HierarchicalSSLDataset(
            data_root=data_root,
            dataset_location_pairs=[[dataset, sensor_location]],
            split="test",
        )

        if len(test_dataset) == 0:
            return {"error": f"No data found for {dataset}/{sensor_location}"}

        # 評価
        all_preds = []
        all_labels = []
        all_activities = []

        for sample_idx in range(len(test_dataset)):
            sample = test_dataset[sample_idx]
            n_total = len(sample["data"])

            # ランダムにサンプリング（データがラベル順に並んでいる場合があるため）
            n_select = min(samples_per_file, n_total)
            indices = np.random.choice(n_total, n_select, replace=False)

            data = sample["data"][indices].to(self.device)  # (N, C, T)
            labels = sample["labels"][indices]

            # Activity名を取得
            activities = []
            for label in labels.tolist():
                activity = self.atlas.get_activity_name_by_label(dataset, label)
                activities.append(activity)

            # 予測
            predictions = self.predict_activity(data, body_part, dataset)

            all_preds.extend(predictions)
            all_labels.extend(labels.tolist())
            all_activities.extend(activities)

        # 精度計算
        # Activity名ベースで比較
        correct = sum(1 for p, a in zip(all_preds, all_activities) if p == a)
        accuracy = correct / len(all_preds) if all_preds else 0.0

        # 動的Activityのみの精度（静的を除外）
        static_keywords = {"sitting", "lying", "standing", "stationary"}
        dynamic_mask = [
            not any(kw in a.lower() for kw in static_keywords)
            for a in all_activities
        ]
        dynamic_preds = [p for p, m in zip(all_preds, dynamic_mask) if m]
        dynamic_activities = [a for a, m in zip(all_activities, dynamic_mask) if m]

        dynamic_correct = sum(1 for p, a in zip(dynamic_preds, dynamic_activities) if p == a)
        dynamic_accuracy = dynamic_correct / len(dynamic_preds) if dynamic_preds else 0.0

        return {
            "dataset": dataset,
            "sensor_location": sensor_location,
            "body_part": body_part,
            "total_samples": len(all_preds),
            "accuracy": accuracy,
            "dynamic_samples": len(dynamic_preds),
            "dynamic_accuracy": dynamic_accuracy,
            "unique_activities": list(set(all_activities)),
            "unique_predictions": list(set(all_preds)),
        }


def main():
    parser = argparse.ArgumentParser(description="Zeroshot Activity Recognition Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--data-root", type=str, default="har-unified-dataset/data/processed",
        help="Data root directory"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["dsads", "mhealth", "pamap2"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # 評価器を初期化
    evaluator = ZeroshotEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    # データセットごとに評価
    results = []

    # データセット・センサー位置のペアを取得（pretrain.yamlから）
    import yaml
    with open("configs/pretrain.yaml") as f:
        config = yaml.safe_load(f)

    dataset_location_pairs = config["data"]["dataset_location_pairs"]

    for dataset, location in dataset_location_pairs:
        if args.datasets and dataset not in args.datasets:
            continue

        print(f"\nEvaluating {dataset}/{location}...")
        result = evaluator.evaluate_dataset(
            data_root=args.data_root,
            dataset=dataset,
            sensor_location=location,
        )
        results.append(result)

        if "error" not in result:
            print(f"  Accuracy: {result['accuracy']:.4f} ({result['total_samples']} samples)")
            print(f"  Dynamic Accuracy: {result['dynamic_accuracy']:.4f} ({result['dynamic_samples']} samples)")

    # サマリー
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_accuracy = sum(r["accuracy"] for r in valid_results) / len(valid_results)
        avg_dynamic_accuracy = sum(r["dynamic_accuracy"] for r in valid_results) / len(valid_results)

        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Dynamic Accuracy: {avg_dynamic_accuracy:.4f}")
        print(f"Evaluated {len(valid_results)} dataset-location pairs")

    # 結果を保存
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
