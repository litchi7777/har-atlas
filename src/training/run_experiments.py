"""
Run Multiple Experiments with Grid Search
"""

import os
import sys
import argparse
import json
import copy
from pathlib import Path
from datetime import datetime
from itertools import product
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def get_available_gpus(memory_threshold: int = 10000) -> List[int]:
    """
    利用可能なGPUのリストを取得

    Args:
        memory_threshold: 空きメモリの閾値（MB）

    Returns:
        利用可能なGPUのインデックスリスト
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )

        available_gpus = []
        for line in result.stdout.strip().split("\n"):
            gpu_id, memory_free = line.split(",")
            gpu_id = int(gpu_id.strip())
            memory_free = int(memory_free.strip())

            if memory_free >= memory_threshold:
                available_gpus.append(gpu_id)

        return available_gpus

    except Exception as e:
        print(f"Warning: Could not detect GPUs: {e}")
        return []


def load_yaml(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(config, save_path):
    """Save configuration to YAML file"""
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def deep_update(base_dict, update_dict):
    """
    Recursively update nested dictionary

    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates

    Returns:
        Updated dictionary
    """
    result = copy.deepcopy(base_dict)

    for key, value in update_dict.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def generate_grid_experiments(base_config, grid_params):
    """
    Generate experiments from grid search parameters

    Args:
        base_config: Base configuration dict (entire config without grid_search)
        grid_params: Dictionary of parameters to grid search

    Returns:
        List of experiment configurations
    """
    experiments = []

    # Flatten grid parameters
    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    flat_params = flatten_dict(grid_params)

    param_names = []
    param_values = []

    for param_path, values in flat_params:
        param_names.append(param_path)
        param_values.append(values if isinstance(values, list) else [values])

    # Generate all combinations
    for combination in product(*param_values):
        config = copy.deepcopy(base_config)
        exp_name_parts = []

        for param_path, value in zip(param_names, combination):
            # Update config
            keys = param_path.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

            # Build experiment name
            exp_name_parts.append(f"{keys[-1]}={value}")

        exp_name = "_".join(exp_name_parts)
        experiments.append({"name": exp_name, "config": config})

    return experiments


def run_experiment(exp_name, config, script_path, experiment_dir, gpu_id=None):
    """
    Run a single experiment

    Args:
        exp_name: Experiment name
        config: Experiment configuration
        script_path: Path to training script
        experiment_dir: Directory to save experiment configs and results
        gpu_id: GPU ID to use (None for default)

    Returns:
        Dictionary with experiment results
    """
    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print(f"\n{'='*80}")
    print(f"Running experiment: {exp_name}{gpu_str}")
    print(f"{'='*80}\n")

    # Create experiment directory
    exp_dir = os.path.join(experiment_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Update config paths
    if "checkpoint" in config:
        config["checkpoint"]["save_path"] = os.path.join(exp_dir, "models")
    if "logging" in config:
        config["logging"]["log_dir"] = os.path.join(exp_dir, "logs")

    # Update W&B run name if enabled
    if "wandb" in config and config["wandb"].get("enabled", False):
        config["wandb"]["name"] = exp_name

    # Save experiment config
    config_path = os.path.join(exp_dir, "config.yaml")
    save_yaml(config, config_path)

    # Run training script
    start_time = datetime.now()

    # 環境変数でGPUを指定
    env = os.environ.copy()
    if gpu_id is not None:
        # CUDA_VISIBLE_DEVICESを使う場合、configのdeviceは常にcuda:0にする
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config["device"] = "cuda:0"
        # 更新したconfigを再保存
        save_yaml(config, config_path)

    # ログファイルのパス
    log_file = os.path.join(exp_dir, "experiment.log")

    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                [sys.executable, script_path, "--config", config_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
                env=env,
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success"
        error = None

        print(f"\n✓ Experiment '{exp_name}' completed successfully")
        print(f"Duration: {duration:.2f} seconds")

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "failed"

        # ログファイルから最後の50行を読み取ってエラーとして保存
        try:
            with open(log_file, "r") as f:
                log_lines = f.readlines()
                error = "".join(log_lines[-50:])
        except:
            error = str(e)

        print(f"\n✗ Experiment '{exp_name}' failed")
        print(f"Error: {error}")
        print(f"Log file: {log_file}")
        print(f"Duration: {duration:.2f} seconds")

    return {
        "name": exp_name,
        "status": status,
        "duration": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_path": config_path,
        "gpu_id": gpu_id,
        "error": error,
    }


def main(args):
    # Load configuration
    full_config = load_yaml(args.config)

    # Extract grid_search section
    if "grid_search" not in full_config:
        print("Error: 'grid_search' section not found in config")
        return

    grid_search = full_config.pop("grid_search")
    settings = full_config.pop("settings", {})

    # Base config is everything except grid_search and settings
    base_config = full_config

    # Generate grid search experiments
    experiments = generate_grid_experiments(base_config, grid_search)

    print(f"\n{'='*80}")
    print(f"Grid Search: {len(experiments)} experiments generated")
    print(f"{'='*80}\n")

    # Create experiments directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("experiments", f"run_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment plan (include grid_search and settings back)
    plan = {"base_config": base_config, "grid_search": grid_search, "settings": settings}
    plan_path = os.path.join(experiment_dir, "experiment_plan.yaml")
    save_yaml(plan, plan_path)
    print(f"Experiment plan saved to: {plan_path}\n")

    # GPU並列実行の設定
    parallel = settings.get("parallel", False)
    max_workers = settings.get("max_workers", None)  # Noneの場合は利用可能なGPU数
    stop_on_error = settings.get("stop_on_error", False)

    # Run experiments
    results = []

    if parallel:
        # 並列実行モード
        available_gpus = get_available_gpus()

        if not available_gpus:
            print("Warning: No available GPUs detected, falling back to sequential execution")
            parallel = False
        else:
            if max_workers is None:
                max_workers = len(available_gpus)
            else:
                max_workers = min(max_workers, len(available_gpus))

            print(f"Parallel execution enabled:")
            print(f"  Available GPUs: {available_gpus}")
            print(f"  Max workers: {max_workers}\n")

    if parallel:
        # 並列実行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # GPU割り当てを循環させる
            futures = {}
            for i, exp in enumerate(experiments):
                gpu_id = available_gpus[i % len(available_gpus)]
                future = executor.submit(
                    run_experiment,
                    exp["name"],
                    exp["config"],
                    args.script,
                    experiment_dir,
                    gpu_id
                )
                futures[future] = (i + 1, exp["name"], gpu_id)

            # 完了した実験から結果を収集
            for future in as_completed(futures):
                exp_num, exp_name, gpu_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"\n[{exp_num}/{len(experiments)}] Experiment '{exp_name}' on GPU {gpu_id}: {result['status']}")
                except Exception as e:
                    print(f"\n[{exp_num}/{len(experiments)}] Experiment '{exp_name}' raised an exception: {e}")
                    results.append({
                        "name": exp_name,
                        "status": "failed",
                        "error": str(e),
                        "gpu_id": gpu_id,
                    })
    else:
        # 順次実行（既存の動作）
        for i, exp in enumerate(experiments, 1):
            print(f"\nExperiment {i}/{len(experiments)}")

            result = run_experiment(exp["name"], exp["config"], args.script, experiment_dir)
            results.append(result)

            # Stop if experiment failed and stop_on_error is True
            if result["status"] == "failed" and stop_on_error:
                print(f"\nStopping experiments due to failure (stop_on_error=True)")
                break

    # Save results summary
    summary = {
        "timestamp": timestamp,
        "total_experiments": len(experiments),
        "completed_experiments": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "total_duration": sum(r["duration"] for r in results),
        "results": results,
    }

    if settings.get("save_summary", True):
        summary_path = settings.get("summary_path", os.path.join(experiment_dir, "summary.json"))
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Experiment Summary")
        print(f"{'='*80}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Completed: {summary['completed_experiments']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total duration: {summary['total_duration']:.2f} seconds")
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search experiments")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file "
        "(e.g., configs/pretrain.yaml or configs/finetune.yaml)",
    )
    parser.add_argument(
        "--script",
        type=str,
        required=True,
        help="Path to training script "
        "(e.g., src/training/pretrain.py or src/training/finetune.py)",
    )

    args = parser.parse_args()
    main(args)
