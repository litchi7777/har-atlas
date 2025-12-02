"""
Run Multiple Experiments with Grid Search
"""

import os
import sys
import argparse
import json
import copy
import signal
import atexit
import psutil
from pathlib import Path
from datetime import datetime
from itertools import product
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

import yaml

# Global executor reference for signal handling
_executor = None
_futures = []
_running_processes = []  # 実行中のプロセスを追跡

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def cleanup_processes():
    """
    全ての子プロセスとプロセスグループを強制終了
    """
    global _running_processes

    if _running_processes:
        print("\nCleaning up running processes...")
        for proc in _running_processes:
            try:
                if proc.is_running():
                    # プロセスグループ全体をkill（子プロセス含む）
                    children = proc.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    # 親プロセスもkill
                    proc.kill()
                    proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass
        _running_processes.clear()
        print("All processes terminated.")


def signal_handler(signum, frame):
    """
    シグナルハンドラー（Ctrl+C等）
    全ての実行中の実験を停止する
    """
    print("\n\n" + "="*80)
    print("Interrupt received (Ctrl+C). Stopping all experiments...")
    print("="*80 + "\n")

    global _executor, _futures

    # 実行中のプロセスを全て終了
    cleanup_processes()

    # 全てのfutureをキャンセル
    if _futures:
        for future in _futures:
            future.cancel()

    # Executorをシャットダウン
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=True)

    print("All experiments stopped.")
    sys.exit(0)


def get_available_gpus(memory_threshold: int = 100) -> List[int]:
    """
    利用可能なGPUのリストを取得

    Args:
        memory_threshold: 空きメモリの閾値（MB、デフォルト100MB）

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


def _get_common_prefix_suffix(strings):
    """
    文字列リストから共通の接頭辞と接尾辞を取得

    Args:
        strings: 文字列のリスト

    Returns:
        (common_prefix, common_suffix)のタプル
    """
    if not strings or len(strings) == 1:
        return "", ""

    # 共通接頭辞を見つける
    common_prefix = ""
    for chars in zip(*strings):
        if len(set(chars)) == 1:
            common_prefix += chars[0]
        else:
            break

    # 共通接尾辞を見つける
    common_suffix = ""
    for chars in zip(*[s[::-1] for s in strings]):
        if len(set(chars)) == 1:
            common_suffix = chars[0] + common_suffix
        else:
            break

    return common_prefix, common_suffix


def _remove_common_parts(value_str, all_value_strs):
    """
    複数の値から共通部分を削除して差分のみを残す

    Args:
        value_str: 処理対象の値文字列
        all_value_strs: 全ての値文字列のリスト

    Returns:
        共通部分を削除した文字列
    """
    if len(all_value_strs) <= 1:
        return value_str

    prefix, suffix = _get_common_prefix_suffix(all_value_strs)

    # 接頭辞と接尾辞を削除
    result = value_str
    if prefix and result.startswith(prefix):
        result = result[len(prefix):]
    if suffix and result.endswith(suffix):
        result = result[:-len(suffix)]

    # 空になった場合、または極端に短くなった場合は元の値を返す
    # 例: "0.001" -> "0" は情報が失われすぎるので元の値を使う
    if not result or len(result) < 2:
        return value_str

    return result


def generate_grid_experiments(base_config, grid_params):
    """
    Generate experiments from grid search parameters

    Args:
        base_config: Base configuration dict (entire config without grid_search)
        grid_params: Dictionary of parameters to grid search

    Returns:
        List of experiment configurations

    Supports two formats for grid_search values:
    1. Simple list: [value1, value2, ...] - creates cartesian product
    2. List of dicts: [{key1: v1, key2: v2}, ...] - each dict is one configuration
       This is useful for paired parameters like backbone + pretrained_path
    """
    experiments = []

    # Flatten grid parameters, handling list-of-dicts specially
    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                # Recurse into nested dicts
                items.extend(flatten_dict(v, new_key))
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                # List of dicts: treat each dict as a bundle of settings
                # Convert [{a: 1, b: 2}, {a: 3, b: 4}] to a single entry
                # that will be expanded into multiple key-value pairs
                items.append((new_key, v))  # Keep as list of dicts
            else:
                items.append((new_key, v))
        return items

    flat_params = flatten_dict(grid_params)

    param_names = []
    param_values = []

    for param_path, values in flat_params:
        param_names.append(param_path)
        param_values.append(values if isinstance(values, list) else [values])

    # 各パラメータの全ての値を収集（共通部分削除のため）
    all_value_strings = {}  # param_name -> [value_str1, value_str2, ...]
    for param_idx, values in enumerate(param_values):
        param_name = param_names[param_idx]
        value_strs = []

        for value in values:
            # 値を文字列に変換（実験名生成と同じロジック）
            if isinstance(value, str) and len(str(value)) > 30:
                if value is None or value == "null":
                    value_str = "null"
                elif "/" in value or "\\" in value:
                    path_parts = value.replace("\\", "/").split("/")
                    if path_parts[-1].endswith(".pth"):
                        parent_dir = path_parts[-3] if len(path_parts) >= 3 else ""
                        filename = Path(value).stem
                        value_str = f"{parent_dir}_{filename}" if parent_dir else filename
                    else:
                        value_str = "_".join(path_parts[-2:]) if len(path_parts) >= 2 else path_parts[-1]
                else:
                    value_str = str(value)[:30]
            else:
                value_str = str(value)

            value_strs.append(value_str)

        all_value_strings[param_name] = value_strs

    # Generate all combinations
    for idx, combination in enumerate(product(*param_values)):
        config = copy.deepcopy(base_config)

        # グリッドサーチパラメータを記録
        grid_params_dict = {}
        for param_path, value in zip(param_names, combination):
            keys = param_path.split(".")

            if isinstance(value, dict):
                # List-of-dicts case: value is a dict like {backbone: "limu_bert", pretrained_path: "..."}
                # Apply each key-value pair to the parent path
                for sub_key, sub_value in value.items():
                    full_keys = keys + [sub_key]
                    current = config
                    for key in full_keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[full_keys[-1]] = sub_value

                    # Record for logging
                    full_path = f"{param_path}.{sub_key}"
                    grid_params_dict[full_path] = sub_value
            else:
                # Simple value case
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value

                # グリッドサーチパラメータを記録（human-readable形式）
                grid_params_dict[param_path] = value

        # Use simple experiment ID
        exp_name = f"exp_{idx}"

        experiments.append({
            "name": exp_name,
            "config": config,
            "grid_params": grid_params_dict  # グリッドサーチパラメータを追加
        })

    return experiments


def run_experiment(exp_name, config, script_path, experiment_dir, gpu_id=None, run_id=None, grid_params=None):
    """
    Run a single experiment

    Args:
        exp_name: Experiment name
        config: Experiment configuration
        script_path: Path to training script
        experiment_dir: Directory to save experiment configs and results
        gpu_id: GPU ID to use (None for default)
        run_id: Grid search run ID (shared across all experiments in the same grid search)
        grid_params: Grid search parameters for this experiment (dict)

    Returns:
        Dictionary with experiment results
    """
    # script_pathからfinetuneかpretrainかを判定
    is_finetune = "finetune" in script_path

    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print(f"\n{'='*80}")
    print(f"Running experiment: {exp_name}{gpu_str}")
    print(f"{'='*80}\n")

    # Create experiment directory
    exp_dir = os.path.join(experiment_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Update config paths
    # Pretrainのみcheckpoint.save_pathを設定（finetuneは不要）
    if "checkpoint" in config and not is_finetune:
        config["checkpoint"]["save_path"] = os.path.join(exp_dir, "models")

    # Update W&B run name and group if enabled
    if "wandb" in config and config["wandb"].get("enabled", False):
        config["wandb"]["name"] = exp_name

        # グリッドサーチのrun_idをgroupとして設定
        if run_id:
            config["wandb"]["group"] = run_id
            config["wandb"]["tags"] = config["wandb"].get("tags", []) + [run_id]

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

    proc = None
    try:
        # プロセスグループを作成して、終了時に全てのサブプロセスも確実に終了させる
        # pretrainとfinetuneの両方でログファイルに保存
        global _running_processes

        with open(log_file, "w") as f:
            # Popenを使用してプロセスを追跡
            proc = subprocess.Popen(
                [sys.executable, script_path, "--config", config_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,  # 新しいセッションを開始
            )

            # psutilでプロセスをラップして追跡
            psutil_proc = psutil.Process(proc.pid)
            _running_processes.append(psutil_proc)

            # プロセスの完了を待つ
            returncode = proc.wait()

            # 完了したプロセスをリストから削除
            if psutil_proc in _running_processes:
                _running_processes.remove(psutil_proc)

            # 戻り値をチェック
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, proc.args)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "success"
        error = None

        print(f"\n✓ Experiment '{exp_name}' completed successfully")
        print(f"Duration: {duration:.2f} seconds")

        # results.jsonからメトリクスを読み取る
        metrics = {}
        results_file = os.path.join(exp_dir, "results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    metrics = json.load(f)
                print(f"Metrics loaded from {results_file}")
            except Exception as e:
                print(f"Warning: Failed to load metrics from {results_file}: {e}")

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        status = "failed"
        metrics = {}  # 失敗時はメトリクスなし

        # プロセスをクリーンアップ
        if proc:
            try:
                psutil_proc = psutil.Process(proc.pid)
                if psutil_proc in _running_processes:
                    _running_processes.remove(psutil_proc)
                # 子プロセスも終了
                children = psutil_proc.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                psutil_proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

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

    result = {
        "name": exp_name,
        "status": status,
        "duration": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_path": config_path,
        "gpu_id": gpu_id,
        "error": error,
    }

    # グリッドサーチパラメータを追加
    if grid_params:
        result["grid_params"] = grid_params

    # メトリクスが存在する場合は追加
    if metrics:
        result["metrics"] = metrics

    return result


def main(args):
    # atexitハンドラーを登録（プログラム終了時に必ずクリーンアップ）
    atexit.register(cleanup_processes)

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
    # scriptパスから実験タイプ（pretrain/finetune）を判定
    script_name = Path(args.script).stem  # "pretrain" or "finetune"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{script_name}_run_{timestamp}"  # Grid search全体を識別するID

    # finetuneの場合、pretrained_pathからpretrainの日付を抽出
    pretrain_suffix = ""
    if script_name == "finetune":
        # grid_searchからpretrained_pathを探す
        pretrained_paths = []
        if "model" in grid_search and "pretrained_path" in grid_search["model"]:
            pretrained_paths = grid_search["model"]["pretrained_path"]

        # nullではないpathを探す
        valid_paths = [p for p in pretrained_paths if p is not None and p != "null"]

        if valid_paths:
            # 最初のパスからpretrainの日付を抽出
            # 例: experiments/pretrain/run_20251104_082803/... -> 20251104_082803
            import re
            match = re.search(r'run_(\d{8}_\d{6})', valid_paths[0])
            if match:
                pretrain_date = match.group(1)
                pretrain_suffix = f"_from_{pretrain_date}"

    experiment_dir = os.path.join("experiments", script_name, f"run_{timestamp}{pretrain_suffix}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment plan (include grid_search and settings back)
    plan = {"base_config": base_config, "grid_search": grid_search, "settings": settings}
    plan_path = os.path.join(experiment_dir, "experiment_plan.yaml")
    save_yaml(plan, plan_path)
    print(f"Experiment plan saved to: {plan_path}\n")

    # GPU並列実行の設定
    parallel = settings.get("parallel", False)
    max_workers = settings.get("max_workers", None)  # Noneの場合は利用可能なGPU数 × experiments_per_gpu
    experiments_per_gpu = settings.get("experiments_per_gpu", 1)  # 1GPUあたりの同時実行数
    stop_on_error = settings.get("stop_on_error", False)
    specified_gpus = settings.get("available_gpus", None)  # 使用するGPUのリスト

    # Run experiments
    results = []

    if parallel:
        # 並列実行モード
        # 優先順位: settings.available_gpus > device > 自動検出
        if specified_gpus is not None:
            # settings.available_gpusが指定されている場合（最優先）
            if isinstance(specified_gpus, list):
                available_gpus = specified_gpus
                print(f"Using GPUs specified in settings.available_gpus: {available_gpus}")
            else:
                print(f"Warning: settings.available_gpus should be a list, got {type(specified_gpus)}. Using auto-detection.")
                available_gpus = get_available_gpus()
        else:
            # base_configのdevice設定を確認
            config_device = base_config.get("device", None)
            if config_device and config_device.startswith("cuda:"):
                # 設定ファイルで特定のGPUが指定されている場合、それを使用
                try:
                    gpu_id = int(config_device.split(":")[1])
                    available_gpus = [gpu_id]
                    print(f"Using GPU specified in device config: cuda:{gpu_id}")
                except (ValueError, IndexError):
                    # パースに失敗した場合は自動検出
                    available_gpus = get_available_gpus()
            else:
                # device指定がない場合は自動検出
                available_gpus = get_available_gpus()

        if not available_gpus:
            print("Warning: No available GPUs detected, falling back to sequential execution")
            parallel = False
        else:
            if max_workers is None:
                # max_workersが未指定の場合、GPU数 × experiments_per_gpuで計算
                max_workers = len(available_gpus) * experiments_per_gpu

            print(f"Parallel execution enabled:")
            print(f"  Available GPUs: {available_gpus}")
            print(f"  Experiments per GPU: {experiments_per_gpu}")
            print(f"  Max workers: {max_workers}\n")

    if parallel:
        # シグナルハンドラーを登録（Ctrl+C対応）
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 並列実行
        global _executor, _futures

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                _executor = executor

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
                        gpu_id,
                        run_id,  # Grid search run IDを渡す
                        exp.get("grid_params")  # Grid paramsを渡す
                    )
                    futures[future] = (i + 1, exp["name"], gpu_id)

                _futures = list(futures.keys())

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
        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt received. Cleaning up...")
            cleanup_processes()
            raise
        finally:
            # Executorが正常に終了するのを確認
            _executor = None
            _futures.clear()
    else:
        # 順次実行（既存の動作）
        for i, exp in enumerate(experiments, 1):
            print(f"\nExperiment {i}/{len(experiments)}")

            result = run_experiment(
                exp["name"],
                exp["config"],
                args.script,
                experiment_dir,
                run_id=run_id,
                grid_params=exp.get("grid_params")
            )
            results.append(result)

            # Stop if experiment failed and stop_on_error is True
            if result["status"] == "failed" and stop_on_error:
                print(f"\nStopping experiments due to failure (stop_on_error=True)")
                break

    # Save results summary
    # 軽量版: 分析に必要な最小限の情報のみ保存
    compact_results = []
    for r in results:
        compact_result = {
            "name": r["name"],
            "status": r["status"],
        }

        # Grid searchパラメータ（最重要）
        if "grid_params" in r:
            compact_result["grid_params"] = r["grid_params"]

        # メトリクス（最重要）
        if "metrics" in r:
            compact_result["metrics"] = r["metrics"]

        # エラー情報（失敗時のみ、最初の300文字）
        if r["status"] == "failed" and "error" in r:
            error_msg = r["error"]
            if len(error_msg) > 300:
                error_msg = error_msg[:300] + "..."
            compact_result["error"] = error_msg

        compact_results.append(compact_result)

    summary = {
        "timestamp": timestamp,
        "total_experiments": len(experiments),
        "completed_experiments": len(results),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "total_duration": sum(r.get("duration", 0) for r in results),
        "results": compact_results,
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

        # メトリクスの概要を表示（成功した実験のみ）
        successful_results = [r for r in results if r.get("status") == "success" and "metrics" in r]
        if successful_results:
            print(f"\n{'='*80}")
            print(f"Metrics Summary (Successful Experiments)")
            print(f"{'='*80}")

            # グリッドサーチパラメータのヘッダーを生成
            header_parts = ["Experiment"]
            param_keys = []
            # 最初の実験からgrid_paramsのキーを取得
            if successful_results[0].get("grid_params"):
                param_keys = list(successful_results[0]["grid_params"].keys())
                header_parts.extend(param_keys)
            header_parts.extend(["Val Acc", "Test Acc", "Test F1"])

            # ヘッダーを表示
            header_line = "  ".join(f"{h:<15}" for h in header_parts)
            print(header_line)
            print(f"{'-'*len(header_line)}")

            for r in successful_results:
                exp_name = r["name"]

                # グリッドサーチパラメータを表示
                line_parts = [f"{exp_name:<15}"]
                grid_params = r.get("grid_params", {})
                for key in (param_keys if param_keys else []):
                    value = grid_params.get(key, "N/A")
                    # 値を短縮表示（パスなど長い場合）
                    if isinstance(value, str) and len(value) > 15:
                        if "/" in value:
                            value = value.split("/")[-1][:15]
                        else:
                            value = value[:15]
                    line_parts.append(f"{str(value):<15}")

                # メトリクスを表示
                metrics = r.get("metrics", {})
                val_acc = metrics.get("best_val_accuracy", 0.0)
                test_acc = metrics.get("test_accuracy", 0.0)
                test_f1 = metrics.get("test_f1", 0.0)
                line_parts.extend([
                    f"{val_acc:<15.4f}",
                    f"{test_acc:<15.4f}",
                    f"{test_f1:<15.4f}"
                ])

                print("  ".join(line_parts))

        print(f"\nSummary saved to: {summary_path}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run grid search experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="Path to experiment configuration file "
        "(default: configs/pretrain.yaml)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=None,
        help="Path to training script. If not specified, auto-detect from config filename "
        "(e.g., pretrain.yaml -> src/training/pretrain.py)",
    )

    args = parser.parse_args()

    # scriptが指定されていない場合、configから自動推測
    if args.script is None:
        config_name = Path(args.config).stem  # "pretrain" or "finetune"
        args.script = f"src/training/{config_name}.py"
        print(f"Auto-detected script: {args.script}")

    main(args)
