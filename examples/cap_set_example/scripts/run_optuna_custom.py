#!/usr/bin/env python3
"""
通用 Optuna 调参脚本

用法示例：
python run_optuna_custom.py --module_path ../exports/iter10/final_code.py --class_name GreedyCapSetGenerator --trials 1000 --csv_path ../openevolve_output/tuning_results.csv

参数：
  --module_path   目标 Python 文件路径（如 final_code.py 或 best_program.py）
  --class_name    目标类名（如 GreedyCapSetGenerator）
  --trials        调参迭代次数（默认 1000）
  --csv_path      结果 CSV 输出路径（默认 tuning_results.csv）
  --n             传递给类的 n 参数（默认 8）
"""
import importlib.util
import os
import sys
import random
import csv
import argparse
from typing import Any

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception as e:
    print("Optuna not found. Install optuna (pip install optuna) and retry.")
    raise

def main():
    parser = argparse.ArgumentParser(description="通用 Optuna 调参脚本")
    parser.add_argument('--module_path', type=str, required=True, help='目标 Python 文件路径')
    parser.add_argument('--class_name', type=str, required=True, help='目标类名')
    parser.add_argument('--trials', type=int, default=1000, help='调参迭代次数')
    parser.add_argument('--csv_path', type=str, default='tuning_results.csv', help='结果 CSV 输出路径')
    parser.add_argument('--n', type=int, default=8, help='传递给类的 n 参数')
    args = parser.parse_args()

    module_path = os.path.abspath(args.module_path)
    csv_path = os.path.abspath(args.csv_path)
    class_name = args.class_name
    n_trials = args.trials
    n_param = args.n

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"目标文件未找到: {module_path}")

    # 动态加载模块
    spec = importlib.util.spec_from_file_location("target_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(mod)

    if not hasattr(mod, class_name):
        raise AttributeError(f"{module_path} 未定义 {class_name}")

    Gen = getattr(mod, class_name)
    RSEED = getattr(mod, '_RANDOM_SEED', 42)
    generator = Gen(n=n_param)

    # 准备 CSV 输出
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['trial_number', 'value'])

    # 定义 objective
    def objective(trial: Any) -> float:
        sampled = {
            'balance_weight': trial.suggest_float('balance_weight', 0.1, 2.0),
            'diversity_weight': trial.suggest_float('diversity_weight', 0.0, 1.0),
            'position_weight': trial.suggest_float('position_weight', 0.0, 0.5),
            'random_noise': trial.suggest_float('random_noise', 0.0, 0.1),
        }
        seed = RSEED# + (trial.number or 0)
        cap = generator.generate(params=sampled, rng=random.Random(seed))
        return float(len(cap))

    # 回调函数，记录每次 trial
    def record_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        val = trial.value
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow([trial.number, float(val) if val is not None else ''])
        except Exception as e:
            print(f"Failed to write CSV row for trial {trial.number}: {e}")
        print(f"TRIAL {trial.number} completed -> value={val}")

    # 创建 study 并运行
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RSEED))
    print(f"Starting Optuna study: {n_trials} trials on {module_path} [{class_name}]")
    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[record_callback])
    except KeyboardInterrupt:
        print("Interrupted by user; saving progress...")

    # 最终结果
    with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['study_best_value', study.best_value if study.best_trial is not None else ''])
        writer.writerow(['study_best_params', study.best_params if study.best_trial is not None else ''])

    print("Study complete. Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("Results saved to:", csv_path)

if __name__ == '__main__':
    main()
