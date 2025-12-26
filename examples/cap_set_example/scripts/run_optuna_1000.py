#!/usr/bin/env python3
"""Run Optuna tuning for 1000 trials using the GreedyCapSetGenerator in best_program.py

This script loads `examples/cap_set_example/openevolve_output/best/best_program.py` (the best program
produced by the evolution run), constructs the `GreedyCapSetGenerator`, and runs an Optuna study with
1000 trials (no timeout). Each completed trial appends a row to
`examples/cap_set_example/openevolve_output/tuning_1000_results.csv` with columns:

  trial_number, value

The script prints progress to stdout as trials finish.

WARNING: This may take a very long time depending on machine speed.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import random
import csv
import time
from typing import Any

try:
    import optuna
    from optuna.samplers import TPESampler
except Exception as e:
    print("Optuna not found. Install optuna (pip install optuna) and retry.")
    raise

# Path to the best program module produced by the evolution run
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openevolve_output', 'best', 'best_program.py'))
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'openevolve_output', 'tuning_1000_results.csv'))

if not os.path.exists(MODULE_PATH):
    raise FileNotFoundError(f"best_program.py not found at {MODULE_PATH}")

# Dynamically load the module
spec = importlib.util.spec_from_file_location("best_program", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(mod)

# Expect the module to provide GreedyCapSetGenerator and _RANDOM_SEED
if not hasattr(mod, 'GreedyCapSetGenerator'):
    raise AttributeError("best_program does not define GreedyCapSetGenerator")

Gen = mod.GreedyCapSetGenerator
RSEED = getattr(mod, '_RANDOM_SEED', 42)

# Instantiate generator
generator = Gen(n=8)

# Prepare CSV output
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
with open(CSV_PATH, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['trial_number', 'value'])

# Define objective
def objective(trial: Any) -> float:
    sampled = {
        'balance_weight': trial.suggest_float('balance_weight', 0.1, 2.0),
        'diversity_weight': trial.suggest_float('diversity_weight', 0.0, 1.0),
        'position_weight': trial.suggest_float('position_weight', 0.0, 0.5),
        'random_noise': trial.suggest_float('random_noise', 0.0, 0.1),
    }
    # Use a deterministic but varying seed per trial to get reproducible variation
    seed = RSEED + (trial.number or 0) + int(time.time()) % 1  # time%1 yields 0, keep seed deterministic per run
    seed = RSEED + (trial.number or 0)
    cap = generator.generate(params=sampled, rng=random.Random(seed))
    return float(len(cap))

# Callback to record trial results as they complete
def record_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
    val = trial.value
    try:
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow([trial.number, float(val) if val is not None else ''])
    except Exception as e:
        print(f"Failed to write CSV row for trial {trial.number}: {e}")
    print(f"TRIAL {trial.number} completed -> value={val}")

# Create study and run
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RSEED))
print("Starting Optuna study: 1000 trials")
try:
    study.optimize(objective, n_trials=1000, callbacks=[record_callback])
except KeyboardInterrupt:
    print("Interrupted by user; saving progress...")

# Final summary
with open(CSV_PATH, 'a', newline='', encoding='utf-8') as fh:
    writer = csv.writer(fh)
    writer.writerow(['study_best_value', study.best_value if study.best_trial is not None else ''])
    writer.writerow(['study_best_params', study.best_params if study.best_trial is not None else ''])

print("Study complete. Best value:", study.best_value)
print("Best params:", study.best_params)
print("Results saved to:", CSV_PATH)
