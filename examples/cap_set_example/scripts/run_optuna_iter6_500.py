#!/usr/bin/env python3
"""Run 500 Optuna trials on iter6/final_code.py CapSetBuilder and save results.

Usage:
  python run_optuna_iter6_500.py

Writes JSON summary to examples/cap_set_example/exports/iter6_optuna500_<ts>.json
"""
import importlib.util
import json
import os
import time
from datetime import datetime

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exports', 'iter6', 'final_code.py'))
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exports'))

if not os.path.isfile(MODULE_PATH):
    raise SystemExit(f"Module not found: {MODULE_PATH}")

spec = importlib.util.spec_from_file_location('iter6_final', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Expect mod to define CapSetBuilder and _RANDOM_SEED
if not hasattr(mod, 'CapSetBuilder'):
    raise SystemExit('CapSetBuilder not found in module')

try:
    import optuna
except Exception as e:
    raise SystemExit('Optuna not installed in the environment: ' + str(e))

RANDOM_SEED = getattr(mod, '_RANDOM_SEED', 42)

n = 7

def objective(trial):
    params = {
        'w_nz': trial.suggest_float('w_nz', -2.0, 2.0),
        'w_ac': trial.suggest_float('w_ac', -2.0, 2.0),
        'w_ref': trial.suggest_float('w_ref', -2.0, 2.0),
        'w_c0': trial.suggest_float('w_c0', -5.0, 5.0),
        'w_sum': trial.suggest_float('w_sum', -5.0, 5.0),
        'w_imb': trial.suggest_float('w_imb', -5.0, 5.0),
        'target_sum': trial.suggest_int('target_sum', 0, 2),
        'noise': trial.suggest_float('noise', 0.0, 0.1),
    }
    builder = mod.CapSetBuilder(n, params)
    res = builder.build()
    return float(len(res))

print('Starting Optuna study (500 trials) using', MODULE_PATH)
start = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
import threading

def _save_intermediate(study_obj, trial_obj, out_dir=OUT_DIR, module_path=MODULE_PATH, n_trials=500):
    # Save a lightweight summary every 10 trials
    try:
        if (trial_obj.number + 1) % 10 == 0 or trial_obj.state.name == 'COMPLETE':
            cur = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'module_path': module_path,
                'n_trials_requested': n_trials,
                'trials_completed': trial_obj.number + 1,
                'best_value': study_obj.best_value if study_obj.best_trial is not None else None,
                'best_params': dict(study_obj.best_params) if study_obj.best_trial is not None else None,
            }
            fname = os.path.join(out_dir, f'iter6_optuna500_intermediate_{trial_obj.number+1}.json')
            with open(fname, 'w', encoding='utf-8') as fh:
                json.dump(cur, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

# Run optimization with callback that saves progress
study.optimize(objective, n_trials=500, callbacks=[_save_intermediate])
end = time.time()

best_params = study.best_params
best_value = study.best_value

# Build final set with best params
best_builder = mod.CapSetBuilder(n, best_params)
final_set = best_builder.build()
final_size = len(final_set)

os.makedirs(OUT_DIR, exist_ok=True)
summary = {
    'timestamp': datetime.utcnow().isoformat() + 'Z',
    'module_path': MODULE_PATH,
    'n_trials': 500,
    'duration_seconds': end - start,
    'best_value': best_value,
    'best_params': best_params,
    'final_size': final_size,
}

ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
out_path = os.path.join(OUT_DIR, f'iter6_optuna500_{ts}.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('Study complete')
print('Duration (s):', summary['duration_seconds'])
print('Best trial value:', best_value)
print('Final set size with best params:', final_size)
print('Best params saved to:', out_path)
