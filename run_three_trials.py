from examples.cap_set_example.initial_program import GreedyCapSetGenerator

g = GreedyCapSetGenerator(n=8)
try:
    best_params, best_cap = g.tune_with_optuna(max_trials=3, timeout=60.0)
    print('BEST_PARAMS', best_params)
    print('CAPSET_LEN', len(best_cap))
except Exception as e:
    print('ERROR:', e)
