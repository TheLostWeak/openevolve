import json
import os
import sys

# Usage: python scripts/run_test_eval.py <program_json_path> [n] [timeout_seconds]
if len(sys.argv) < 2:
    print("Usage: run_test_eval.py <program_json_path> [n] [timeout_seconds]")
    sys.exit(2)

program_json_path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TIMEOUT = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

import importlib.util

# load evaluator module by path (avoid relying on package import)
evaluator_path = os.path.join(os.path.dirname(__file__), '..', 'examples', 'cap_set_example', 'evaluator.py')
evaluator_path = os.path.normpath(evaluator_path)
spec = importlib.util.spec_from_file_location('cap_set_evaluator', evaluator_path)
cap_evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap_evaluator)

with open(program_json_path, 'r', encoding='utf-8') as f:
    pj = json.load(f)

code = pj.get('code')
if not code:
    print(json.dumps({'error': 'no code field in program json'}))
    sys.exit(3)

# write program code to a stable path next to the program json
out_dir = os.path.dirname(program_json_path)
prog_path = os.path.join(out_dir, f"test_prog_{os.path.basename(program_json_path)}.py")
with open(prog_path, 'w', encoding='utf-8') as pf:
    pf.write(code)

# Call the evaluator helper that runs generation in subprocess
try:
    res = cap_evaluator._call_generate_in_subprocess(prog_path, N, TIMEOUT)
except Exception as e:
    res = {'success': False, 'error': f'exception calling _call_generate_in_subprocess: {e}', 'traceback': None}

print(json.dumps(res, indent=2))

# exit code 0 on success, 4 on failure
if not res.get('success'):
    sys.exit(4)
else:
    sys.exit(0)
