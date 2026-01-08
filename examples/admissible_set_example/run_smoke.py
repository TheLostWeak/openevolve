from evaluator import evaluate
import json
res = evaluate('initial_program.py')
print(json.dumps(res, indent=2, ensure_ascii=False))
