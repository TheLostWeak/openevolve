import asyncio
import json
import os
import yaml
from openevolve.config import EvaluatorConfig
from openevolve.evaluator import Evaluator

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
initial_program = os.path.join(BASE, 'initial_program.py')
evaluator_file = os.path.join(BASE, 'evaluator.py')
config_path = os.path.join(BASE, 'config.yaml')

with open(initial_program, 'r', encoding='utf-8') as f:
    code = f.read()

# Load evaluator config from example config.yaml if present
econf = {}
if os.path.exists(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as cf:
            cfg_all = yaml.safe_load(cf) or {}
            econf = cfg_all.get('evaluator') or {}
    except Exception:
        econf = {}

# Build EvaluatorConfig using values from config.yaml when available
timeout = int(econf.get('timeout', 900))
max_retries = int(econf.get('max_retries', 1))
cascade = bool(econf.get('cascade_evaluation', False))

cfg = EvaluatorConfig(timeout=timeout, max_retries=max_retries, cascade_evaluation=cascade, use_llm_feedback=False)

ev = Evaluator(cfg, evaluator_file)

async def run():
    metrics = await ev.evaluate_program(code, program_id='initial_program_test')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    asyncio.run(run())
