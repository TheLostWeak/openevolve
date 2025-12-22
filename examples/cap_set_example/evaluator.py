"""Evaluator for cap set example.

Implements `evaluate(program_path)` so it can be used directly by OpenEvolve.
The evaluator loads the provided `initial_program.py` module, calls 
`generate_set(n)`, verifies the cap set property, and returns metrics. It also
supports an environment variable `CAP_N` to set the dimension.
"""

import importlib.util
import sys
import os
from typing import Dict, Any, Tuple, Optional

def _get_n_from_env_or_config(program_path: str) -> int:
    """Determine the F_3^n dimension `n`.

    Precedence:
      1. Environment variable `CAP_N` (if set)
      2. `config.yaml` in the same directory as `program_path` (keys: `evaluator.cap_n` or `cap_n`)
      3. Default value 4
    """
    # 1) check environment
    val = os.environ.get("CAP_N")
    if val:
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    # 2) check config.yaml next to the program
    cfg_path = os.path.join(os.path.dirname(program_path), "config.yaml")
    if os.path.exists(cfg_path):
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # look for evaluator.cap_n or cap_n
            if isinstance(cfg, dict):
                ev = cfg.get("evaluator")
                if isinstance(ev, dict) and "cap_n" in ev:
                    try:
                        return int(ev["cap_n"])
                    except (TypeError, ValueError):
                        pass
                if "cap_n" in cfg:
                    try:
                        return int(cfg["cap_n"])
                    except (TypeError, ValueError):
                        pass
        except Exception:
            # ignore parsing errors and fall back to default
            pass

    # default
    return 4

def _load_generate_set(program_path: str):
    spec = importlib.util.spec_from_file_location("cap_module", program_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load evaluation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate_set"):
        raise AttributeError("Module must provide `generate_set(n)`")
    return module.generate_set


def _is_cap_set(S: list) -> Tuple[bool, Optional[tuple]]:
    """验证集合 S 是否为合法的 Cap Set。
    Args:
        S: list of tuples, 每个tuple是F_3^n中的一个向量。
    Returns:
        (is_valid, witness): 如果非法，witness为三个违规向量 (a,b,c)。
    """
    Sset = set(S)
    if len(S) != len(Sset):
        seen = {}
        for vec in S:
            if vec in seen:
                return False, (seen[vec], vec, vec)
            seen[vec] = vec

    m = len(S)
    for i in range(m):
        a = S[i]
        for j in range(i + 1, m):
            b = S[j]
            c = tuple((-(a[k] + b[k])) % 3 for k in range(len(a)))
            if c in Sset:
                if c != a and c != b:
                    return False, (a, b, c)
    return True, None

def evaluate(program_path: str) -> Dict[str, Any]:
    """Full evaluation: verify cap set property and size.

    Returns metrics where larger `combined_score` is better. We set
    `combined_score` = size if valid, otherwise 0.0.
    """
    n = _get_n_from_env_or_config(program_path)
    try:
        gen = _load_generate_set(program_path)
        S = gen(n)

        if not isinstance(S, (list, tuple)):
            return {"combined_score": 0.0, "error": "generate_set must return list/tuple"}

        S_tuples = []
        for s in S:
            try:
                t = tuple(int(x) for x in s)
            except (TypeError, ValueError):
                return {"combined_score": 0.0, "error": "elements must be convertible to tuple of ints"}
            if len(t) != n:
                return {"combined_score": 0.0, "error": f"element length mismatch: expected {n}, got {len(t)}"}
            if any(c not in (0, 1, 2) for c in t):
                return {"combined_score": 0.0, "error": "coordinates must be in {0,1,2}"}
            S_tuples.append(t)

        valid, witness = _is_cap_set(S_tuples)
        size = len(S_tuples)

        if valid:
            return {"combined_score": float(size), "size": size, "valid": True}
        else:
            return {"combined_score": 0.0, "size": size, "valid": False, "witness": witness}

    except Exception as e:
        return {"combined_score": 0.0, "error": str(e)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("program_path")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()
    os.environ["CAP_N"] = str(args.n)
    print(evaluate(args.program_path))